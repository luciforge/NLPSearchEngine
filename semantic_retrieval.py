import faiss, os
import numpy as np
import pandas as pd
from visualize import plot_faiss_results, plot_ranking_comparison, plot_sentiment_impact
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import spacy, requests
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Load Gemini API Key from environment variable
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable is not set.")
genai.configure(api_key=GENAI_API_KEY)

# Initialize NLP tools
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Configure paths
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default paths
FAISS_METADATA_INDEX_PATH = os.getenv("FAISS_METADATA_INDEX_PATH", os.path.join(MODELS_DIR, "faiss_metadata.index"))
FAISS_REVIEWS_INDEX_PATH = os.getenv("FAISS_REVIEWS_INDEX_PATH", os.path.join(MODELS_DIR, "faiss_reviews.index"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
CLEANED_METADATA_PATH = os.getenv("CLEANED_METADATA_PATH", os.path.join(OUTPUT_DIR, "electronics_cleaned.parquet"))
AGGREGATED_REVIEWS_PATH = os.getenv("AGGREGATED_REVIEWS_PATH", os.path.join(OUTPUT_DIR, "electronics_reviews.parquet"))

# Create directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Caching for API calls
response_cache = {}

# Function to extract price constraints from query
def extract_price_constraints(query):
    min_price, max_price = None, None
    doc = nlp(query)

    price_tokens = [int(token.text) for token in doc if token.like_num]
    
    for i, token in enumerate(doc):
        prev_token = doc[i - 1].text.lower() if i > 0 else ""
        
        if prev_token in ["under", "below", "cheaper", "within"]:
            max_price = price_tokens.pop(0) if price_tokens else None
        elif prev_token in ["above", "over", "minimum"]:
            min_price = price_tokens.pop(0) if price_tokens else None
        elif prev_token == "between" and i+2 < len(doc):
            min_price = int(doc[i+1].text) if doc[i+1].like_num else None
            max_price = int(doc[i+3].text) if doc[i+3].like_num else None

    return {"min_price": min_price, "max_price": max_price}

def load_faiss_index(input_path: str) -> faiss.IndexFlatL2:
    print(f"Loading FAISS index from {input_path}...")
    index = faiss.read_index(input_path)
    print("FAISS index loaded successfully.")
    return index

# Function to query FAISS with price-based filtering
def query_faiss_index(index, query_embedding, metadata, model, price_constraints, top_k=10):
    """
    Queries FAISS index with pre-filtering based on extracted price constraints.
    
    Args:
        index (faiss.Index): FAISS index to search.
        query_embedding (np.ndarray): The embedding of the search query.
        metadata (pd.DataFrame): The metadata containing product details.
        model (SentenceTransformer): Embedding model for similarity scoring.
        price_constraints (dict): Extracted price constraints with keys "min_price" and "max_price".
        top_k (int): Number of top results to retrieve.

    Returns:
        pd.DataFrame: Ranked and filtered results.
    """
    print(f"Querying FAISS index for top {top_k} results...")

    # Ensure the query embedding has the correct shape (1, embedding_dim)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Search FAISS for top_k nearest neighbors
    distances, indices = index.search(query_embedding, top_k)

    # Handle case where no results are found
    if indices.shape[1] == 0 or np.all(indices == -1):
        print("No relevant products found.")
        return pd.DataFrame()  # Return an empty DataFrame if no products are found

    retrieved_products = metadata.iloc[indices[0]].copy()

    # Filter products based on extracted price constraints
    if price_constraints["min_price"] is not None:
        retrieved_products = retrieved_products[retrieved_products["price"] >= price_constraints["min_price"]]
    if price_constraints["max_price"] is not None:
        retrieved_products = retrieved_products[retrieved_products["price"] <= price_constraints["max_price"]]

    # If no products remain after filtering, return an empty DataFrame
    if retrieved_products.empty:
        print("No products matched the price constraints.")
        return pd.DataFrame()

    # Encode combined title and description for similarity calculation
    combined_texts = (retrieved_products['title'] + " " + retrieved_products['description']).tolist()
    combined_embeddings = model.encode(combined_texts, convert_to_numpy=True)

    # Compute similarity scores
    similarities = cosine_similarity(query_embedding, combined_embeddings).flatten()

    # Add similarity scores to dataframe
    retrieved_products["similarity_score"] = similarities

    # Sort results by similarity score in descending order
    retrieved_products = retrieved_products.sort_values(by="similarity_score", ascending=False)

    return retrieved_products


def re_rank_candidates(query, retrieved_products, cross_encoder_model):
    """
    Re-ranks the retrieved candidates using a cross-encoder model,
    and then combines the cross-encoder score with a normalized rating.
    """
    # Combine title, description, and rating (with context) into a single text string.
    candidate_texts = (
        retrieved_products['title'].astype(str)
        + " " + retrieved_products['description'].astype(str)
        + " Rating: " + retrieved_products['rating'].astype(str) + " out of 5"
    ).tolist()
    
    # Prepare a list of (query, candidate_text) tuples for the cross-encoder.
    pairs = [(query, text) for text in candidate_texts]
    
    # Get cross-encoder relevance scores.
    cross_scores = cross_encoder_model.predict(pairs)
    retrieved_products['cross_score'] = cross_scores

    # Normalize the rating (assuming ratings are out of 5).
    retrieved_products['normalized_rating'] = retrieved_products['rating'] / 5.0

    # Normalize cross_score using min-max scaling:
    min_score = retrieved_products['cross_score'].min()
    max_score = retrieved_products['cross_score'].max()
    retrieved_products['normalized_cross'] = (retrieved_products['cross_score'] - min_score) / (max_score - min_score)

    # Combine the cross-encoder score and normalized rating with a weighted sum.
    # Adjust alpha and beta to control the influence of the semantic match vs. the rating.
    alpha = 0.7  # weight for the cross-encoder score
    beta = 0.3   # weight for the rating score
    retrieved_products['final_score'] = alpha * retrieved_products['cross_score'] + beta * retrieved_products['normalized_rating']

    # Sort the products based on the final combined score.
    retrieved_products = retrieved_products.sort_values(by='final_score', ascending=False)
    print(retrieved_products[['title', 'rating', 'cross_score', 'normalized_rating', 'final_score']])
    return retrieved_products

def query_gemini_rag(query: str, retrieved_context: list) -> str:
    """
    Queries Gemini AI API using retrieved product metadata and reviews.

    Args:
        query (str): User's search query.
        retrieved_context (list): List of retrieved metadata and reviews.

    Returns:
        str: AI-generated response.
    """

    if query in response_cache:
        return response_cache[query]  # Return cached response if available

    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GENAI_API_KEY}"

    # Improved AI prompt to generate useful insights
    formatted_context = "\n".join([f"- {p}" for p in retrieved_context[:5]])

    prompt = f"""
    You are an AI product assistant for an e-commerce store.

    **User Question:** "{query}"

    **Relevant Product Details & Reviews:**
    {formatted_context}

    **Your Task:**
    - Summarize the key features of the best product.
    - Explain what users liked most and disliked.
    - Provide a clear recommendation based on available data.

    Keep your response **short and informative**.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response_data = response.json()

        if "candidates" in response_data:
            gemini_response = response_data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            if gemini_response:
                response_cache[query] = gemini_response  # Cache response
                return gemini_response

        return "Sorry, Gemini AI could not generate a response."
    
    except Exception as e:
        print(f"Error querying Gemini AI: {e}")
        return "Sorry, I couldn't process this request."

def summarize_reviews_with_gemini(reviews):
    """
    Summarizes reviews into positive and negative points using Gemini AI.

    Args:
        reviews (list): List of review texts.

    Returns:
        dict: Summarized positive and negative points.
    """
    if not reviews:
        return {"positive": ["No reviews available"], "negative": ["No reviews available"]}

    reviews_text = "\n".join(reviews[:50])  # Limit to top 50 reviews to avoid exceeding token limits.

    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GENAI_API_KEY}"

    prompt = f"""
    Based on the following product reviews, extract exactly **three positive** and **three negative** key points.

    Reviews:
    {reviews_text}

    **Output Format (Must be in this exact structure):**
    Positive:
    - [Point 1]
    - [Point 2]
    - [Point 3]

    Negative:
    - [Point 1]
    - [Point 2]
    - [Point 3]
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response_data = response.json()

        if "candidates" in response_data:
            gemini_response = response_data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

            if gemini_response:
                # Parse the response into structured data
                summary = {"positive": [], "negative": []}
                if "Negative:" in gemini_response:
                    positive_part, negative_part = gemini_response.split("Negative:")
                    positive_part = positive_part.replace("Positive:", "").strip()
                    summary["positive"] = [line.strip("- ") for line in positive_part.split("\n") if line.strip()]
                    summary["negative"] = [line.strip("- ") for line in negative_part.split("\n") if line.strip()]
                return summary

        print(f"Gemini API Error: {response.status_code} - {response.text}")
        return {"positive": ["Could not generate summary"], "negative": ["Could not generate summary"]}

    except Exception as e:
        print(f"Error summarizing reviews with Gemini AI: {e}")
        return {"positive": ["Error fetching review summary"], "negative": ["Error fetching review summary"]}

# Display results with price and reviews
def display_combined_results(retrieved_products, metadata, reviews, query_embedding, review_index, query):
    """
    Displays retrieved results with metadata, price, and summarized reviews.
    """
    print("\nTop Combined Results:")
    retrieved_context = []
    results = []

    for _, product in retrieved_products.iterrows():
        product_reviews = reviews[reviews["product_id"] == product["product_id"]]
        review_texts = product_reviews["aggregated_reviews"].tolist() if not product_reviews.empty else ["No reviews available"]

        formatted_product_info = f"- {product['title']} | Price: ${product['price']} | Rating: {product['rating']} | {product['description']}"
        retrieved_context.append(formatted_product_info)
        retrieved_context.extend(review_texts[:5])

        review_summary = summarize_reviews_with_gemini(review_texts)

        results.append({
            "title": product["title"],
            "description": product["description"],
            "rating": product["rating"],
            "price": product["price"],
            "review_summary": review_summary,
            "similarity_score": product["similarity_score"],
            "final_score": product["final_score"],
            "cross_score": product["cross_score"]
        })

    gemini_response = query_gemini_rag(query, retrieved_context)

    # Convert results to DataFrame for visualization
    results_df = pd.DataFrame(results)

    for rank, product in enumerate(results, start=1):
        print(f"Rank {rank}: {product['title']} - Price: ${product['price']} - Rating: {product['rating']}")
        print(f"  Description: {product['description']}")
        print(f"  Reviews (Summarized):")

        print(f"    Positive:")
        for point in product["review_summary"]["positive"]:
            print(f"    - {point}")

        print(f"    Negative:")
        for point in product["review_summary"]["negative"]:
            print(f"    - {point}")
        print("-" * 50)

    print("\n💡 **AI-Powered Summary from Gemini:**\n")
    print("--------------------------------------------------")
    print(gemini_response)
    print("--------------------------------------------------")

    if not results_df.empty:
        # Plot FAISS search results
        plot_faiss_results(results_df)

        # Compare rankings before and after cross-encoder re-ranking
        plot_ranking_comparison(results_df)

        # Show impact of sentiment and rating on ranking
        plot_sentiment_impact(results_df)

if __name__ == "__main__":

    TOP_K = 10

    print(f"Loading FAISS metadata index from {FAISS_METADATA_INDEX_PATH}...")
    metadata_index = load_faiss_index(FAISS_METADATA_INDEX_PATH)
    
    print(f"Loading FAISS reviews index from {FAISS_REVIEWS_INDEX_PATH}...")
    review_index = load_faiss_index(FAISS_REVIEWS_INDEX_PATH)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Initialize cross-encoder for the second stage re-ranking
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print(f"Loading metadata dataset from {CLEANED_METADATA_PATH}...")
    metadata = pd.read_parquet(CLEANED_METADATA_PATH)
    
    print(f"Loading aggregated reviews dataset from {AGGREGATED_REVIEWS_PATH}...")
    reviews = pd.read_parquet(AGGREGATED_REVIEWS_PATH)

    user_query = input("Enter your search query: ")
    query_embedding = model.encode([user_query])

    # Extract price constraints and query the FAISS index (dense retrieval stage)
    price_constraints = extract_price_constraints(user_query)
    retrieved_products = query_faiss_index(metadata_index, query_embedding, metadata, model, price_constraints, top_k=TOP_K)

    if retrieved_products.empty:
        print("No relevant products found.")
    else:
        # Re-rank the initial candidates using the cross-encoder (second stage re-ranking)
        retrieved_products = re_rank_candidates(user_query, retrieved_products, cross_encoder_model)
        display_combined_results(retrieved_products, metadata, reviews, query_embedding, review_index, user_query)
