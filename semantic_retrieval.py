import faiss, os
import numpy as np
import pandas as pd
from visualize import plot_faiss_results, plot_ranking_comparison, plot_sentiment_impact
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import spacy

# Initialize NLP tools
download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default paths
FAISS_METADATA_INDEX_PATH = os.getenv("FAISS_METADATA_INDEX_PATH", os.path.join(MODELS_DIR, "faiss_metadata.index"))
FAISS_REVIEWS_INDEX_PATH = os.getenv("FAISS_REVIEWS_INDEX_PATH", os.path.join(MODELS_DIR, "faiss_reviews.index"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CLEANED_METADATA_PATH = os.getenv("CLEANED_METADATA_PATH", os.path.join(OUTPUT_DIR, "electronics_cleaned.parquet"))
AGGREGATED_REVIEWS_PATH = os.getenv("AGGREGATED_REVIEWS_PATH", os.path.join(OUTPUT_DIR, "electronics_reviews.parquet"))

# Create directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_faiss_index(input_path: str) -> faiss.IndexFlatL2:
    print(f"Loading FAISS index from {input_path}...")
    index = faiss.read_index(input_path)
    print("FAISS index loaded successfully.")
    return index


def query_faiss_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, top_k: int = 10):
    print(f"Querying FAISS index for top {top_k} results...")
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices


def calculate_sentiment(review: str) -> float:
    sentiment = sia.polarity_scores(review)
    return sentiment['compound']


def extract_keywords_with_embeddings(query_embedding: np.ndarray, text_embeddings: np.ndarray, top_k: int = 5) -> list:
    """
    Finds the top-k most similar sentences or keywords between query and text embeddings.
    Args:
        query_embedding (np.ndarray): Embedding of the query.
        text_embeddings (np.ndarray): Embeddings of the text segments.
        top_k (int): Number of top matches to return.
    Returns:
        list: Indices of the top-k most similar embeddings.
    """
    similarities = np.dot(text_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Get top-k indices
    return top_indices


def get_matching_keywords(query: str, text_segments: list, model: SentenceTransformer, top_k: int = 5) -> list:
    """
    Finds matching keywords or phrases between the query and text using embeddings.
    Args:
        query (str): User query.
        text_segments (list): List of text segments to extract keywords from.
        model (SentenceTransformer): Pre-trained SentenceTransformer model for embeddings.
        top_k (int): Number of top matches to return.
    Returns:
        list: Top matching keywords/phrases.
    """
    query_embedding = model.encode([query])
    text_embeddings = model.encode(text_segments)
    top_indices = extract_keywords_with_embeddings(query_embedding, text_embeddings, top_k=top_k)
    return [text_segments[i] for i in top_indices]


def explain_ranking(metadata_score, review_score, rating_score, sentiment, metadata_weight, review_weight, rating_weight):
    explanation = {
        "Metadata Match": f"Weighted contribution: {metadata_weight * metadata_score:.4f}",
        "Review Relevance": f"Weighted contribution: {review_weight * review_score:.4f}",
        "Rating Contribution": f"Weighted contribution: {rating_weight * rating_score:.4f}",
        "Sentiment Impact": f"Sentiment adjustment: {-sentiment:.4f}" if sentiment else "No sentiment impact",
    }
    return explanation


def display_combined_results(metadata_indices, metadata_distances, metadata, reviews, query_embedding, review_index, query, metadata_weight=0.4, review_weight=0.3, rating_weight=0.3):
    print("\nTop Combined Results:")
    results = []

    # Normalize ratings for consistent scoring
    metadata['normalized_rating'] = (metadata['rating'] - metadata['rating'].min()) / (metadata['rating'].max() - metadata['rating'].min())

    for idx, metadata_distance in zip(metadata_indices[0], metadata_distances[0]):
        product = metadata.iloc[idx]
        product_reviews = reviews[reviews['product_id'] == product['product_id']]

        if not product_reviews.empty:
            review_texts = product_reviews['aggregated_reviews'].tolist()
            matching_keywords = get_matching_keywords(query, review_texts, model)

            review_embeddings = model.encode(review_texts)
            review_distances, _ = review_index.search(query_embedding, len(product_reviews))
            review_score = np.mean(review_distances)

            sentiment_scores = product_reviews['aggregated_reviews'].apply(calculate_sentiment)
            avg_sentiment = sentiment_scores.mean()
        else:
            review_score = 1.5
            avg_sentiment = 0
            matching_keywords = []

        rating_score = product['normalized_rating']

        # Compute combined score
        combined_score = (
            metadata_weight * metadata_distance +
            review_weight * review_score +
            rating_weight * rating_score - avg_sentiment
        )

        explanation = explain_ranking(
            metadata_distance, review_score, rating_score, avg_sentiment,
            metadata_weight, review_weight, rating_weight
        )

        results.append((product, combined_score, matching_keywords, explanation))

    results.sort(key=lambda x: x[1])

    for rank, (product, score, matching_keywords, explanation) in enumerate(results, start=1):
        print(f"Rank {rank}:")
        print(f"  Product: {product['title']}")
        print(f"  Description: {product['description']}")
        print(f"  Combined Score: {score:.4f}")
        print(f"  Matching Keywords: {', '.join(matching_keywords) if matching_keywords else 'No matching keywords'}")
        print(f"  Explanation:")
        for key, value in explanation.items():
            print(f"    {key}: {value}")
        print("-" * 50)


if __name__ == "__main__":

    TOP_K = 10
    METADATA_WEIGHT = 0.4
    REVIEW_WEIGHT = 0.3
    RATING_WEIGHT = 0.3

    print(f"Loading FAISS metadata index from {FAISS_METADATA_INDEX_PATH}...")
    metadata_index = load_faiss_index(FAISS_METADATA_INDEX_PATH)
    
    print(f"Loading FAISS reviews index from {FAISS_REVIEWS_INDEX_PATH}...")
    review_index = load_faiss_index(FAISS_REVIEWS_INDEX_PATH)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading metadata dataset from {CLEANED_METADATA_PATH}...")
    metadata = pd.read_parquet(CLEANED_METADATA_PATH)
    
    print(f"Loading aggregated reviews dataset from {AGGREGATED_REVIEWS_PATH}...")
    reviews = pd.read_parquet(AGGREGATED_REVIEWS_PATH)

    user_query = input("Enter your search query: ")
    query_embedding = model.encode([user_query])

    metadata_distances, metadata_indices = query_faiss_index(metadata_index, query_embedding, top_k=TOP_K)

    # Display results and prepare data for visualization
    results = []
    metadata['normalized_rating'] = (metadata['rating'] - metadata['rating'].min()) / (metadata['rating'].max() - metadata['rating'].min())

    for idx, metadata_distance in zip(metadata_indices[0], metadata_distances[0]):
        product = metadata.iloc[idx]
        product_reviews = reviews[reviews['product_id'] == product['product_id']]

        if not product_reviews.empty:
            review_texts = product_reviews['aggregated_reviews'].tolist()
            matching_keywords = get_matching_keywords(user_query, review_texts, model)

            review_embeddings = model.encode(review_texts)
            review_distances, _ = review_index.search(query_embedding, len(product_reviews))
            review_score = np.mean(review_distances)

            sentiment_scores = product_reviews['aggregated_reviews'].apply(calculate_sentiment)
            avg_sentiment = sentiment_scores.mean()
        else:
            review_score = 1.5
            avg_sentiment = 0
            matching_keywords = []

        rating_score = product['normalized_rating']
        combined_score = (
            METADATA_WEIGHT * metadata_distance +
            REVIEW_WEIGHT * review_score +
            RATING_WEIGHT * rating_score - avg_sentiment
        )

        explanation = explain_ranking(
            metadata_distance, review_score, rating_score, avg_sentiment,
            METADATA_WEIGHT, REVIEW_WEIGHT, RATING_WEIGHT
        )
        results.append((product, combined_score, matching_keywords, explanation))

    results.sort(key=lambda x: x[1])

    # Display combined results
    for rank, (product, score, matching_keywords, explanation) in enumerate(results, start=1):
        print(f"Rank {rank}:")
        print(f"  Product: {product['title']}")
        print(f"  Description: {product['description']}")
        print(f"  Combined Score: {score:.4f}")
        print(f"  Matching Keywords: {', '.join(matching_keywords) if matching_keywords else 'No matching keywords'}")
        print(f"  Explanation:")
        for key, value in explanation.items():
            print(f"    {key}: {value}")
        print("-" * 50)

    # Prepare data for visualization
    original_scores = [r[1] for r in results]  # Combined scores
    weighted_scores = original_scores  # Use weighted scores if available
    sentiment_scores = [r[3]["Sentiment Impact"] for r in results if "Sentiment Impact" in r[3]]

    # Visualizations
    plot_faiss_results(metadata_distances, metadata_indices, user_query)
    plot_ranking_comparison(original_scores, weighted_scores)
    plot_sentiment_impact(sentiment_scores, original_scores)
    
