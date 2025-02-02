import sqlite3, os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Default paths
DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "products.db"))
CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH", os.path.join(OUTPUT_DIR, "electronics_cleaned.parquet"))
AGGREGATED_REVIEWS_PATH = os.getenv("AGGREGATED_REVIEWS_PATH", os.path.join(OUTPUT_DIR, "electronics_reviews.parquet"))
EMBEDDINGS_METADATA_PATH = os.getenv("EMBEDDINGS_METADATA_PATH", os.path.join(MODELS_DIR, "electronics_metadata_embeddings.npy"))
EMBEDDINGS_REVIEWS_PATH = os.getenv("EMBEDDINGS_REVIEWS_PATH", os.path.join(MODELS_DIR, "electronics_reviews_embeddings.npy"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Phase 1: Data Preparation
def load_data(db_path: str) -> pd.DataFrame:
    print("Connecting to the database...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        id,
        product_id,
        title,
        description,
        category,
        price,
        brand,
        rating,
        normalized_category
    FROM subset_products
    """
    print("Executing query and loading data...")
    data = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Data loaded: {len(data)} rows.")
    return data

def load_reviews(db_path: str, output_path: str) -> pd.DataFrame:
    print("Connecting to the database...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        product_id,
        GROUP_CONCAT(text, ' ') AS aggregated_reviews
    FROM subset_reviews
    WHERE text IS NOT NULL
    GROUP BY product_id
    """
    print("Executing query and loading reviews...")
    reviews = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Reviews loaded and aggregated: {len(reviews)} products with reviews.")
    print(f"Saving aggregated reviews to {output_path}...")
    reviews.to_parquet(output_path, index=False)
    print("Aggregated reviews saved successfully.")
    return reviews

def preprocess_and_save_data(data: pd.DataFrame, output_path: str):
    print("Preprocessing the data...")
    data['description'] = data['description'].fillna('No description available')
    data['description'] = data['description'].str.replace(r'[^a-zA-Z0-9., ]', '', regex=True)
    print(f"Saving cleaned dataset to {output_path}...")
    data.to_parquet(output_path, index=False)
    print("Dataset saved successfully.")

def generate_embeddings(data: pd.DataFrame, text_column: str, model_name: str) -> np.ndarray:
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    texts = data[text_column].tolist()
    print(f"Generating embeddings for {len(texts)} items in column '{text_column}'...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings: np.ndarray, output_path: str):
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    print("Embeddings saved successfully.")

if __name__ == "__main__":

    metadata = load_data(DB_PATH)
    reviews = load_reviews(DB_PATH, AGGREGATED_REVIEWS_PATH)
    preprocess_and_save_data(metadata, CLEANED_DATA_PATH)
    metadata_embeddings = generate_embeddings(metadata, 'description', EMBEDDING_MODEL_NAME)
    save_embeddings(metadata_embeddings, EMBEDDINGS_METADATA_PATH)
    review_embeddings = generate_embeddings(reviews, 'aggregated_reviews', EMBEDDING_MODEL_NAME)
    save_embeddings(review_embeddings, EMBEDDINGS_REVIEWS_PATH)