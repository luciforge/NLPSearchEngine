import faiss, os
import numpy as np

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default paths
EMBEDDINGS_METADATA_PATH = os.getenv("EMBEDDINGS_METADATA_PATH", os.path.join(MODELS_DIR, "electronics_metadata_embeddings.npy"))
EMBEDDINGS_REVIEWS_PATH = os.getenv("EMBEDDINGS_REVIEWS_PATH", os.path.join(MODELS_DIR, "electronics_reviews_embeddings.npy"))
FAISS_METADATA_INDEX_PATH = os.getenv("FAISS_METADATA_INDEX_PATH", os.path.join(MODELS_DIR, "faiss_metadata.index"))
FAISS_REVIEWS_INDEX_PATH = os.getenv("FAISS_REVIEWS_INDEX_PATH", os.path.join(MODELS_DIR, "faiss_reviews.index"))

# Create directory if it does not exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Phase 2: FAISS Indexing
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index for the given embeddings.

    Args:
        embeddings (np.ndarray): The embeddings to index.

    Returns:
        faiss.IndexFlatL2: The FAISS index.
    """
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 similarity
    print(f"Adding {len(embeddings)} embeddings to the index...")
    index.add(embeddings)
    print("Index built successfully.")
    return index

def save_faiss_index(index: faiss.IndexFlatL2, output_path: str):
    """
    Saves the FAISS index to a file.

    Args:
        index (faiss.IndexFlatL2): The FAISS index.
        output_path (str): Path to save the index.
    """
    print(f"Saving FAISS index to {output_path}...")
    faiss.write_index(index, output_path)
    print("FAISS index saved successfully.")

def load_faiss_index(input_path: str) -> faiss.IndexFlatL2:
    """
    Loads a FAISS index from a file.

    Args:
        input_path (str): Path to the FAISS index file.

    Returns:
        faiss.IndexFlatL2: The loaded FAISS index.
    """
    print(f"Loading FAISS index from {input_path}...")
    index = faiss.read_index(input_path)
    print("FAISS index loaded successfully.")
    return index

if __name__ == "__main__":

    print("Loading metadata embeddings...")
    metadata_embeddings = np.load(EMBEDDINGS_METADATA_PATH)
    print("Loading review embeddings...")
    review_embeddings = np.load(EMBEDDINGS_REVIEWS_PATH)

    metadata_index = build_faiss_index(metadata_embeddings)
    review_index = build_faiss_index(review_embeddings)

    save_faiss_index(metadata_index, FAISS_METADATA_INDEX_PATH)
    save_faiss_index(review_index, FAISS_REVIEWS_INDEX_PATH)
