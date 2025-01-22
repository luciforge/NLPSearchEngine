import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_faiss_results(distances, indices, query):
    """
    Visualizes the FAISS search results by plotting the distances of top results.

    Args:
        distances (np.ndarray): Distances returned by FAISS search.
        indices (np.ndarray): Indices of the matching items.
        query (str): The search query.
    """
    top_k = len(distances[0])
    plt.figure(figsize=(10, 5))
    plt.bar(range(top_k), distances[0], color='skyblue')

    # Add labels to the bars
    for i, val in enumerate(distances[0]):
        plt.text(i, val + 0.01, f"{val:.2f}", ha='center')

    plt.xlabel("Top-K Matches")
    plt.ylabel("Distance")
    plt.title(f"FAISS Search Results for Query: '{query}'")
    plt.legend(["Lower distance = better match"], loc='upper left')
    plt.show()


def plot_ranking_comparison(original_scores, weighted_scores):
    """
    Compares original scores and weighted scores using a bar plot.

    Args:
        original_scores (list): Original ranking scores before weighting.
        weighted_scores (list): Ranking scores after applying weights.
    """
    indices = range(len(original_scores))
    plt.figure(figsize=(12, 6))
    width = 0.4

    plt.bar(indices, original_scores, width, label='Original Scores', alpha=0.7, color='blue')
    plt.bar([i + width for i in indices], weighted_scores, width, label='Weighted Scores', alpha=0.7, color='green')

    # Add exact score values on top of the bars
    for i, (orig, weight) in enumerate(zip(original_scores, weighted_scores)):
        plt.text(i, orig + 0.01, f"{orig:.4f}", ha='center', color='blue')
        plt.text(i + width, weight + 0.01, f"{weight:.4f}", ha='center', color='green')

    plt.xlabel("Items")
    plt.ylabel("Scores")
    plt.title("Ranking Comparison: Original vs Weighted Scores")
    plt.xticks(indices, [f"Item {i+1}" for i in indices], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sentiment_impact(sentiment_scores, ranking_scores):
    """
    Visualizes the impact of sentiment on ranking scores.

    Args:
        sentiment_scores (list): Sentiment scores for each item.
        ranking_scores (list): Final ranking scores after sentiment adjustment.
    """
    try:
        # Extract numeric values from strings or filter out invalid entries
        cleaned_sentiment_scores = []
        cleaned_ranking_scores = []

        for sent, rank in zip(sentiment_scores, ranking_scores):
            if isinstance(sent, str) and "Sentiment adjustment" in sent:
                # Extract numeric part from "Sentiment adjustment: -0.9970"
                cleaned_sentiment_scores.append(float(sent.split(":")[-1]))
                cleaned_ranking_scores.append(rank)
            elif isinstance(sent, (int, float)):
                # Include valid numeric values directly
                cleaned_sentiment_scores.append(float(sent))
                cleaned_ranking_scores.append(rank)

        # Ensure both arrays are numeric
        sentiment_array = np.array(cleaned_sentiment_scores, dtype=float).reshape(-1, 1)
        ranking_array = np.array(cleaned_ranking_scores, dtype=float)

        # Add regression line
        reg = LinearRegression().fit(sentiment_array, ranking_array)
        trend_line = reg.predict(sentiment_array)

        # Create scatter plot
        plt.figure(figsize=(12, 6))
        plt.scatter(cleaned_sentiment_scores, cleaned_ranking_scores, c=cleaned_ranking_scores, cmap='viridis', s=100, alpha=0.7, edgecolors='k', label='Data Points')
        plt.plot(cleaned_sentiment_scores, trend_line, color='red', label='Trend Line')

        # Annotate key data points
        for i, (sent, rank) in enumerate(zip(cleaned_sentiment_scores, cleaned_ranking_scores)):
            if abs(sent) > 0.2:  # Threshold for labeling outliers
                plt.annotate(f"Item {i+1}\n({sent:.2f}, {rank:.2f})", (sent, rank), textcoords="offset points", xytext=(5, 5), ha='center')

        plt.xlabel("Sentiment Scores")
        plt.ylabel("Ranking Scores")
        plt.title("Sentiment Impact on Rankings")
        plt.colorbar(label='Ranking Score Intensity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print(f"Error in sentiment impact visualization: {e}")
        print("Ensure sentiment and ranking scores are numeric.")
