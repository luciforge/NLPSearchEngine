import matplotlib.pyplot as plt
import seaborn as sns

def plot_faiss_results(results_df):
    """
    Visualizes FAISS search results based on similarity scores.
    """
    plt.figure(figsize=(12, 8))  # Increased figure size
    sns.barplot(
        y=results_df["title"].apply(lambda x: x[:50] + '...' if len(x) > 50 else x), 
        x=results_df["similarity_score"], 
        palette="viridis"
    )
    plt.xlabel("Similarity Score", fontsize=12)
    plt.ylabel("Product Title", fontsize=12)
    plt.title("Top Retrieved Products (FAISS Similarity)", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Ensures no text overflow
    plt.show()


def plot_ranking_comparison(results_df):
    """
    Compares the ranking of products before and after re-ranking.
    """
    plt.figure(figsize=(12, 8))  # Increased figure size
    plt.plot(
        results_df.index, 
        results_df["similarity_score"], 
        label="Initial FAISS Score", 
        marker="o", 
        linestyle="dashed"
    )
    plt.plot(
        results_df.index, 
        results_df["cross_score"], 
        label="Cross-Encoder Score", 
        marker="o", 
        linestyle="dotted"
    )
    plt.plot(
        results_df.index, 
        results_df["final_score"], 
        label="Final Combined Score", 
        marker="o", 
        linestyle="solid"
    )
    
    plt.xlabel("Products", fontsize=12)
    plt.ylabel("Ranking Score", fontsize=12)
    plt.title("Ranking Comparison: FAISS vs Cross-Encoder vs Final", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_sentiment_impact(results_df):
    """
    Shows the correlation between AI scores and user ratings.
    """
    plt.figure(figsize=(12, 8))  # Increased figure size
    sns.scatterplot(
        x=results_df["rating"], 
        y=results_df["final_score"], 
        hue=results_df["title"].apply(lambda x: x[:20] + '...' if len(x) > 20 else x), 
        palette="coolwarm", 
        s=100
    )
    plt.xlabel("User Rating", fontsize=12)
    plt.ylabel("Final Ranking Score", fontsize=12)
    plt.title("Sentiment & Rating Impact on Final Score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, title="Product Title")
    plt.tight_layout()
    plt.show()


