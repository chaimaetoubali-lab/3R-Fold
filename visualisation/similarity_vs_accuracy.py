import pandas as pd
import matplotlib.pyplot as plt


def plot_similarity_vs_accuracy(csv="results/ablation_results.csv"):

    df = pd.read_csv(csv)

    plt.scatter(
        df["avg_similarity"],
        df["f1"],
        alpha=0.6
    )

    plt.xlabel("Embedding similarity")
    plt.ylabel("F1 score")

    plt.title("Similarity vs Folding Accuracy")

    plt.show()