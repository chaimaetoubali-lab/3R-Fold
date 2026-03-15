import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_purity_vs_f1(csv_path, save_path="results/plots/purity_vs_f1.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(5, 5))
    plt.scatter(df["family_purity"], df["f1"], alpha=0.6)
    plt.xlabel("Family Purity")
    plt.ylabel("F1")
    plt.title("Retrieval Purity vs F1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_similarity_vs_f1(csv_path, save_path="results/plots/similarity_vs_f1.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(5, 5))
    plt.scatter(df["avg_similarity"], df["f1"], alpha=0.6)
    plt.xlabel("Average Similarity")
    plt.ylabel("F1")
    plt.title("Average Similarity vs F1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()