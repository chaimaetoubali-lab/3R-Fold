import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_family_f1(csv_path, save_path="results/plots/f1_by_family.png", top_n=20):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    summary = (
        df.groupby("family")
        .agg(f1_mean=("f1", "mean"), n=("query_id", "count"))
        .reset_index()
        .sort_values("f1_mean", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(summary["family"], summary["f1_mean"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Mean F1")
    plt.title("Top Families by F1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()