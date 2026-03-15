import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hybrid_family_confusion(
    csv_path="results/dsr_faiss_hybrid_results.csv",
    save_path="results/plots/hybrid_retrieval_family_confusion.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():

        query_family = row["family"]

        neighbors = str(row["neighbor_families"]).split("|")

        for nf in neighbors:
            if nf:
                rows.append({
                    "query_family": query_family,
                    "neighbor_family": nf
                })

    pair_df = pd.DataFrame(rows)

    mat = pd.crosstab(
        pair_df["query_family"],
        pair_df["neighbor_family"]
    )

    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(9,7))

    sns.heatmap(
        mat,
        cmap="magma",
        annot=True,
        fmt=".2f"
    )

    plt.xlabel("Retrieved neighbor family")
    plt.ylabel("Query family")

    plt.title("Hybrid Retrieval Family Confusion Matrix (DSR + FAISS)")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.show()