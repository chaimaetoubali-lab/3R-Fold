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

    # Debug: Print unique families and their counts
    print("Unique query families:")
    print(df["family"].value_counts())
    print("\nSample neighbor_families entries:")
    for i, nf in enumerate(df["neighbor_families"].head(5)):
        print(f"{i}: {nf[:100]}...")  # First 100 chars

    rows = []

    for _, row in df.iterrows():

        query_family = row["family"]

        neighbors = str(row["neighbor_families"]).split("|")

        for nf in neighbors:
            if nf.strip():  # Add strip() to handle empty strings
                rows.append({
                    "query_family": query_family,
                    "neighbor_family": nf.strip()
                })

    pair_df = pd.DataFrame(rows)

    # Debug: Print neighbor family distribution
    print("\nNeighbor family distribution:")
    print(pair_df["neighbor_family"].value_counts())

    mat = pd.crosstab(
        pair_df["query_family"],
        pair_df["neighbor_family"]
    )

    print(f"\nConfusion matrix shape: {mat.shape}")
    print("Matrix families (rows):", mat.index.tolist())
    print("Matrix families (cols):", mat.columns.tolist())

    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(12,8))  # Increased size for better visibility

    sns.heatmap(
        mat,
        cmap="magma",
        annot=True,
        fmt=".2f",
        linewidths=.5
    )

    plt.xlabel("Retrieved neighbor family")
    plt.ylabel("Query family")

    plt.title("Hybrid Retrieval Family Confusion Matrix (DSR + FAISS)")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.show()