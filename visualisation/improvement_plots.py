import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_improvement_over_baseline(
    csv_path,
    baseline_method="vienna_only",
    compare_method="rnafm_faiss_guided_k5",
    save_path="results/plots/improvement_over_baseline.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    base = df[df["method"] == baseline_method][["query_id", "f1"]].rename(columns={"f1": "f1_baseline"})
    comp = df[df["method"] == compare_method][["query_id", "f1"]].rename(columns={"f1": "f1_method"})

    merged = base.merge(comp, on="query_id", how="inner")
    merged["f1_improvement"] = merged["f1_method"] - merged["f1_baseline"]

    plt.figure(figsize=(6, 4))
    plt.hist(merged["f1_improvement"], bins=20)
    plt.xlabel("F1 improvement over baseline")
    plt.ylabel("Count")
    plt.title(f"{compare_method} vs {baseline_method}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_improvement_vs_purity(
    csv_path,
    baseline_method="vienna_only",
    compare_method="rnafm_faiss_guided_k5",
    save_path="results/plots/improvement_vs_purity.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    base = df[df["method"] == baseline_method][["query_id", "f1"]].rename(columns={"f1": "f1_baseline"})
    comp = df[df["method"] == compare_method][["query_id", "f1", "family_purity"]].rename(columns={"f1": "f1_method"})

    merged = base.merge(comp, on="query_id", how="inner")
    merged["f1_improvement"] = merged["f1_method"] - merged["f1_baseline"]

    plt.figure(figsize=(5, 5))
    plt.scatter(merged["family_purity"], merged["f1_improvement"], alpha=0.6)
    plt.xlabel("Family purity")
    plt.ylabel("F1 improvement over baseline")
    plt.title("Retrieval purity vs F1 improvement")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()