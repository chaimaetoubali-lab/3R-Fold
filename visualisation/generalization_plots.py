import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_generalization_gap(
    csv_path="results/generalization_gap.csv",
    save_path="results/plots/generalization_gap.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("generalization_gap", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df["rna_family"], df["generalization_gap"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Same-family F1 - LOFO F1")
    plt.title("Generalization Gap by RNA Family")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_same_vs_lofo(
    csv_path="results/generalization_gap.csv",
    save_path="results/plots/same_vs_lofo.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["same_family_f1"], df["lofo_f1"], s=80)

    for _, row in df.iterrows():
        plt.text(row["same_family_f1"], row["lofo_f1"], row["rna_family"], fontsize=8)

    maxv = max(df["same_family_f1"].max(), df["lofo_f1"].max()) + 0.05
    plt.plot([0, maxv], [0, maxv], linestyle="--")

    plt.xlabel("Same-family F1")
    plt.ylabel("LOFO F1")
    plt.title("Same-family vs LOFO Performance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()