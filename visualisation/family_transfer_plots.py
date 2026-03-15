import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_family_transfer_gain(
    csv_path="results/family_transfer.csv",
    save_path="results/plots/family_transfer_gain.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("transfer_gain", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="rna_family", y="transfer_gain")
    plt.axhline(0, linestyle="--", color="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("LOFO F1 - ViennaRNA F1")
    plt.xlabel("")
    plt.title("Family Transfer Gain over ViennaRNA")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_family_transfer_scatter(
    csv_path="results/family_transfer.csv",
    save_path="results/plots/family_transfer_scatter.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="baseline_f1", y="lofo_f1", s=90)

    for _, row in df.iterrows():
        plt.text(row["baseline_f1"], row["lofo_f1"], row["rna_family"], fontsize=8)

    maxv = max(df["baseline_f1"].max(), df["lofo_f1"].max()) + 0.05
    plt.plot([0, maxv], [0, maxv], linestyle="--", color="black")

    plt.xlabel("ViennaRNA F1")
    plt.ylabel("LOFO F1")
    plt.title("Family Transfer: LOFO vs ViennaRNA")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()