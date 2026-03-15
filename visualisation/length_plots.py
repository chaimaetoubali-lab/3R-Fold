import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def length_bin(length):
    if length < 100:
        return "<100"
    elif length < 300:
        return "100-299"
    elif length < 700:
        return "300-699"
    else:
        return "700+"


def plot_improvement_vs_length(csv_path, method="rnafm_faiss_guided_k14"):
    df = pd.read_csv(csv_path)

    baseline = df[df["method"] == "vienna_only"][["query_id", "f1"]].rename(
        columns={"f1": "baseline_f1"}
    )

    df = df.merge(baseline, on="query_id", how="left")
    df["improvement"] = df["f1"] - df["baseline_f1"]
    df["length_bin"] = df["length"].apply(length_bin)

    plot_df = df[df["method"] == method].copy()

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="length_bin",
        y="improvement",
        order=["<100", "100-299", "300-699", "700+"]
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"F1 Improvement vs RNA Length ({method})")
    plt.ylabel("F1 improvement over ViennaRNA")
    plt.xlabel("RNA length bin")
    plt.tight_layout()
    plt.savefig("results/plots/improvement_vs_length.png", dpi=300)
    plt.show()


def plot_f1_by_length_summary(csv_path):
    df = pd.read_csv(csv_path)

    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="length_bin",
        y="f1_mean",
        hue="method",
        order=["<100", "100-299", "300-699", "700+"]
    )
    plt.title("Mean F1 by RNA Length")
    plt.ylabel("Mean F1")
    plt.xlabel("RNA length bin")
    plt.tight_layout()
    plt.savefig("results/plots/f1_by_length.png", dpi=300)
    plt.show()