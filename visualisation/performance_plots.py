import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_f1_by_method(csv_path, save_path="results/plots/f1_by_method.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    summary = df.groupby("method")["f1"].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    summary.plot(kind="bar")
    plt.ylabel("Mean F1")
    plt.title("Mean F1 by Method")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_k_sweep(csv_path, save_path="results/plots/f1_vs_k.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["method"].str.contains("guided_k", na=False)]

    summary = df.groupby("k").agg(f1_mean=("f1", "mean"), f1_std=("f1", "std")).reset_index()
    summary = summary.sort_values("k")

    plt.figure(figsize=(6, 4))
    plt.errorbar(summary["k"], summary["f1_mean"], yerr=summary["f1_std"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Mean F1")
    plt.title("F1 vs k")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_violin_f1(csv_path, save_path="results/plots/f1_violin.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    methods = list(df["method"].unique())
    data = [df[df["method"] == m]["f1"].values for m in methods]

    plt.figure(figsize=(10, 5))
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks(range(1, len(methods) + 1), methods, rotation=45, ha="right")
    plt.ylabel("F1")
    plt.title("F1 Distribution by Method")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()