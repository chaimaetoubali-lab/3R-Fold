import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def plot_dsr_global_comparison(
    benchmark_csv=None,
    dsr_csv=None,
    save_path=None,
):
    if benchmark_csv is None:
        benchmark_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_ablation.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_global_comparison.png")

    df1 = pd.read_csv(benchmark_csv)
    df2 = pd.read_csv(dsr_csv)

    keep_methods = [
        "vienna_only",
        "rnafm_faiss_guided_k5",
        "rnafm_faiss_guided_k7",
        "rnafm_faiss_guided_k14",
    ]

    df1 = df1[df1["method"].isin(keep_methods)]

    df = pd.concat([df1, df2], ignore_index=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9,5))

    sns.barplot(
        data=df,
        x="method",
        y="f1"
    )

    plt.xticks(rotation=45)
    plt.ylabel("Mean F1")
    plt.title("FAISS vs Global DSR")

    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()


def plot_dsr_gain_distribution(
    faiss_csv=None,
    dsr_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_ablation.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_gain_distribution.png")

    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)

    faiss = faiss[
        faiss["method"] == "rnafm_faiss_guided_k5"
    ][["query_id","f1"]].rename(
        columns={"f1":"f1_faiss"}
    )

    dsr = dsr[
        ["query_id","f1"]
    ].rename(
        columns={"f1":"f1_dsr"}
    )

    merged = faiss.merge(
        dsr,
        on="query_id"
    )

    merged["gain"] = merged["f1_dsr"] - merged["f1_faiss"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7,4))

    plt.hist(
        merged["gain"],
        bins=20
    )

    plt.axvline(0, linestyle="--")

    plt.xlabel("F1 gain (DSR - FAISS)")
    plt.ylabel("Count")

    plt.title("DSR Gain Distribution")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


def plot_dsr_gain_vs_length(
    faiss_csv=None,
    dsr_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_ablation.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_gain_vs_length.png")

    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)

    faiss = faiss[
        faiss["method"] == "rnafm_faiss_guided_k5"
    ][["query_id","f1","length"]].rename(
        columns={"f1":"f1_faiss"}
    )

    dsr = dsr[
        ["query_id","f1"]
    ].rename(
        columns={"f1":"f1_dsr"}
    )

    merged = faiss.merge(
        dsr,
        on="query_id"
    )

    merged["gain"] = merged["f1_dsr"] - merged["f1_faiss"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6,5))

    plt.scatter(
        merged["length"],
        merged["gain"],
        alpha=0.7
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("RNA length")
    plt.ylabel("F1 gain (DSR - FAISS)")

    plt.title("DSR Gain vs RNA Length")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()