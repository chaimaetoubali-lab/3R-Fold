import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def plot_dsr_all_vs_faiss_k14(
    faiss_csv=None,
    dsr_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_all_results.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_all_vs_faiss_k14.png")
    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)

    faiss = faiss[faiss["method"] == "rnafm_faiss_guided_k14"].copy()

    df = pd.concat([faiss, dsr], ignore_index=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="method", y="f1")
    plt.xticks(rotation=20)
    plt.ylabel("Mean F1")
    plt.title("Global DSR vs FAISS (k=14)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_dsr_all_gain_distribution(
    faiss_csv=None,
    dsr_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_all_results.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_all_gain_distribution.png")
    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)

    faiss = faiss[faiss["method"] == "rnafm_faiss_guided_k14"][["query_id", "f1"]].rename(columns={"f1": "f1_faiss"})
    dsr = dsr[["query_id", "f1"]].rename(columns={"f1": "f1_dsr"})

    merged = faiss.merge(dsr, on="query_id", how="inner")
    merged["gain"] = merged["f1_dsr"] - merged["f1_faiss"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(merged["gain"], bins=30)
    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("F1 gain (DSR - FAISS)")
    plt.ylabel("Count")
    plt.title("Global DSR Gain Distribution (k=14)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_dsr_all_gain_vs_length(
    faiss_csv=None,
    dsr_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_all_results.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_all_gain_vs_length.png")
    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)

    faiss = faiss[faiss["method"] == "rnafm_faiss_guided_k14"][["query_id", "f1", "length"]].rename(columns={"f1": "f1_faiss"})
    dsr = dsr[["query_id", "f1"]].rename(columns={"f1": "f1_dsr"})

    merged = faiss.merge(dsr, on="query_id", how="inner")
    merged["gain"] = merged["f1_dsr"] - merged["f1_faiss"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(merged["length"], merged["gain"], alpha=0.6)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("RNA length")
    plt.ylabel("F1 gain (DSR - FAISS)")
    plt.title("Global DSR Gain vs RNA Length")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()