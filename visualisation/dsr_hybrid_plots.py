import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def plot_dsr_hybrid_comparison(
    faiss_csv=None,
    dsr_csv=None,
    hybrid_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if dsr_csv is None:
        dsr_csv = os.path.join(get_results_path(), "dsr_global_all_results.csv")
    if hybrid_csv is None:
        hybrid_csv = os.path.join(get_results_path(), "dsr_faiss_hybrid_results.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_hybrid_comparison.png")
    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)
    hybrid = pd.read_csv(hybrid_csv)

    faiss = faiss[faiss["method"] == "rnafm_faiss_guided_k14"]
    dsr = dsr[dsr["method"] == "rnafm_dsr_global_guided_k14"]
    hybrid = hybrid[hybrid["method"] == "rnafm_dsr_faiss_guided_k14"]

    df = pd.concat([faiss, dsr, hybrid], ignore_index=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="method", y="f1")
    plt.xticks(rotation=20)
    plt.ylabel("Mean F1")
    plt.title("FAISS vs DSR vs DSR+FAISS")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_dsr_hybrid_gain_distribution(
    faiss_csv=None,
    hybrid_csv=None,
    save_path=None,
):
    if faiss_csv is None:
        faiss_csv = os.path.join(get_results_path(), "ablation_results.csv")
    if hybrid_csv is None:
        hybrid_csv = os.path.join(get_results_path(), "dsr_faiss_hybrid_results.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_hybrid_gain_distribution.png")
    faiss = pd.read_csv(faiss_csv)
    hybrid = pd.read_csv(hybrid_csv)

    faiss = faiss[faiss["method"] == "rnafm_faiss_guided_k14"][["query_id", "f1"]].rename(columns={"f1": "f1_faiss"})
    hybrid = hybrid[["query_id", "f1"]].rename(columns={"f1": "f1_hybrid"})

    merged = faiss.merge(hybrid, on="query_id", how="inner")
    merged["gain"] = merged["f1_hybrid"] - merged["f1_faiss"]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(merged["gain"], bins=30)
    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("F1 gain (DSR+FAISS - FAISS)")
    plt.ylabel("Count")
    plt.title("Hybrid Gain Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()