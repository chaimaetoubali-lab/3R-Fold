import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def plot_final_model_comparison():

    base = get_results_path()

    vienna = pd.read_csv(f"{base}/benchmark_main.csv")
    faiss = pd.read_csv(f"{base}/ablation_results.csv")
    dsr = pd.read_csv(f"{base}/dsr_global_all_results.csv")
    hybrid = pd.read_csv(f"{base}/dsr_faiss_hybrid_results.csv")

    vienna = vienna[vienna.method=="vienna_only"]
    faiss = faiss[faiss.method=="rnafm_faiss_guided_k14"]
    dsr = dsr[dsr.method=="rnafm_dsr_global_guided_k14"]
    hybrid = hybrid[hybrid.method=="rnafm_dsr_faiss_guided_k14"]

    rows = []

    rows.append({
        "method":"ViennaRNA",
        "f1":vienna.f1.mean()
    })

    rows.append({
        "method":"RNA-FM + FAISS",
        "f1":faiss.f1.mean()
    })

    rows.append({
        "method":"RNA-FM + DSR",
        "f1":dsr.f1.mean()
    })

    rows.append({
        "method":"RNA-FM + DSR + FAISS",
        "f1":hybrid.f1.mean()
    })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8,5))

    sns.barplot(
        data=df,
        x="method",
        y="f1"
    )

    plt.ylabel("Mean F1")
    plt.xlabel("")
    plt.title("RNA Secondary Structure Prediction Performance")

    plt.xticks(rotation=20)

    plt.tight_layout()

    os.makedirs(get_plots_path(), exist_ok=True)

    plt.savefig(
        os.path.join(get_plots_path(), "final_model_comparison.png"),
        dpi=300
    )

    plt.show()

    print(df)