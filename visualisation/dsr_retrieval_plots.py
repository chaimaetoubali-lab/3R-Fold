import pandas as pd
import matplotlib.pyplot as plt
import os

from config import get_results_path, get_plots_path


def plot_neighbor_quality():

    df = pd.read_csv(os.path.join(get_results_path(), "ablation_results.csv"))

    methods = [
        "rnafm_faiss_guided_k5",
        "rnafm_dsr_guided_k5"
    ]

    sub = df[df.method.isin(methods)]

    plt.figure(figsize=(6,5))

    for m in methods:

        plt.hist(
            sub[sub.method == m]["avg_similarity"],
            alpha=0.5,
            label=m,
            bins=20
        )

    plt.legend()

    plt.xlabel("Average neighbor similarity")
    plt.ylabel("Count")

    plt.title("Neighbor quality: FAISS vs DSR")

    plt.tight_layout()

    plt.savefig(os.path.join(get_plots_path(), "dsr_neighbor_quality.png"))

    plt.show()