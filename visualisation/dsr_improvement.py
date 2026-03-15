import pandas as pd
import matplotlib.pyplot as plt
import os

from config import get_results_path, get_plots_path


def plot_dsr_f1_gain():

    df = pd.read_csv(os.path.join(get_results_path(), "ablation_results.csv"))

    faiss = df[df.method=="rnafm_faiss_guided_k5"]["f1"].values
    dsr = df[df.method=="rnafm_dsr_guided_k5"]["f1"].values

    gain = dsr - faiss

    plt.figure(figsize=(6,4))

    plt.hist(gain, bins=20)

    plt.axvline(0, linestyle="--")

    plt.xlabel("F1 gain from DSR")
    plt.ylabel("Count")

    plt.title("DSR improvement distribution")

    plt.tight_layout()

    plt.savefig(os.path.join(get_plots_path(), "dsr_gain.png"))

    plt.show()