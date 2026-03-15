import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import get_results_path, get_plots_path


def plot_dsr_comparison():

    df1 = pd.read_csv(os.path.join(get_results_path(), "benchmark_main.csv"))
    df2 = pd.read_csv(os.path.join(get_results_path(), "dsr_ablation.csv"))

    df = pd.concat([df1, df2])

    plt.figure(figsize=(7,5))

    sns.barplot(
        data=df,
        x="method",
        y="f1"
    )

    plt.xticks(rotation=45)
    plt.ylabel("Mean F1")

    plt.title("Impact of Deep Self Representation")

    plt.tight_layout()

    plt.savefig(os.path.join(get_plots_path(), "dsr_comparison.png"))

    plt.show()