import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_lofo_f1(csv_path="results/lofo_summary.csv",
                 save_path="results/plots/lofo_f1_by_family.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("f1_mean", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df["heldout_family"], df["f1_mean"], yerr=df["f1_std"], capsize=4)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean F1")
    plt.title("LOFO generalization by held-out family")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()