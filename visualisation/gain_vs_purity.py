import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gain_vs_purity(csv_path="results/all_case_gains.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="family_purity", y="gain", alpha=0.6)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Retrieval purity")
    plt.ylabel("F1 gain (method - ViennaRNA)")
    plt.title("F1 Gain vs Retrieval Purity")
    plt.tight_layout()
    plt.show()