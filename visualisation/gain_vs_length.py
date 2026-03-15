import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gain_vs_length(csv_path="results/all_case_gains.csv"):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(7,5))

    bins = [0,100,300,700,10000]

    labels = ["<100","100-299","300-699","700+"]

    df["length_bin"] = pd.cut(df["length"], bins=bins, labels=labels)

    sns.boxplot(
    data=df,
    x="length_bin",
    y="gain")

    plt.axhline(0, linestyle="--")
    plt.title("Retrieval Gain vs RNA Length")

    plt.tight_layout()

    plt.show()