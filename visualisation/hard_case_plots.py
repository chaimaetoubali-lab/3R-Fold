import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_hard_cases_by_family(
    csv_path="results/failure_cases.csv",
    save_path="results/plots/hard_cases_by_family.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    counts = df["family"].value_counts()

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.ylabel("Count")
    plt.title("Hard Cases by RNA Family")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_hard_cases_by_length(
    csv_path="results/failure_cases.csv",
    save_path="results/plots/hard_cases_by_length.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(6, 4))
    plt.hist(df["length"], bins=20)
    plt.xlabel("RNA length")
    plt.ylabel("Count")
    plt.title("Hard Cases by RNA Length")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()