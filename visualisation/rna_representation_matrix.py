import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list


def plot_rna_representation_matrix(
    z_path="results/dsr_global_Z.pt",
    save_path="results/plots/rna_representation_matrix.png"
):

    Z = torch.load(z_path).cpu().numpy()

    plt.figure(figsize=(7,6))
    sns.heatmap(Z, cmap="viridis")
    plt.title("RNA Representation Matrix (DSR)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_block_diagonal_rna_matrix(
    z_path="results/dsr_global_Z.pt",
    dataset=None,
    save_path="results/plots/rna_block_diagonal_matrix.png"
):

    Z = torch.load(z_path).cpu().numpy()

    families = [d["family"] for d in dataset]

    order = np.argsort(families)

    Z = Z[order][:,order]

    plt.figure(figsize=(7,6))
    sns.heatmap(Z, cmap="magma")
    plt.title("Block Diagonal RNA Representation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()