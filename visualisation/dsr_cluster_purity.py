import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def compute_dsr_neighbor_family_matrix(
    z_path=None,
    k=14,
):
    if z_path is None:
        z_path = os.path.join(get_results_path(), "dsr_global_Z.pt")
    Z = torch.load(z_path, map_location="cpu")
    if hasattr(Z, "detach"):
        Z = Z.detach().cpu()

    from data.build_dataset import build_dataset
    dataset = list(build_dataset())
    families = [d["family"] for d in dataset]
    unique_families = sorted(set(families))

    counts = pd.DataFrame(
        0,
        index=unique_families,
        columns=unique_families,
        dtype=float
    )

    n = Z.shape[0]
    Z_final = Z * (1 - torch.eye(n))

    for i in range(n):
        query_family = families[i]
        row = Z_final[i].clone()
        row[i] = -1e9

        vals, idx = torch.topk(row, k)
        neighbor_families = [families[j] for j in idx.tolist()]

        for nf in neighbor_families:
            counts.loc[query_family, nf] += 1

    row_sums = counts.sum(axis=1).replace(0, 1)
    norm = counts.div(row_sums, axis=0)

    return counts, norm


def plot_dsr_cluster_purity_heatmap(
    z_path=None,
    k=14,
    save_path=None
):
    if z_path is None:
        z_path = os.path.join(get_results_path(), "dsr_global_Z.pt")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "dsr_cluster_purity_heatmap.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    _, norm = compute_dsr_neighbor_family_matrix(z_path=z_path, k=k)

    plt.figure(figsize=(9, 7))
    sns.heatmap(norm, cmap="viridis", annot=True, fmt=".2f")
    plt.xlabel("Neighbor family")
    plt.ylabel("Query family")
    plt.title(f"DSR Neighbor Family Purity Heatmap (k={k})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()