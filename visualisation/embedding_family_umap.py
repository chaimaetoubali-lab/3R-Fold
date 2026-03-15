import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from config import get_plots_path, get_results_path


def plot_embedding_family_umap(
    embeddings_path=None,
    title="RNA-FM Embedding UMAP by Family",
    save_path=None,
):
    if embeddings_path is None:
        embeddings_path = os.path.join(get_results_path(), "kb_embeddings_rnafm_len256_full.pt")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "embedding_family_umap.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    emb = torch.load(embeddings_path, map_location="cpu")
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    else:
        emb = np.asarray(emb)

    from data.build_dataset import build_dataset
    dataset = list(build_dataset())
    families = [d["family"] for d in dataset]

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(emb)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "family": families,
    })

    plt.figure(figsize=(10, 8))
    unique_families = sorted(df["family"].unique())

    for fam in unique_families:
        sub = df[df["family"] == fam]
        plt.scatter(sub["x"], sub["y"], s=18, alpha=0.75, label=fam)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()