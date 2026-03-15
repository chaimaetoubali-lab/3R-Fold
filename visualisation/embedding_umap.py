import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap

from config import get_plots_path


def plot_embedding_umap(
    embeddings,
    families,
    save_path=None,
    title="RNA-FM Embedding UMAP"
):
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "embedding_umap.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if hasattr(embeddings, "detach"):
        emb = embeddings.detach().cpu().numpy()
    else:
        emb = np.asarray(embeddings)

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

    plt.figure(figsize=(9, 7))
    unique_families = sorted(df["family"].unique())

    for fam in unique_families:
        sub = df[df["family"] == fam]
        plt.scatter(sub["x"], sub["y"], s=18, alpha=0.7, label=fam)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()