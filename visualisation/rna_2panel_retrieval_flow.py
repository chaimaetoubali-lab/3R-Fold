import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

from data.build_dataset import build_dataset
from retrieval.faiss_index import FaissIndex


def _get_faiss_neighbors(X, query_idx, k=5):
    index = FaissIndex(X.shape[1])
    index.build(X.astype("float32"))

    query_embedding = X[query_idx:query_idx + 1]
    distances, indices = index.search(query_embedding, k=k + 1)

    neighbor_indices = [int(i) for i in indices[0] if int(i) != query_idx][:k]
    return neighbor_indices


def _get_dsr_neighbors(Z, query_idx, k=5):
    Z = np.asarray(Z)
    row = Z[query_idx].copy()
    row[query_idx] = -1e9
    idx = np.argsort(row)[-k:][::-1]
    return [int(i) for i in idx]


def _get_dsr_faiss_neighbors(latent, query_idx, k=5):
    index = FaissIndex(latent.shape[1])
    index.build(latent.astype("float32"))

    query_embedding = latent[query_idx:query_idx + 1]
    distances, indices = index.search(query_embedding, k=k + 1)

    neighbor_indices = [int(i) for i in indices[0] if int(i) != query_idx][:k]
    return neighbor_indices


def _plot_panel(ax, X2, families, query_idx, neighbor_indices, title):
    query_family = families[query_idx]

    # background
    ax.scatter(
        X2[:, 0], X2[:, 1],
        c="lightgray", s=10, alpha=0.15
    )

    # query
    qx, qy = X2[query_idx]
    ax.scatter(
        [qx], [qy],
        c="red", s=180, marker="*",
        label="Query"
    )

    # neighbors
    for ni in neighbor_indices:
        nx, ny = X2[ni]
        fam = families[ni]
        color = "green" if fam == query_family else "orange"

        ax.scatter([nx], [ny], c=color, s=80, alpha=0.95)
        ax.plot(
            [qx, nx], [qy, ny],
            color=color, alpha=0.45, linewidth=1.4
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


def plot_2panel_retrieval_flow(
    query_idx=0,
    k=5,
    emb_path="results/kb_embeddings_rnafm_len256_full.pt",
    z_path="results/dsr_global_Z.pt",
    latent_path="results/dsr_latent_embeddings.pt",
    save_path="results/plots/rna_2panel_retrieval_flow.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = list(build_dataset())
    families = [d["family"] for d in dataset]

    X = torch.load(emb_path, map_location="cpu")
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X)

    Z = torch.load(z_path, map_location="cpu")
    if hasattr(Z, "detach"):
        Z = Z.detach().cpu().numpy()
    else:
        Z = np.asarray(Z)

    latent = torch.load(latent_path, map_location="cpu")
    if hasattr(latent, "detach"):
        latent = latent.detach().cpu().numpy()
    else:
        latent = np.asarray(latent)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X2 = reducer.fit_transform(X)

    faiss_neighbors = _get_faiss_neighbors(X, query_idx, k=k)
    dsr_neighbors = _get_dsr_neighbors(Z, query_idx, k=k)
    hybrid_neighbors = _get_dsr_faiss_neighbors(latent, query_idx, k=k)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _plot_panel(axes[0], X2, families, query_idx, faiss_neighbors, "RNA-FM + FAISS")
    _plot_panel(axes[1], X2, families, query_idx, dsr_neighbors, "RNA-FM + DSR")
    _plot_panel(axes[2], X2, families, query_idx, hybrid_neighbors, "RNA-FM + DSR + FAISS")

    query_family = families[query_idx]
    fig.suptitle(
        f"2D Retrieval Flow Comparison | Query idx={query_idx} | Family={query_family}",
        fontsize=14
    )

    # compact legend
    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=12, label="Query"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Same-family neighbor"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Cross-family neighbor"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Query family:", query_family)
    print("FAISS neighbors:", [families[i] for i in faiss_neighbors])
    print("DSR neighbors:", [families[i] for i in dsr_neighbors])
    print("Hybrid neighbors:", [families[i] for i in hybrid_neighbors])
    print("Saved:", save_path)