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


def _plot_panel(ax, X3, families, query_idx, neighbor_indices, title):
    query_family = families[query_idx]

    # background
    ax.scatter(
        X3[:, 0], X3[:, 1], X3[:, 2],
        c="lightgray", s=8, alpha=0.14
    )

    # query
    qx, qy, qz = X3[query_idx]
    ax.scatter(
        [qx], [qy], [qz],
        c="red", s=130, marker="*"
    )

    # neighbors
    for ni in neighbor_indices:
        nx, ny, nz = X3[ni]
        fam = families[ni]
        color = "green" if fam == query_family else "orange"

        ax.scatter([nx], [ny], [nz], c=color, s=70, alpha=0.95)
        ax.plot(
            [qx, nx], [qy, ny], [qz, nz],
            color=color, alpha=0.45, linewidth=1.2
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")


def plot_3panel_retrieval_flow(
    query_idx=0,
    k=5,
    emb_path="results/kb_embeddings_rnafm_len256_full.pt",
    z_path="results/dsr_global_Z.pt",
    latent_path="results/dsr_latent_embeddings.pt",
    save_path="results/plots/rna_3panel_retrieval_flow.png",
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
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X3 = reducer.fit_transform(X)

    faiss_neighbors = _get_faiss_neighbors(X, query_idx, k=k)
    dsr_neighbors = _get_dsr_neighbors(Z, query_idx, k=k)
    hybrid_neighbors = _get_dsr_faiss_neighbors(latent, query_idx, k=k)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    _plot_panel(ax1, X3, families, query_idx, faiss_neighbors, "RNA-FM + FAISS")

    ax2 = fig.add_subplot(132, projection="3d")
    _plot_panel(ax2, X3, families, query_idx, dsr_neighbors, "RNA-FM + DSR")

    ax3 = fig.add_subplot(133, projection="3d")
    _plot_panel(ax3, X3, families, query_idx, hybrid_neighbors, "RNA-FM + DSR + FAISS")

    query_family = families[query_idx]
    fig.suptitle(f"3D Retrieval Flow Comparison | Query idx={query_idx} | Family={query_family}", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Query family:", query_family)
    print("FAISS neighbors:", [families[i] for i in faiss_neighbors])
    print("DSR neighbors:", [families[i] for i in dsr_neighbors])
    print("Hybrid neighbors:", [families[i] for i in hybrid_neighbors])
    print("Saved:", save_path)