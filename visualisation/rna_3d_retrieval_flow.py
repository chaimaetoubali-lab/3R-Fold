import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

from data.build_dataset import build_dataset
from retrieval.faiss_index import FaissIndex
from config import get_plots_path, get_results_path


def plot_3d_retrieval_flow(
    query_idx=0,
    k=5,
    embeddings_path=None,
    save_path=None,
):
    if embeddings_path is None:
        embeddings_path = os.path.join(get_results_path(), "kb_embeddings_rnafm_len256_full.pt")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "rna_3d_retrieval_flow.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = list(build_dataset())
    families = [d["family"] for d in dataset]

    X = torch.load(embeddings_path, map_location="cpu")
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X)

    # 3D manifold
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X3 = reducer.fit_transform(X)

    # retrieval in original embedding space
    index = FaissIndex(X.shape[1])
    index.build(X.astype("float32"))

    query_embedding = X[query_idx:query_idx + 1]
    distances, indices = index.search(query_embedding, k=k + 1)

    neighbor_indices = [int(i) for i in indices[0] if int(i) != query_idx][:k]
    neighbor_distances = [float(d) for i, d in zip(indices[0], distances[0]) if int(i) != query_idx][:k]

    query_family = families[query_idx]
    neighbor_families = [families[i] for i in neighbor_indices]

    # plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    for ni, dist in zip(neighbor_indices, neighbor_distances):
        nx, ny, nz = X3[ni]
        fam = families[ni]

        color = "green" if fam == query_family else "orange"

        ax.scatter(
            [nx], [ny], [nz],
            c=color,
            s=80,
            alpha=0.95
        )

        ax.plot(
            [qx, nx],
            [qy, ny],
            [qz, nz],
            color=color,
            alpha=0.45,
            linewidth=1.2
        )

    # neighbors
    unique_fams = sorted(set(families))
    fam_to_color = {f: i for i, f in enumerate(unique_fams)}

    for ni, dist in zip(neighbor_indices, neighbor_distances):
        nx, ny, nz = X3[ni]
        fam = families[ni]
        color_val = fam_to_color[fam]

        ax.scatter(
            [nx], [ny], [nz],
            c=[color_val],
            cmap="tab10",
            vmin=0,
            vmax=max(1, len(unique_fams) - 1),
            s=70,
            alpha=0.95
        )

        ax.plot(
            [qx, nx],
            [qy, ny],
            [qz, nz],
            color="black",
            alpha=0.45,
            linewidth=1.2
        )

    # annotate only query + neighbors
    ax.text(qx, qy, qz, f"Q:{query_family}", fontsize=9)
    for ni in neighbor_indices:
        nx, ny, nz = X3[ni]
        ax.text(nx, ny, nz, families[ni], fontsize=7)

    ax.set_title(f"3D Retrieval Flow Plot (query idx={query_idx}, k={k})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Query family:", query_family)
    print("Neighbor families:", neighbor_families)
    print("Saved:", save_path)