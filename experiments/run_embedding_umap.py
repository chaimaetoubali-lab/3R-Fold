import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from data.build_dataset import build_dataset
from visualisation.embedding_umap import plot_embedding_umap
from config import get_results_path, get_plots_path


def main():
    dataset = build_dataset()
    dataset = list(dataset)

    cache_path = os.path.join(get_results_path(), "kb_embeddings_rnafm_len256_full.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found. Run the full embedding build first."
        )

    embeddings = torch.load(cache_path, map_location="cpu")
    families = [d["family"] for d in dataset]

    plot_embedding_umap(
        embeddings,
        families,
        save_path=os.path.join(get_plots_path(), "embedding_umap.png"),
        title="RNA-FM Embedding UMAP by Family",
    )


if __name__ == "__main__":
    main()