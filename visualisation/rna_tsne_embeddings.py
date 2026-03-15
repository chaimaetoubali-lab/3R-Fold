import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_rna_tsne(
    emb_path="results/kb_embeddings_rnafm_len256_full.pt",
    dataset=None,
):

    X = torch.load(emb_path).cpu().numpy()
    families = [d["family"] for d in dataset]

    coords = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    ).fit_transform(X)

    plt.figure(figsize=(7,6))

    for fam in sorted(set(families)):

        idx = [i for i,f in enumerate(families) if f==fam]

        plt.scatter(
            coords[idx,0],
            coords[idx,1],
            label=fam,
            s=15
        )

    plt.legend()
    plt.title("t-SNE of RNA-FM Embedding Space")

    plt.show()