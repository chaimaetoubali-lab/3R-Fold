import faiss
import torch
import numpy as np


class FaissIndex:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

    def build(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy().astype("float32")
        else:
            emb = np.asarray(embeddings, dtype="float32")

        self.index.add(emb)

    def search(self, query_embedding, k=5):
        if isinstance(query_embedding, torch.Tensor):
            q = query_embedding.detach().cpu().numpy().astype("float32")
        else:
            q = np.asarray(query_embedding, dtype="float32")

        D, I = self.index.search(q, k)
        return D, I