import os
import torch
from tqdm import tqdm


def build_kb_embeddings(
    dataset,
    embedder,
    batch_size=4,
    max_length=256,
    cache_path="kb_embeddings_rnafm_len256.pt",
):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        return torch.load(cache_path, map_location="cpu")

    sequences = [item["sequence"][:max_length] for item in dataset]

    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    sorted_sequences = [sequences[i] for i in order]

    all_embeddings = []

    for i in tqdm(range(0, len(sorted_sequences), batch_size), desc="Embedding KB"):
        batch = sorted_sequences[i:i + batch_size]
        emb = embedder.embed(batch)
        all_embeddings.append(emb.detach().cpu())

    sorted_embeddings = torch.cat(all_embeddings, dim=0)

    restored = torch.empty_like(sorted_embeddings)
    for sorted_idx, original_idx in enumerate(order):
        restored[original_idx] = sorted_embeddings[sorted_idx]

    torch.save(restored, cache_path)
    print(f"Saved embeddings to {cache_path}")

    return restored