import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch

from data.build_dataset import build_dataset
from models.pretrained_embedder import PretrainedRNAEmbedder
from retrieval.build_kb import build_kb_embeddings
from retrieval.faiss_index import FaissIndex
from evaluation.benchmark import run_single_query


def run_lofo():
    os.makedirs("results", exist_ok=True)

    print("Building dataset...")
    dataset = build_dataset()
    dataset = list(dataset)
    print("Dataset built:", len(dataset))

    families = sorted(set(d["family"] for d in dataset))
    print("Families:", families)

    embedder = PretrainedRNAEmbedder(device="cpu", max_length=256)

    full_cache_path = "results/kb_embeddings_rnafm_len256_full.pt"

    if os.path.exists(full_cache_path):
        print(f"Loading full cached embeddings from {full_cache_path}...")
        all_embeddings = torch.load(full_cache_path, map_location="cpu")
    else:
        print("Building full embeddings once for all data...")
        all_embeddings = build_kb_embeddings(
            dataset,
            embedder,
            batch_size=4,
            max_length=256,
            cache_path=full_cache_path,
        )

    print("All embeddings shape:", all_embeddings.shape)

    all_rows = []

    for fam in families:
        print(f"\nLOFO: {fam}")

        kb_indices = [i for i, d in enumerate(dataset) if d["family"] != fam]
        test_indices = [i for i, d in enumerate(dataset) if d["family"] == fam]

        kb_data = [dataset[i] for i in kb_indices]
        test_data = [dataset[i] for i in test_indices]
        kb_embeddings = all_embeddings[kb_indices]

        print("KB size:", len(kb_data), "Test size:", len(test_data))
        print("KB embedding shape:", kb_embeddings.shape)

        if len(test_data) < 5:
            print("Skipping, too few test examples")
            continue

        index = FaissIndex(kb_embeddings.shape[1])
        index.build(kb_embeddings)

        for query_item in test_data:
            row = run_single_query(
                query_item=query_item,
                kb_data=kb_data,
                embedder=embedder,
                method_name="rnafm_faiss_guided_k14_lofo",
                retrieval_type="faiss",
                index=index,
                k=14,
                use_guidance=True,
                guidance_mode="consensus",
            )
            row["heldout_family"] = fam
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv("results/lofo_results.csv", index=False)

    summary = (
        df.groupby("heldout_family")
        .agg(
            n=("query_id", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            ppv_mean=("ppv", "mean"),
            purity_mean=("family_purity", "mean"),
            avg_similarity_mean=("avg_similarity", "mean"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )

    summary.to_csv("results/lofo_summary.csv", index=False)

    print("\nSaved:")
    print("results/lofo_results.csv")
    print("results/lofo_summary.csv")

    print("\nLOFO summary:")
    print(summary)


if __name__ == "__main__":
    try:
        run_lofo()
    except Exception:
        import traceback
        traceback.print_exc()
        raise