import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import torch

from data.build_dataset import build_dataset
from models.autoencoder import RNADeepAE
from models.deep_self_representation import DeepSelfRepresentation
from training.train_dsr import train_dsr
from evaluation.metrics import evaluate_structure
from retrieval.retrieval_guided_fold import fold_with_neighbor_guidance


def retrieve_neighbors_from_global_dsr(Z, dataset, query_idx, k=5):
    n = Z.shape[0]
    Z_final = Z * (1 - torch.eye(n, device=Z.device))

    z_query = Z_final[query_idx].detach().cpu().clone()
    z_query[query_idx] = -1e9

    vals, idx = torch.topk(z_query, k)
    neighbor_indices = idx.tolist()

    neighbors = [dataset[i] for i in neighbor_indices]
    neighbor_families = [n["family"] for n in neighbors]

    return neighbors, neighbor_families, vals.tolist(), neighbor_indices


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_csv = os.path.join(results_dir, "dsr_global_ablation.csv")
    emb_path = os.path.join(results_dir, "kb_embeddings_rnafm_len256_full.pt")

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RESULTS_DIR:", results_dir)
    print("OUTPUT CSV:", out_csv)
    print("EMBEDDINGS PATH:", emb_path)

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")

    dataset = list(build_dataset())
    embeddings = torch.load(emb_path, map_location="cpu")

    if len(dataset) != len(embeddings):
        raise ValueError(
            f"Dataset size ({len(dataset)}) != embeddings size ({len(embeddings)})"
        )

    print("Training global DSR once on all embeddings...")
    H_star = embeddings.float()

    ae = RNADeepAE(
        input_dim=H_star.shape[1],
        latent_dim=128
    )

    dsr = DeepSelfRepresentation(
        n_samples=H_star.shape[0],
        latent_dim=128
    )

    dsr = train_dsr(
        ae,
        dsr,
        H_star,
        epochs=20,
        lr=1e-3
    )

    z_path = os.path.join(results_dir, "dsr_global_Z.pt")
    torch.save(dsr.Z.detach().cpu(), z_path)
    print("Saved DSR matrix to:", z_path)

    rows = []

    n_queries = min(50, len(dataset))

    for i in range(n_queries):
        print(f"Evaluating query {i+1}/{n_queries}")

        query = dataset[i]

        neighbors, fams, vals, neighbor_indices = retrieve_neighbors_from_global_dsr(
            dsr.Z,
            dataset,
            query_idx=i,
            k=5
        )

        neighbor_structures = [n["structure"] for n in neighbors if n.get("structure", "")]

        pred = fold_with_neighbor_guidance(
            query["sequence"],
            neighbor_structures
        )

        sens, ppv, f1 = evaluate_structure(
            pred,
            query["structure"]
        )

        rows.append({
            "query_id": query["id"],
            "family": query["family"],
            "length": len(query["sequence"]),
            "method": "rnafm_dsr_global_guided_k5",
            "f1": f1,
            "sensitivity": sens,
            "ppv": ppv,
            "neighbor_families": "|".join(fams),
            "avg_dsr_score": sum(vals) / len(vals) if len(vals) > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(df.groupby("method")["f1"].mean())
    print(df.head())


if __name__ == "__main__":
    main()