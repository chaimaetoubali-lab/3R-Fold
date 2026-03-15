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
from retrieval.dsr_retrieval import retrieve_neighbors_dsr
from evaluation.metrics import evaluate_structure
from retrieval.retrieval_guided_fold import fold_with_neighbor_guidance


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_csv = os.path.join(results_dir, "dsr_ablation.csv")
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

    rows = []

    n_queries = min(50, len(dataset))

    for i in range(n_queries):
        print(f"Running DSR query {i+1}/{n_queries}")

        query = dataset[i]
        query_emb = embeddings[i].unsqueeze(0)

        # use all embeddings as KB for now
        kb_embeddings = embeddings

        H_star = torch.cat([query_emb, kb_embeddings], dim=0)

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

        neighbors, fams, vals = retrieve_neighbors_dsr(
            dsr.Z,
            dataset,
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
            "method": "rnafm_dsr_guided_k5",
            "f1": f1,
            "sensitivity": sens,
            "ppv": ppv
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()