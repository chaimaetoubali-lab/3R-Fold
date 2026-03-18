import sys
import os

from evaluation.benchmark import split_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import torch

from data.build_dataset import build_dataset
from models.autoencoder import RNADeepAE
from models.deep_self_representation import DeepSelfRepresentation
from training.train_dsr import train_dsr
from retrieval.faiss_index import FaissIndex
from evaluation.metrics import evaluate_structure
from retrieval.retrieval_guided_fold import fold_with_neighbor_guidance


def main(seed=42, n_queries=50,test_fraction=0.2):
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    emb_path = os.path.join(results_dir, "kb_embeddings_rnafm_len256_full.pt")
    out_csv = os.path.join(results_dir, "dsr_faiss_hybrid_results.csv")
    latent_path = os.path.join(results_dir, "dsr_latent_embeddings.pt")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")

    dataset = list(build_dataset())
    embeddings = torch.load(emb_path, map_location="cpu").float()

    if len(dataset) != len(embeddings):
        raise ValueError(
            f"Dataset size ({len(dataset)}) != embeddings size ({len(embeddings)})"
        )

    print("Dataset size:", len(dataset))
    print("Embeddings shape:", embeddings.shape)

    print("Training DSR once on RNA-FM embeddings...")
    ae = RNADeepAE(
        input_dim=embeddings.shape[1],
        latent_dim=256
    )

    dsr = DeepSelfRepresentation(
        n_samples=embeddings.shape[0],
        latent_dim=256
    )

    dsr = train_dsr(
        ae,
        dsr,
        embeddings,
        epochs=80,
        lr=1e-3
    )

    with torch.no_grad():
        latent = ae.encoder(embeddings).detach().cpu()

    torch.save(latent, latent_path)
    print("Saved latent embeddings to:", latent_path)

    print("Building FAISS on DSR latent embeddings...")
    index = FaissIndex(latent.shape[1])
    index.build(latent)

    rows = []
    kb_data, test_data = split_dataset(dataset, test_fraction=test_fraction, seed=seed)

    if n_queries is not None:
        test_data = test_data[:n_queries]

    for i, query in enumerate(test_data):
        if i == 0 or (i + 1) % 100 == 0:
            print(f"Evaluating query {i+1}/{len(test_data)}")

        query_embedding = latent[i].unsqueeze(0)

        distances, indices = index.search(query_embedding, k=15)

        # remove self if retrieved
        neighbor_indices = [int(j) for j in indices[0] if int(j) != i][:14]
        neighbor_distances = [float(d) for j, d in zip(indices[0], distances[0]) if int(j) != i][:14]

        neighbors = [dataset[j] for j in neighbor_indices]
        neighbor_families = [n["family"] for n in neighbors]
        neighbor_structures = [n["structure"] for n in neighbors if n.get("structure", "")]

        pred = fold_with_neighbor_guidance(
            query["sequence"],
            neighbor_structures
        )

        sens, ppv, f1 = evaluate_structure(pred, query["structure"])

        sim_weights = [1.0 / (d + 1e-8) for d in neighbor_distances]
        avg_similarity = sum(sim_weights) / len(sim_weights) if sim_weights else 0.0

        rows.append({
            "query_id": query["id"],
            "family": query["family"],
            "length": len(query["sequence"]),
            "sequence": query["sequence"],
            "ref_structure": query["structure"],
            "pred_structure": pred,
            "method": "rnafm_dsr_faiss_guided_k14",
            "k": 14,
            "f1": f1,
            "sensitivity": sens,
            "ppv": ppv,
            "neighbor_families": "|".join(neighbor_families),
            "neighbor_structures": "|||".join(neighbor_structures),
            "avg_similarity": avg_similarity,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("method")
        .agg(
            n=("query_id", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            ppv_mean=("ppv", "mean"),
            avg_similarity_mean=("avg_similarity", "mean"),
        )
        .reset_index()
    )

    summary_path = os.path.join(results_dir, "dsr_faiss_hybrid_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(out_csv)
    print(summary_path)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()