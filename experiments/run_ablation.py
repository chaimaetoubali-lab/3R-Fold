import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from data.build_dataset import build_dataset
from evaluation.benchmark import run_benchmark
from models.pretrained_embedder import PretrainedRNAEmbedder
from evaluation.length_analysis import summarize_by_length

def main():
    os.makedirs("results", exist_ok=True)

    dataset = build_dataset()
    embedder = PretrainedRNAEmbedder(device="cpu", max_length=256)

    experiments = [
        {
            "method_name": "vienna_only",
            "retrieval_type": None,
            "use_guidance": False,
            "guidance_mode": "consensus",
            "k": 0,
        },
        {
            "method_name": "rnafm_faiss_no_guidance_k5",
            "retrieval_type": "faiss",
            "use_guidance": False,
            "guidance_mode": "consensus",
            "k": 5,
        },
        {
            "method_name": "rnafm_faiss_guided_k1",
            "retrieval_type": "faiss",
            "use_guidance": True,
            "guidance_mode": "consensus",
            "k": 1,
        },
        {
            "method_name": "rnafm_faiss_guided_k3",
            "retrieval_type": "faiss",
            "use_guidance": True,
            "guidance_mode": "consensus",
            "k": 3,
        },
        {
            "method_name": "rnafm_faiss_guided_k5",
            "retrieval_type": "faiss",
            "use_guidance": True,
            "guidance_mode": "consensus",
            "k": 5,
        },
        {
            "method_name": "rnafm_faiss_guided_k7",
            "retrieval_type": "faiss",
            "use_guidance": True,
            "guidance_mode": "consensus",
            "k": 7,
        },
        {
            "method_name": "rnafm_faiss_guided_k14",
            "retrieval_type": "faiss",
            "use_guidance": True,
            "guidance_mode": "consensus",
            "k": 14,
        },
        {
            "method_name": "rnafm_faiss_template_guided_k5",
            "retrieval_type": "faiss",
            "use_guidance": True,
            "guidance_mode": "template_transfer",
            "k": 5,
        },
        {
            "method_name": "oracle_guided_k5",
            "retrieval_type": "oracle",
            "use_guidance": True,
            "guidance_mode": "consensus",
            "k": 5,
        },
    ]

    all_rows = []

    for exp in experiments:
        print(f"\nRunning: {exp['method_name']}")
        df = run_benchmark(
            dataset=dataset,
            embedder=embedder,
            n_queries=50,
            test_fraction=0.2,
            seed=42,
            method_name=exp["method_name"],
            retrieval_type=exp["retrieval_type"],
            use_guidance=exp["use_guidance"],
            k=exp["k"],
            batch_size=4,
            max_length=256,
            cache_path="results/kb_embeddings_rnafm_len256.pt",
        )
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv("results/ablation_results.csv", index=False)

    summary = (
        out.groupby("method")
        .agg(
            n=("query_id", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            ppv_mean=("ppv", "mean"),
            purity_mean=("family_purity", "mean"),
            runtime_mean=("total_time", "mean"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )

    summary.to_csv("results/ablation_summary.csv", index=False)
    length_summary = summarize_by_length(out)
    length_summary.to_csv("results/length_summary.csv", index=False)

    print("\n=== Ablation summary ===")
    print(summary)


if __name__ == "__main__":
    main()