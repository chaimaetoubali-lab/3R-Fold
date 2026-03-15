import sys
import os

# add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from data.build_dataset import build_dataset
from evaluation.benchmark import run_benchmark
from evaluation.family_analysis import summarize_by_family
from models.pretrained_embedder import PretrainedRNAEmbedder


def main():
    os.makedirs("results", exist_ok=True)

    dataset = build_dataset()

    embedder = PretrainedRNAEmbedder(device="cpu", max_length=256)

    df_baseline = run_benchmark(
        dataset=dataset,
        embedder=embedder,
        n_queries=50,
        test_fraction=0.2,
        seed=42,
        method_name="vienna_only",
        retrieval_type=None,
        use_guidance=False,
        k=0,
        batch_size=4,
        max_length=256,
        cache_path="results/kb_embeddings_rnafm_len256.pt",
    )

    df_guided = run_benchmark(
        dataset=dataset,
        embedder=embedder,
        n_queries=50,
        test_fraction=0.2,
        seed=42,
        method_name="rnafm_faiss_guided_k5",
        retrieval_type="faiss",
        use_guidance=True,
        guidance_mode="template_transfer",
        k=5,
        batch_size=4,
        max_length=256,
        cache_path="results/kb_embeddings_rnafm_len256.pt",
    )

    df = pd.concat([df_baseline, df_guided], ignore_index=True)
    df.to_csv("results/benchmark_main.csv", index=False)

    summary = (
        df.groupby("method")
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
    )
    summary.to_csv("results/benchmark_summary.csv", index=False)

    fam = summarize_by_family(df_guided)
    fam.to_csv("results/family_summary_guided.csv", index=False)

    print("\n=== Benchmark summary ===")
    print(summary)

    print("\nSaved:")
    print("results/benchmark_main.csv")
    print("results/benchmark_summary.csv")
    print("results/family_summary_guided.csv")


if __name__ == "__main__":
    main()