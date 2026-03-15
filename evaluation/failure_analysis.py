import pandas as pd


def retrieval_failure_analysis(
    csv_path="results/ablation_results.csv",
    method="rnafm_faiss_guided_k14",
    out_csv="results/failure_cases.csv",
    top_n=20,
):
    df = pd.read_csv(csv_path)
    df = df[df["method"] == method].copy()

    # hardest cases = lowest F1
    hard = df.sort_values("f1", ascending=True).head(top_n).copy()

    # cases with high purity but still low F1
    high_purity_fail = df[df["family_purity"] >= 0.8].sort_values("f1", ascending=True).head(top_n).copy()
    high_purity_fail["failure_type"] = "high_purity_low_f1"

    # cases with low purity but high F1
    low_purity_success = df[df["family_purity"] < 0.5].sort_values("f1", ascending=False).head(top_n).copy()
    low_purity_success["failure_type"] = "low_purity_high_f1"

    hard["failure_type"] = "hardest_cases"

    out = pd.concat([hard, high_purity_fail, low_purity_success], ignore_index=True)
    out.to_csv(out_csv, index=False)
    return out