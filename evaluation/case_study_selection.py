import pandas as pd


def select_case_studies(
    baseline_csv="results/benchmark_main.csv",
    method_csv="results/dsr_faiss_hybrid_results.csv",
    baseline_method="vienna_only",
    method_name="rnafm_dsr_faiss_guided_k14",
):
    base = pd.read_csv(baseline_csv)
    meth = pd.read_csv(method_csv)

    if "method" in base.columns:
        base = base[base["method"] == baseline_method].copy()

    if "method" in meth.columns:
        meth = meth[meth["method"] == method_name].copy()

    keep_cols = ["query_id", "family", "length", "f1"]
    if "sequence" in meth.columns:
        keep_cols.append("sequence")
    if "ref_structure" in meth.columns:
        keep_cols.append("ref_structure")
    if "pred_structure" in meth.columns:
        keep_cols.append("pred_structure")
    if "neighbor_families" in meth.columns:
        keep_cols.append("neighbor_families")
    if "neighbor_structures" in meth.columns:
        keep_cols.append("neighbor_structures")

    meth = meth[keep_cols].copy()

    base = base[["query_id", "f1"]].rename(columns={"f1": "f1_baseline"})
    meth = meth.rename(columns={"f1": "f1_method"})

    merged = base.merge(meth, on="query_id", how="inner")
    merged["gain"] = merged["f1_method"] - merged["f1_baseline"]

    merged = merged.sort_values("gain", ascending=False).reset_index(drop=True)

    best = merged.iloc[0]
    worst = merged.iloc[-1]
    typical = merged.iloc[len(merged) // 2]

    out = pd.DataFrame([best, typical, worst])
    out["case_type"] = ["best_improvement", "typical_case", "worst_case"]

    return out, merged