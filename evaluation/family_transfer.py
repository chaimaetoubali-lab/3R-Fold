import pandas as pd


def compute_family_transfer(
    baseline_csv="results/benchmark_main.csv",
    lofo_csv="results/lofo_results.csv",
    out_csv="results/family_transfer.csv",
):
    baseline = pd.read_csv(baseline_csv)
    lofo = pd.read_csv(lofo_csv)

    baseline = baseline[baseline["method"] == "vienna_only"].copy()

    base_summary = (
        baseline.groupby("family")
        .agg(
            baseline_n=("query_id", "count"),
            baseline_f1=("f1", "mean"),
            baseline_sensitivity=("sensitivity", "mean"),
            baseline_ppv=("ppv", "mean"),
        )
        .reset_index()
        .rename(columns={"family": "rna_family"})
    )

    lofo_summary = (
        lofo.groupby("heldout_family")
        .agg(
            lofo_n=("query_id", "count"),
            lofo_f1=("f1", "mean"),
            lofo_sensitivity=("sensitivity", "mean"),
            lofo_ppv=("ppv", "mean"),
        )
        .reset_index()
        .rename(columns={"heldout_family": "rna_family"})
    )

    merged = base_summary.merge(lofo_summary, on="rna_family", how="inner")
    merged["transfer_gain"] = merged["lofo_f1"] - merged["baseline_f1"]

    merged.to_csv(out_csv, index=False)
    return merged