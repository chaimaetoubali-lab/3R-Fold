import pandas as pd


def summarize_by_family(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("family")
        .agg(
            n=("query_id", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            ppv_mean=("ppv", "mean"),
            purity_mean=("family_purity", "mean"),
            top1_match_mean=("top1_family_match", "mean"),
            avg_similarity_mean=("avg_similarity", "mean"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )
    return out