import pandas as pd


def length_bin(length):
    if length < 100:
        return "<100"
    elif length < 300:
        return "100-299"
    elif length < 700:
        return "300-699"
    else:
        return "700+"


def summarize_by_length(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["length_bin"] = out["length"].apply(length_bin)

    summary = (
        out.groupby(["method", "length_bin"])
        .agg(
            n=("query_id", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            ppv_mean=("ppv", "mean"),
            purity_mean=("family_purity", "mean"),
        )
        .reset_index()
    )
    return summary