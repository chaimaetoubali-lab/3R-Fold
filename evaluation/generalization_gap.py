import pandas as pd


def compute_generalization_gap(
    family_summary_csv="results/family_summary_guided.csv",
    lofo_summary_csv="results/lofo_summary.csv",
    out_csv="results/generalization_gap.csv",
):
    fam = pd.read_csv(family_summary_csv)
    lofo = pd.read_csv(lofo_summary_csv)

    fam = fam.rename(columns={
        "family": "rna_family",
        "f1_mean": "same_family_f1",
        "f1_std": "same_family_f1_std",
    })

    lofo = lofo.rename(columns={
        "heldout_family": "rna_family",
        "f1_mean": "lofo_f1",
        "f1_std": "lofo_f1_std",
    })

    merged = fam.merge(lofo, on="rna_family", how="inner")
    merged["generalization_gap"] = merged["same_family_f1"] - merged["lofo_f1"]

    merged.to_csv(out_csv, index=False)
    return merged