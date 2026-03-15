import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from data.build_dataset import build_dataset


def wrap_text(s, width=120):
    return "\n".join(textwrap.wrap(str(s), width=width))


def _safe_get(row, key, default=""):
    return row[key] if key in row.index and pd.notna(row[key]) else default


def _find_dataset_row(query_id, family=None):
    dataset = list(build_dataset())

    # exact id match
    for row in dataset:
        if str(row["id"]) == str(query_id):
            return row

    # relaxed fallback: try family + same prefix substring
    if family is not None:
        fam_rows = [r for r in dataset if r["family"] == family]
        if len(fam_rows) > 0:
            return fam_rows[0]

    return None


def plot_side_by_side_case(
    query_id,
    vienna_csv="results/benchmark_main.csv",
    faiss_csv="results/ablation_results.csv",
    dsr_csv="results/dsr_global_all_results.csv",
    hybrid_csv="results/dsr_faiss_hybrid_results.csv",
    save_path="results/plots/case_studies/case_side_by_side.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vienna = pd.read_csv(vienna_csv)
    faiss = pd.read_csv(faiss_csv)
    dsr = pd.read_csv(dsr_csv)
    hybrid = pd.read_csv(hybrid_csv)

    vienna = vienna[vienna["method"] == "vienna_only"]
    faiss = faiss[faiss["method"] == "rnafm_faiss_guided_k14"]
    dsr = dsr[dsr["method"] == "rnafm_dsr_global_guided_k14"]
    hybrid = hybrid[hybrid["method"] == "rnafm_dsr_faiss_guided_k14"]

    r0 = vienna[vienna["query_id"] == query_id].iloc[0]
    r1 = faiss[faiss["query_id"] == query_id].iloc[0]
    r2 = dsr[dsr["query_id"] == query_id].iloc[0]
    r3 = hybrid[hybrid["query_id"] == query_id].iloc[0]

    ds_row = _find_dataset_row(query_id, family=_safe_get(r3, "family", None))

    if ds_row is not None:
        sequence = ds_row["sequence"]
        ref_structure = ds_row["structure"]
        family = ds_row["family"]
    else:
        sequence = _safe_get(r3, "sequence", "[sequence not found]")
        ref_structure = _safe_get(r3, "ref_structure", "[ref_structure not found]")
        family = _safe_get(r3, "family", "[family not found]")

    pred_vienna = _safe_get(r0, "pred_structure", "[pred_structure not saved in CSV]")
    pred_faiss = _safe_get(r1, "pred_structure", "[pred_structure not saved in CSV]")
    pred_dsr = _safe_get(r2, "pred_structure", "[pred_structure not saved in CSV]")
    pred_hybrid = _safe_get(r3, "pred_structure", "[pred_structure not saved in CSV]")

    text = []
    text.append(f"Query ID: {query_id}")
    text.append(f"Family: {family}")
    text.append("")
    text.append("Sequence:")
    text.append(wrap_text(sequence))
    text.append("")
    text.append("Reference structure:")
    text.append(wrap_text(ref_structure))
    text.append("")
    text.append(f"ViennaRNA | F1={r0['f1']:.3f}")
    text.append(wrap_text(pred_vienna))
    text.append("")
    text.append(f"RNA-FM + FAISS | F1={r1['f1']:.3f}")
    text.append(wrap_text(pred_faiss))
    text.append("")
    text.append(f"RNA-FM + DSR | F1={r2['f1']:.3f}")
    text.append(wrap_text(pred_dsr))
    text.append("")
    text.append(f"RNA-FM + DSR + FAISS | F1={r3['f1']:.3f}")
    text.append(wrap_text(pred_hybrid))

    plt.figure(figsize=(16, 12))
    plt.axis("off")
    plt.text(0.01, 0.99, "\n".join(text), va="top", family="monospace", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()