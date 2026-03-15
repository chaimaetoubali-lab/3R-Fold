import os
import textwrap
import matplotlib.pyplot as plt
import pandas as pd

from config import get_plots_path


def wrap_text(s, width=120):
    return "\n".join(textwrap.wrap(str(s), width=width))


def plot_structure_transfer_case(
    query_id,
    query_family,
    sequence,
    ref_structure,
    pred_structure,
    neighbor_families,
    neighbor_structures,
    neighbor_scores=None,
    baseline_structure=None,
    save_path=None,
):
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "structure_transfer_case.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = []
    lines.append(f"Query ID: {query_id}")
    lines.append(f"Query family: {query_family}")
    lines.append("")

    lines.append("Sequence:")
    lines.append(wrap_text(sequence))
    lines.append("")

    lines.append("Reference structure:")
    lines.append(wrap_text(ref_structure))
    lines.append("")

    if baseline_structure is not None:
        lines.append("ViennaRNA baseline:")
        lines.append(wrap_text(baseline_structure))
        lines.append("")

    lines.append("3RFold predicted structure:")
    lines.append(wrap_text(pred_structure))
    lines.append("")

    for i, (fam, struct) in enumerate(zip(neighbor_families, neighbor_structures), start=1):
        if neighbor_scores is not None and i - 1 < len(neighbor_scores):
            score_str = f"{neighbor_scores[i-1]:.4f}"
        else:
            score_str = "NA"

        lines.append(f"Neighbor {i} | family={fam} | score={score_str}")
        lines.append(wrap_text(struct))
        lines.append("")

    text = "\n".join(lines)

    plt.figure(figsize=(16, 12))
    plt.axis("off")
    plt.text(
        0.01, 0.99, text,
        va="top",
        family="monospace",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_structure_transfer_from_csv(
    csv_path,
    query_id,
    save_path=None,
):
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "structure_transfer_case.png")
    df = pd.read_csv(csv_path)
    row = df[df["query_id"] == query_id].iloc[0]

    neighbor_families = str(row["neighbor_families"]).split("|") if pd.notna(row["neighbor_families"]) else []
    neighbor_structures = str(row["neighbor_structures"]).split("|||") if "neighbor_structures" in row and pd.notna(row["neighbor_structures"]) else []
    neighbor_scores = None

    baseline_structure = row["baseline_structure"] if "baseline_structure" in row and pd.notna(row["baseline_structure"]) else None

    plot_structure_transfer_case(
        query_id=row["query_id"],
        query_family=row["family"],
        sequence=row["sequence"],
        ref_structure=row["ref_structure"],
        pred_structure=row["pred_structure"],
        neighbor_families=neighbor_families,
        neighbor_structures=neighbor_structures,
        neighbor_scores=neighbor_scores,
        baseline_structure=baseline_structure,
        save_path=save_path,
    )