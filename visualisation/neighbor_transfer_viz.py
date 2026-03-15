import os
import matplotlib.pyplot as plt


def plot_structure_strings(
    query_id,
    query_family,
    query_seq,
    ref_structure,
    pred_structure,
    neighbor_structures,
    neighbor_families,
    save_path="results/plots/neighbor_transfer_example.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = []
    lines.append(f"Query ID: {query_id}")
    lines.append(f"Query family: {query_family}")
    lines.append("")
    lines.append("Sequence:")
    lines.append(query_seq)
    lines.append("")
    lines.append("Reference structure:")
    lines.append(ref_structure)
    lines.append("")
    lines.append("Predicted structure:")
    lines.append(pred_structure)
    lines.append("")

    for i, (fam, struct) in enumerate(zip(neighbor_families, neighbor_structures), start=1):
        lines.append(f"Neighbor {i} ({fam}):")
        lines.append(struct)
        lines.append("")

    text = "\n".join(lines)

    plt.figure(figsize=(16, 10))
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", family="monospace", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()