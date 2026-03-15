import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import os
from evaluation.case_study_selection import select_case_studies
from visualisation.structure_transfer_viz import plot_structure_transfer_case


def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    plots_dir = os.path.join(results_dir, "plots", "case_studies")
    os.makedirs(plots_dir, exist_ok=True)

    cases, merged = select_case_studies(
        baseline_csv=os.path.join(results_dir, "benchmark_main.csv"),
        method_csv=os.path.join(results_dir, "dsr_faiss_hybrid_results.csv"),
        baseline_method="vienna_only",
        method_name="rnafm_dsr_faiss_guided_k14",
    )

    cases.to_csv(os.path.join(results_dir, "selected_case_studies.csv"), index=False)
    merged.to_csv(os.path.join(results_dir, "all_case_gains.csv"), index=False)

    print(cases[["case_type", "query_id", "family", "length", "f1_baseline", "f1_method", "gain"]])

    for _, row in cases.iterrows():
        neighbor_families = []
        neighbor_structures = []

        if "neighbor_families" in row and isinstance(row["neighbor_families"], str):
            neighbor_families = row["neighbor_families"].split("|")

        if "neighbor_structures" in row and isinstance(row["neighbor_structures"], str):
            neighbor_structures = row["neighbor_structures"].split("|||")

        save_path = os.path.join(
            plots_dir,
            f"{row['case_type']}_{row['query_id']}.png"
        )

        plot_structure_transfer_case(
            query_id=row["query_id"],
            query_family=row["family"],
            sequence=row["sequence"] if "sequence" in row else "",
            ref_structure=row["ref_structure"] if "ref_structure" in row else "",
            pred_structure=row["pred_structure"] if "pred_structure" in row else "",
            neighbor_families=neighbor_families,
            neighbor_structures=neighbor_structures,
            baseline_structure=None,
            save_path=save_path,
        )

        print("Saved:", save_path)


if __name__ == "__main__":
    main()