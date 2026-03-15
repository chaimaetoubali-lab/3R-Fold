import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def _prepare_method_summary():
    bench = pd.read_csv(os.path.join(get_results_path(), "benchmark_main.csv"))
    ab = pd.read_csv(os.path.join(get_results_path(), "ablation_results.csv"))
    dsr = pd.read_csv(os.path.join(get_results_path(), "dsr_global_all_results.csv"))
    hyb = pd.read_csv(os.path.join(get_results_path(), "dsr_faiss_hybrid_results.csv"))

    vienna = bench[bench["method"] == "vienna_only"].copy()
    faiss = ab[ab["method"] == "rnafm_faiss_guided_k14"].copy()
    dsr = dsr[dsr["method"] == "rnafm_dsr_global_guided_k14"].copy()
    hyb = hyb[hyb["method"] == "rnafm_dsr_faiss_guided_k14"].copy()

    rows = [
        {"method": "ViennaRNA", "f1": vienna["f1"].mean()},
        {"method": "RNA-FM + FAISS", "f1": faiss["f1"].mean()},
        {"method": "RNA-FM + DSR", "f1": dsr["f1"].mean()},
        {"method": "RNA-FM + DSR + FAISS", "f1": hyb["f1"].mean()},
    ]
    return pd.DataFrame(rows)


def _prepare_k_sweep():
    ab = pd.read_csv(os.path.join(get_results_path(), "ablation_results.csv"))
    k_methods = ab[ab["method"].str.contains("guided_k", na=False)].copy()
    k_methods = k_methods[~k_methods["method"].str.contains("template", na=False)]
    out = (
        k_methods.groupby("k")
        .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"))
        .reset_index()
        .sort_values("k")
    )
    return out


def _prepare_gain_vs_length():
    gains = pd.read_csv("results/all_case_gains.csv").copy()

    # if length_bin missing, create it
    if "length_bin" not in gains.columns:
        bins = [0, 100, 300, 700, 100000]
        labels = ["<100", "100-299", "300-699", "700+"]
        gains["length_bin"] = pd.cut(gains["length"], bins=bins, labels=labels)

    return gains


def _prepare_gain_vs_purity():
    gains = pd.read_csv("results/all_case_gains.csv").copy()
    return gains


def plot_final_eccb_figure(
    save_path="results/plots/final_eccb_figure.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A — mean F1 comparison
    method_df = _prepare_method_summary()
    ax = axes[0, 0]
    sns.barplot(data=method_df, x="method", y="f1", palette="viridis", ax=ax)
    ax.set_title("A. Mean F1 by method")
    ax.set_xlabel("")
    ax.set_ylabel("Mean F1")
    ax.tick_params(axis="x", rotation=20)

    # Panel B — F1 vs k
    k_df = _prepare_k_sweep()
    ax = axes[0, 1]
    ax.errorbar(k_df["k"], k_df["f1_mean"], yerr=k_df["f1_std"], marker="o")
    ax.set_title("B. Retrieval neighborhood size (k)")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean F1")

    # Panel C — gain vs length bin
    gain_len_df = _prepare_gain_vs_length()
    ax = axes[1, 0]
    sns.boxplot(
        data=gain_len_df,
        x="length_bin",
        y="gain",
        order=["<100", "100-299", "300-699", "700+"],
        ax=ax
    )
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("C. Gain over ViennaRNA vs length")
    ax.set_xlabel("RNA length bin")
    ax.set_ylabel("F1 gain")

    # Panel D — gain vs purity
    gain_purity_df = _prepare_gain_vs_purity()
    ax = axes[1, 1]
    if "family_purity" in gain_purity_df.columns:
        sns.scatterplot(
            data=gain_purity_df,
            x="family_purity",
            y="gain",
            hue="family" if "family" in gain_purity_df.columns else None,
            palette="tab10",
            alpha=0.7,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("Retrieval purity")
    else:
        # fallback if purity not present
        sns.scatterplot(
            data=gain_purity_df,
            x="length",
            y="gain",
            alpha=0.7,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("Length")
    ax.axhline(0, linestyle="--", color="black")
    ax.set_title("D. Gain vs retrieval purity")
    ax.set_ylabel("F1 gain")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", save_path)