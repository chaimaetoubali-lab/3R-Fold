import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_results_path, get_plots_path


def _compute_family_order(df, order_by="vienna_only"):
    """
    Compute a consistent family order across all panels.
    Default: order by ViennaRNA mean F1.
    """
    sub = df[df["method"] == order_by].copy()

    if sub.empty:
        # fallback: alphabetical
        return sorted(df["family"].unique())

    order = (
        sub.groupby("family")["f1"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return order


def plot_family_benchmark_panel_ordered(
    csv_path=None,
    methods=None,
    order_by="vienna_only",
    exclude_families=None,
    save_path=None,
):
    if csv_path is None:
        csv_path = os.path.join(get_results_path(), "final_family_benchmark_methods.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "final_family_benchmark_panel_ordered.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    if exclude_families is not None:
        df = df[~df["family"].isin(exclude_families)].copy()

    if methods is None:
        methods = [
            "vienna_only",
            "rnafm_faiss_guided_k14",
            "rnafm_dsr_global_guided_k14",
            "rnafm_dsr_faiss_guided_k14",
        ]

    df = df[df["method"].isin(methods)].copy()

    summary = (
        df.groupby(["method", "family"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n=("query_id", "count"),
        )
        .reset_index()
    )

    family_order = _compute_family_order(df, order_by=order_by)

    palette = sns.color_palette("tab10", n_colors=len(family_order))
    family_to_color = {fam: palette[i] for i, fam in enumerate(family_order)}

    n_methods = len(methods)
    ncols = 2
    nrows = math.ceil(n_methods / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, 4.8 * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    ymax = min(1.0, max(0.05, summary["f1_mean"].max() + 0.08))

    for ax, method in zip(axes, methods):
        sub = summary[summary["method"] == method].copy()
        sub["family"] = pd.Categorical(sub["family"], categories=family_order, ordered=True)
        sub = sub.sort_values("family")

        colors = [family_to_color[f] for f in sub["family"]]

        ax.bar(
            sub["family"],
            sub["f1_mean"],
            yerr=sub["f1_std"].fillna(0),
            color=colors,
            capsize=3,
            alpha=0.9,
        )

        ax.set_title(method)
        ax.set_ylabel("Mean F1")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, ymax)

    # hide unused axes
    for i in range(len(methods), len(axes)):
        axes[i].axis("off")

    handles = [
        plt.Line2D([0], [0], color=family_to_color[f], lw=8, label=f)
        for f in family_order
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Family"
    )

    fig.suptitle(
        f"Family-wise F1 panel comparison (ordered by {order_by})",
        fontsize=16,
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_family_gain_panel(
    csv_path=None,
    baseline_method="vienna_only",
    compare_methods=None,
    exclude_families=None,
    save_path=None,
):
    """
    Plot gain over ViennaRNA per family for each non-baseline method.
    """
    if csv_path is None:
        csv_path = os.path.join(get_results_path(), "final_family_benchmark_methods.csv")
    if save_path is None:
        save_path = os.path.join(get_plots_path(), "family_gain_panel.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    if exclude_families is not None:
        df = df[~df["family"].isin(exclude_families)].copy()

    if compare_methods is None:
        compare_methods = [
            "rnafm_faiss_guided_k14",
            "rnafm_dsr_global_guided_k14",
            "rnafm_dsr_faiss_guided_k14",
        ]

    summary = (
        df.groupby(["method", "family"])
        .agg(f1_mean=("f1", "mean"))
        .reset_index()
    )

    base = summary[summary["method"] == baseline_method][["family", "f1_mean"]].rename(
        columns={"f1_mean": "baseline_f1"}
    )

    comp = summary[summary["method"].isin(compare_methods)].copy()
    comp = comp.merge(base, on="family", how="inner")
    comp["gain"] = comp["f1_mean"] - comp["baseline_f1"]

    family_order = (
        base.sort_values("baseline_f1", ascending=False)["family"]
        .tolist()
    )

    palette = sns.color_palette("tab10", n_colors=len(family_order))
    family_to_color = {fam: palette[i] for i, fam in enumerate(family_order)}

    n_methods = len(compare_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), squeeze=False)
    axes = axes.flatten()

    for ax, method in zip(axes, compare_methods):
        sub = comp[comp["method"] == method].copy()
        sub["family"] = pd.Categorical(sub["family"], categories=family_order, ordered=True)
        sub = sub.sort_values("family")

        colors = [family_to_color[f] for f in sub["family"]]

        ax.bar(sub["family"], sub["gain"], color=colors, alpha=0.9)
        ax.axhline(0, linestyle="--", color="black")
        ax.set_title(f"{method} gain over {baseline_method}")
        ax.set_ylabel("F1 gain")
        ax.tick_params(axis="x", rotation=45)

    handles = [
        plt.Line2D([0], [0], color=family_to_color[f], lw=8, label=f)
        for f in family_order
    ]
    fig.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Family"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()