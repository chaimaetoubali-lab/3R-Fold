import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _add_length_bin(df):
    bins = [0, 100, 300, 700, 100000]
    labels = ["<100", "100-299", "300-699", "700+"]
    out = df.copy()
    out["length_bin"] = pd.cut(out["length"], bins=bins, labels=labels)
    return out


def _compute_length_order(df, order_by="vienna_only"):
    sub = df[df["method"] == order_by].copy()

    if sub.empty:
        return ["<100", "100-299", "300-699", "700+"]

    sub = _add_length_bin(sub)

    order = (
        sub.groupby("length_bin", observed=False)["f1"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return order


def plot_length_benchmark_panel_ordered(
    csv_path="results/final_family_benchmark_methods.csv",
    methods=None,
    order_by="vienna_only",
    save_path="results/plots/final_length_benchmark_panel_ordered.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _add_length_bin(df)

    if methods is None:
        methods = [
            "vienna_only",
            "rnafm_faiss_guided_k14",
            "rnafm_dsr_global_guided_k14",
            "rnafm_dsr_faiss_guided_k14",
        ]

    df = df[df["method"].isin(methods)].copy()

    summary = (
        df.groupby(["method", "length_bin"], observed=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n=("query_id", "count"),
        )
        .reset_index()
    )

    length_order = _compute_length_order(df, order_by=order_by)

    palette = sns.color_palette("Set2", n_colors=len(length_order))
    length_to_color = {lb: palette[i] for i, lb in enumerate(length_order)}

    n_methods = len(methods)
    ncols = 2
    nrows = math.ceil(n_methods / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4.8 * nrows),
        squeeze=False
    )
    axes = axes.flatten()

    ymax = min(1.0, max(0.05, summary["f1_mean"].max() + 0.08))

    for ax, method in zip(axes, methods):
        sub = summary[summary["method"] == method].copy()
        sub["length_bin"] = pd.Categorical(sub["length_bin"], categories=length_order, ordered=True)
        sub = sub.sort_values("length_bin")

        colors = [length_to_color[str(lb)] for lb in sub["length_bin"]]

        ax.bar(
            sub["length_bin"].astype(str),
            sub["f1_mean"],
            yerr=sub["f1_std"].fillna(0),
            color=colors,
            capsize=3,
            alpha=0.9,
        )

        ax.set_title(method)
        ax.set_ylabel("Mean F1")
        ax.set_xlabel("")
        ax.set_ylim(0, ymax)

    # hide unused axes
    for i in range(len(methods), len(axes)):
        axes[i].axis("off")

    handles = [
        plt.Line2D([0], [0], color=length_to_color[lb], lw=8, label=lb)
        for lb in length_order
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Length bin"
    )

    fig.suptitle(
        f"Length-wise F1 panel comparison (ordered by {order_by})",
        fontsize=16,
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_length_gain_panel(
    csv_path="results/final_family_benchmark_methods.csv",
    baseline_method="vienna_only",
    compare_methods=None,
    save_path="results/plots/length_gain_panel.png",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _add_length_bin(df)

    if compare_methods is None:
        compare_methods = [
            "rnafm_faiss_guided_k14",
            "rnafm_dsr_global_guided_k14",
            "rnafm_dsr_faiss_guided_k14",
        ]

    summary = (
        df.groupby(["method", "length_bin"], observed=False)
        .agg(f1_mean=("f1", "mean"))
        .reset_index()
    )

    base = summary[summary["method"] == baseline_method][["length_bin", "f1_mean"]].rename(
        columns={"f1_mean": "baseline_f1"}
    )

    comp = summary[summary["method"].isin(compare_methods)].copy()
    comp = comp.merge(base, on="length_bin", how="inner")
    comp["gain"] = comp["f1_mean"] - comp["baseline_f1"]

    length_order = (
        base.sort_values("baseline_f1", ascending=False)["length_bin"]
        .astype(str)
        .tolist()
    )

    palette = sns.color_palette("Set2", n_colors=len(length_order))
    length_to_color = {lb: palette[i] for i, lb in enumerate(length_order)}

    n_methods = len(compare_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5.8 * n_methods, 5), squeeze=False)
    axes = axes.flatten()

    for ax, method in zip(axes, compare_methods):
        sub = comp[comp["method"] == method].copy()
        sub["length_bin"] = sub["length_bin"].astype(str)
        sub["length_bin"] = pd.Categorical(sub["length_bin"], categories=length_order, ordered=True)
        sub = sub.sort_values("length_bin")

        colors = [length_to_color[str(lb)] for lb in sub["length_bin"]]

        ax.bar(sub["length_bin"].astype(str), sub["gain"], color=colors, alpha=0.9)
        ax.axhline(0, linestyle="--", color="black")
        ax.set_title(f"{method} gain over {baseline_method}")
        ax.set_ylabel("F1 gain")
        ax.set_xlabel("Length bin")

    handles = [
        plt.Line2D([0], [0], color=length_to_color[lb], lw=8, label=lb)
        for lb in length_order
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Length bin"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()