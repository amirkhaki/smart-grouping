"""Evaluation helpers, metrics serialization, and Rate-Accuracy Pareto plotting."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def export_master_results(
    records: List[Dict[str, Any]],
    out_csv_path: Union[str, Path],
    out_json_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Save benchmarking evaluation records into master_results.csv and metrics.json."""
    df = pd.DataFrame(records)

    # Sort in standard benchmarking presentation order
    sort_cols = [c for c in ["lmbda", "category", "view", "arch", "quant"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    out_csv = Path(out_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[RESULTS] Saved master results to: {out_csv} ({len(df)} configurations)")

    if out_json_path:
        out_json = Path(out_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[RESULTS] Saved JSON metrics to: {out_json}")

    return df


def generate_pareto_plot(
    results_df: pd.DataFrame,
    out_plot_path: Union[str, Path],
    title: str = "Rate-Accuracy Pareto Frontier: Frozen ELIC on Caltech-256",
):
    """Generate Rate-Accuracy Pareto curve for cumulative slices across probe models."""
    plt.figure(figsize=(10, 6), dpi=150)

    # Filter to cumulative slices where progressive bitrate is decodable
    cum_df = results_df[results_df["category"] == "cumulative"].copy()

    # Style mapping for configurations
    styles = {
        ("latent_head", "quantized"): {"color": "#1f77b4", "marker": "o", "label": "LatentHead (Quantized y_hat)", "linestyle": "-"},
        ("latent_head", "continuous"): {"color": "#1f77b4", "marker": "s", "label": "LatentHead (Continuous y)", "linestyle": "--"},
        ("linear_probe", "quantized"): {"color": "#ff7f0e", "marker": "^", "label": "LinearProbe (Quantized y_hat)", "linestyle": "-"},
        ("linear_probe", "continuous"): {"color": "#ff7f0e", "marker": "v", "label": "LinearProbe (Continuous y)", "linestyle": "--"},
    }

    # Dynamically discover any custom architectures (e.g. latent_head_w8, latent_head_w16, etc.)
    all_arch_quants = cum_df[["arch", "quant"]].drop_duplicates().values
    palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    p_idx = 0
    for arch, quant in all_arch_quants:
        key = (arch, quant)
        if key not in styles:
            styles[key] = {
                "color": palette[p_idx % len(palette)],
                "marker": "D" if quant == "quantized" else "X",
                "label": f"{arch} ({quant})",
                "linestyle": "-" if quant == "quantized" else "--",
            }
            p_idx += 1

    has_data = False
    for (arch, quant), style in styles.items():
        subset = cum_df[(cum_df["arch"] == arch) & (cum_df["quant"] == quant)]
        if not subset.empty:
            has_data = True
            subset = subset.sort_values(by="bpp")
            plt.plot(
                subset["bpp"],
                subset["top1_acc"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=8,
                label=style["label"],
            )
            # Annotate slice names
            for _, row in subset.iterrows():
                plt.annotate(
                    row["view"],
                    (row["bpp"], row["top1_acc"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )

    if not has_data:
        plt.text(
            0.5, 0.5, "No cumulative slice data available",
            horizontalalignment="center", verticalalignment="center",
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Cumulative Decodable Rate (bpp)", fontsize=12)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_path = Path(out_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Generated Rate-Accuracy Pareto curve: {out_path}")


def generate_stem_ablation_plot(
    results_df: pd.DataFrame,
    out_plot_path: Union[str, Path],
    view: str = "C4",
    title: Optional[str] = None,
):
    """Generate stem width ablation plot (Stem Width vs Top-1 Accuracy) for a given slice view."""
    df_view = results_df[results_df["view"] == view].copy()
    if df_view.empty or "stem_channels" not in df_view.columns:
        return

    # Filter to rows that have stem_channels defined
    df_valid = df_view[df_view["stem_channels"].notna()].copy()
    if df_valid["stem_channels"].nunique() <= 1:
        return  # No ablation to plot if only 1 stem width

    plt.figure(figsize=(9, 5.5), dpi=150)
    colors = {"quantized": "#1f77b4", "continuous": "#2ca02c"}
    markers = {"quantized": "o", "continuous": "s"}
    linestyles = {"quantized": "-", "continuous": "--"}

    for quant in ["quantized", "continuous"]:
        sub = df_valid[df_valid["quant"] == quant].sort_values(by="stem_channels")
        if not sub.empty:
            label = f"{view} ({quant.capitalize()})"
            plt.plot(
                sub["stem_channels"],
                sub["top1_acc"],
                color=colors.get(quant, "blue"),
                marker=markers.get(quant, "o"),
                linestyle=linestyles.get(quant, "-"),
                linewidth=2.2,
                markersize=8,
                label=label,
            )
            for _, row in sub.iterrows():
                p_val = row.get("num_params", 0)
                p_str = f"{p_val/1e3:.1f}k" if p_val < 1e6 else f"{p_val/1e6:.2f}M"
                plt.annotate(
                    f"{row['top1_acc']:.1f}%\n({p_str})",
                    (row["stem_channels"], row["top1_acc"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

    plot_title = title or f"Stem Width Capacity Ablation on {view} (Frozen ELIC, Caltech-256)"
    widths = sorted(df_valid["stem_channels"].astype(int).unique())
    plt.xscale("log", base=2)
    plt.xticks(widths, [str(w) for w in widths])
    plt.xlabel("Stem Channel Width (W)", fontsize=12)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.title(plot_title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_path = Path(out_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Generated Stem Width ablation curve: {out_path}")


def generate_least_sparse_plot(
    results_df: pd.DataFrame,
    out_plot_path: Union[str, Path],
    title: str = "Least-Sparse Channel Subsets vs. Classification Accuracy",
):
    """Plot Top-1 accuracy for dynamically selected least-sparse channel subsets."""
    ls_df = results_df[results_df["category"] == "least_sparse"].copy()
    if ls_df.empty:
        return

    plt.figure(figsize=(9, 5.5), dpi=150)
    ls_sorted = ls_df.sort_values(by="in_channels")

    plt.plot(
        ls_sorted["in_channels"],
        ls_sorted["top1_acc"],
        color="#2ca02c",
        marker="o",
        linewidth=2.2,
        markersize=9,
        label="Least-Sparse Channels (Quantized y_hat)",
    )

    for _, row in ls_sorted.iterrows():
        plt.annotate(
            f"{row['view']}: {row['top1_acc']:.1f}%\n({row['in_channels']} ch)",
            (row["in_channels"], row["top1_acc"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Add reference baselines from S0 and C4 if available
    s0_acc = results_df[(results_df["view"] == "S0") & (results_df["quant"] == "quantized")]["top1_acc"].values
    if len(s0_acc) > 0:
        plt.axhline(y=s0_acc[0], color="#ff7f0e", linestyle="--", alpha=0.7, label=f"Native S0 (16 ch: {s0_acc[0]:.1f}%)")

    c4_acc = results_df[(results_df["view"] == "C4") & (results_df["quant"] == "quantized")]["top1_acc"].values
    if len(c4_acc) > 0:
        plt.axhline(y=c4_acc[0], color="#1f77b4", linestyle=":", alpha=0.7, label=f"Native C4 (320 ch: {c4_acc[0]:.1f}%)")

    channels = sorted(ls_sorted["in_channels"].unique())
    plt.xticks(channels, [f"{c} ch" for c in channels])
    plt.xlabel("Number of Least-Sparse Channels (K)", fontsize=12)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_path = Path(out_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Generated Least-Sparse channel curve: {out_path}")
