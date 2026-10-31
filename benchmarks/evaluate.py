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
