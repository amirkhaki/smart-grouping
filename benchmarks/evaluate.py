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

    # Add reference baselines from S0, S2, and C4 if available
    s0_acc = results_df[(results_df["view"] == "S0") & (results_df["quant"] == "quantized")]["top1_acc"].values
    if len(s0_acc) > 0:
        plt.axhline(y=s0_acc[0], color="#ff7f0e", linestyle="--", alpha=0.7, label=f"Slice S0 (16 ch: {s0_acc[0]:.1f}%)")

    s2_acc = results_df[(results_df["view"] == "S2") & (results_df["quant"] == "quantized")]["top1_acc"].values
    if len(s2_acc) > 0:
        plt.axhline(y=s2_acc[0], color="#9467bd", linestyle="-.", alpha=0.7, label=f"Slice S2 (32 ch: {s2_acc[0]:.1f}%)")

    c4_acc = results_df[(results_df["view"] == "C4") & (results_df["quant"] == "quantized")]["top1_acc"].values
    if len(c4_acc) > 0:
        plt.axhline(y=c4_acc[0], color="#1f77b4", linestyle=":", alpha=0.7, label=f"Slice C4 (320 ch: {c4_acc[0]:.1f}%)")

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


def export_training_history(
    history_records: List[Dict[str, Any]],
    out_csv_path: Union[str, Path],
) -> pd.DataFrame:
    """Save epoch-by-epoch training and validation metrics for all heads to CSV."""
    df = pd.DataFrame(history_records)
    out_csv = Path(out_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[HISTORY] Saved full training history to: {out_csv} ({len(df)} epoch-head records)")
    return df


def export_overfitting_analysis(
    master_df: pd.DataFrame,
    history_df: pd.DataFrame,
    out_csv_path: Union[str, Path],
) -> pd.DataFrame:
    """Generate summary table comparing Train Top-1 vs Val Top-1 to diagnose overfitting vs underfitting."""
    records: List[Dict[str, Any]] = []

    head_keys = list(history_df["head_key"].unique()) if "head_key" in history_df.columns else []
    for k in head_keys:
        sub = history_df[history_df["head_key"] == k].sort_values("epoch")
        if sub.empty:
            continue

        row_0 = sub.iloc[0]
        final_train_top1 = float(sub["train_top1"].iloc[-1])
        final_train_loss = float(sub["train_loss"].iloc[-1])
        peak_val_top1 = float(sub["val_top1"].max())
        final_val_top1 = float(sub["val_top1"].iloc[-1])
        final_val_loss = float(sub["val_loss"].iloc[-1])
        num_params = int(row_0.get("num_params", 0))

        test_top1 = None
        if not master_df.empty and "head_key" in master_df.columns:
            m_sub = master_df[master_df["head_key"] == k]
            if not m_sub.empty and "top1_acc" in m_sub.columns:
                test_top1 = float(m_sub["top1_acc"].iloc[0])

        gen_gap = round(final_train_top1 - peak_val_top1, 2)

        if final_train_top1 < 40.0 and gen_gap < 8.0:
            diagnosis = "Underfitting (Capacity Constrained - Can Add Parameters)"
        elif gen_gap <= 8.0:
            diagnosis = "Well Balanced (Generalizing Well - Capacity Headroom)"
        elif gen_gap <= 18.0:
            diagnosis = "Moderate Overfitting (Slight Gap)"
        else:
            diagnosis = "Severe Overfitting (Needs Regularization)"

        records.append({
            "head_key": k,
            "view": row_0.get("view", k),
            "category": row_0.get("category", ""),
            "arch": row_0.get("arch", ""),
            "quant": row_0.get("quant", ""),
            "in_channels": row_0.get("in_channels"),
            "stem_channels": row_0.get("stem_channels"),
            "num_params": num_params,
            "final_train_loss": round(final_train_loss, 4),
            "final_train_top1": round(final_train_top1, 2),
            "final_val_loss": round(final_val_loss, 4),
            "final_val_top1": round(final_val_top1, 2),
            "peak_val_top1": round(peak_val_top1, 2),
            "test_top1": round(test_top1, 2) if test_top1 is not None else None,
            "generalization_gap": gen_gap,
            "diagnosis": diagnosis,
        })

    df = pd.DataFrame(records)
    if not df.empty and "num_params" in df.columns:
        df = df.sort_values(by="num_params", ascending=True).reset_index(drop=True)

    out_csv = Path(out_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[RESULTS] Saved Overfitting Analysis summary to: {out_csv} ({len(df)} heads)")
    return df


def generate_learning_curves_plot(
    history_df: pd.DataFrame,
    out_plot_path: Union[str, Path],
    title: Optional[str] = None,
):
    """Generate multi-panel grid plot of Train Top-1 vs Val Top-1 accuracy across epochs."""
    if history_df.empty or "head_key" not in history_df.columns:
        return

    head_keys = list(history_df["head_key"].unique())
    num_heads = len(head_keys)
    if num_heads == 0:
        return

    if num_heads <= 3:
        ncols = num_heads
        nrows = 1
    elif num_heads == 4:
        ncols, nrows = 2, 2
    elif num_heads <= 6:
        ncols, nrows = 3, 2
    elif num_heads <= 9:
        ncols, nrows = 3, 3
    elif num_heads <= 16:
        ncols = 4
        nrows = (num_heads + 3) // 4
    else:
        ncols = 4
        nrows = (num_heads + 3) // 4

    figsize = (ncols * 4.2, nrows * 3.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, squeeze=False)

    for i, head_key in enumerate(head_keys):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]

        sub = history_df[history_df["head_key"] == head_key].sort_values("epoch")
        epochs = sub["epoch"]
        train_acc = sub["train_top1"]
        val_acc = sub["val_top1"]

        ax.plot(epochs, train_acc, label="Train Top-1", color="#1f77b4", linewidth=2.0)
        ax.plot(epochs, val_acc, label="Val Top-1", color="#ff7f0e", linewidth=2.0)

        # Mark freeze epoch if dynamic early stopping was triggered
        if "is_frozen" in sub.columns and sub["is_frozen"].any():
            frozen_sub = sub[sub["is_frozen"]]
            if not frozen_sub.empty:
                freeze_ep = frozen_sub["epoch"].min()
                ax.axvline(x=freeze_ep, color="#d62728", linestyle="--", alpha=0.8, label=f"Frozen (ep {freeze_ep})")

        final_train = float(train_acc.iloc[-1]) if len(train_acc) > 0 else 0.0
        peak_val = float(val_acc.max()) if len(val_acc) > 0 else 0.0
        gap = final_train - peak_val

        p_val = sub["num_params"].iloc[0] if "num_params" in sub.columns else 0
        p_str = f"{p_val/1e3:.1f}k" if p_val < 1e6 else f"{p_val/1e6:.2f}M"

        ax.set_title(f"{head_key}\n({p_str} params | Gap: {gap:+.1f}%)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Top-1 Acc (%)", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

        y_max = max(train_acc.max() if len(train_acc) else 0, val_acc.max() if len(val_acc) else 0)
        ax.set_ylim(bottom=0, top=min(100, y_max + 10))

    # Hide unused axes
    for j in range(num_heads, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plot_title = title or "Train Top-1 vs. Validation Top-1 Across Epochs (Overfitting Analysis)"
    plt.suptitle(plot_title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_path = Path(out_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated Train vs. Val learning curves: {out_path}")


def generate_loss_curves_plot(
    history_df: pd.DataFrame,
    out_plot_path: Union[str, Path],
    title: Optional[str] = None,
):
    """Generate multi-panel grid plot of Train Loss vs Val Loss across epochs."""
    if history_df.empty or "head_key" not in history_df.columns:
        return

    head_keys = list(history_df["head_key"].unique())
    num_heads = len(head_keys)
    if num_heads == 0:
        return

    if num_heads <= 3:
        ncols = num_heads
        nrows = 1
    elif num_heads == 4:
        ncols, nrows = 2, 2
    elif num_heads <= 6:
        ncols, nrows = 3, 2
    elif num_heads <= 9:
        ncols, nrows = 3, 3
    elif num_heads <= 16:
        ncols = 4
        nrows = (num_heads + 3) // 4
    else:
        ncols = 4
        nrows = (num_heads + 3) // 4

    figsize = (ncols * 4.2, nrows * 3.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, squeeze=False)

    for i, head_key in enumerate(head_keys):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]

        sub = history_df[history_df["head_key"] == head_key].sort_values("epoch")
        epochs = sub["epoch"]
        train_loss = sub["train_loss"]
        val_loss = sub["val_loss"]

        ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2.0)
        ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2.0)

        if "is_frozen" in sub.columns and sub["is_frozen"].any():
            frozen_sub = sub[sub["is_frozen"]]
            if not frozen_sub.empty:
                freeze_ep = frozen_sub["epoch"].min()
                ax.axvline(x=freeze_ep, color="#d62728", linestyle="--", alpha=0.8, label=f"Frozen (ep {freeze_ep})")

        p_val = sub["num_params"].iloc[0] if "num_params" in sub.columns else 0
        p_str = f"{p_val/1e3:.1f}k" if p_val < 1e6 else f"{p_val/1e6:.2f}M"

        ax.set_title(f"{head_key}\n({p_str} params)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    for j in range(num_heads, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plot_title = title or "Train Loss vs. Validation Loss Across Epochs"
    plt.suptitle(plot_title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_path = Path(out_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated Train vs. Val loss curves: {out_path}")


def generate_generalization_gap_plot(
    overfitting_df: pd.DataFrame,
    out_plot_path: Union[str, Path],
    title: Optional[str] = None,
):
    """Generate diagnostic chart showing Generalization Gap (Train Top-1 vs Peak Val Top-1) by parameter volume."""
    if overfitting_df.empty:
        return

    df = overfitting_df.sort_values(by="num_params", ascending=True).copy()
    num_heads = len(df)
    plt.figure(figsize=(10, max(5, num_heads * 0.45 + 2)), dpi=150)

    y_pos = list(range(num_heads))
    labels = []
    for _, row in df.iterrows():
        p_val = row["num_params"]
        p_str = f"{p_val/1e3:.1f}k" if p_val < 1e6 else f"{p_val/1e6:.2f}M"
        labels.append(f"{row['head_key']} ({p_str})")

    # Plot connecting dumbbell lines
    for i, (_, row) in enumerate(df.iterrows()):
        train = row["final_train_top1"]
        val = row["peak_val_top1"]
        gap = row["generalization_gap"]

        # Color-code gap: green if <= 8%, orange if <= 18%, red if > 18%
        line_color = "#2ca02c" if gap <= 8.0 else ("#ff7f0e" if gap <= 18.0 else "#d62728")
        plt.plot([val, train], [i, i], color=line_color, linewidth=2.5, zorder=2)
        plt.text(
            max(train, val) + 1.2, i, f"Gap: {gap:+.1f}% ({row['diagnosis'].split()[0]})",
            va="center", ha="left", fontsize=8, color=line_color, fontweight="bold"
        )

    plt.scatter(df["final_train_top1"], y_pos, color="#1f77b4", s=70, zorder=3, label="Final Train Top-1")
    plt.scatter(df["peak_val_top1"], y_pos, color="#ff7f0e", s=70, zorder=3, marker="s", label="Peak Val Top-1")

    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel("Top-1 Accuracy (%)", fontsize=11)
    plt.ylabel("Classification Head (Parameter Volume)", fontsize=11)
    plot_title = title or "Generalization Gap (Train vs. Val Accuracy) by Model Capacity"
    plt.title(plot_title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6, axis="x")
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    max_acc = max(df["final_train_top1"].max(), df["peak_val_top1"].max()) if not df.empty else 50
    plt.xlim(0, min(105, max(max_acc + 30, 40)))
    plt.tight_layout()

    out_path = Path(out_plot_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generated Generalization Gap diagnostic chart: {out_path}")
