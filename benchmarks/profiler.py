"""Execution profiler for benchmarking and training bottlenecks."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch


class ExecutionProfiler:
    """Tracks latency breakdown across data loading, host-to-device transfer,

    GPU forward pass, backward pass/optimizer, and validation evaluation.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

        # Accumulated seconds
        self.t_data_load: float = 0.0
        self.t_h2d_transfer: float = 0.0
        self.t_forward: float = 0.0
        self.t_backward: float = 0.0
        self.t_val_eval: float = 0.0
        self.t_overhead: float = 0.0
        self.t_total_train: float = 0.0

        # Epoch history records
        self.epoch_profiles: List[Dict[str, Any]] = []

    def sync(self):
        """Synchronize CUDA if running on GPU for accurate kernel boundary timing."""
        if self.is_cuda:
            torch.cuda.synchronize()

    def record_epoch(
        self,
        epoch: int,
        t_data: float,
        t_transfer: float,
        t_forward: float,
        t_backward: float,
        t_val: float,
        t_overhead: float,
    ) -> Dict[str, Any]:
        """Record timing breakdown for a single epoch."""
        t_total = t_data + t_transfer + t_forward + t_backward + t_val + t_overhead
        denom = max(t_total, 1e-6)

        rec = {
            "epoch": epoch,
            "data_loading_sec": round(t_data, 3),
            "data_loading_pct": round((t_data / denom) * 100.0, 1),
            "h2d_transfer_sec": round(t_transfer, 3),
            "h2d_transfer_pct": round((t_transfer / denom) * 100.0, 1),
            "forward_sec": round(t_forward, 3),
            "forward_pct": round((t_forward / denom) * 100.0, 1),
            "backward_sec": round(t_backward, 3),
            "backward_pct": round((t_backward / denom) * 100.0, 1),
            "val_eval_sec": round(t_val, 3),
            "val_eval_pct": round((t_val / denom) * 100.0, 1),
            "overhead_sec": round(t_overhead, 3),
            "overhead_pct": round((t_overhead / denom) * 100.0, 1),
            "total_epoch_sec": round(t_total, 3),
        }
        self.epoch_profiles.append(rec)

        self.t_data_load += t_data
        self.t_h2d_transfer += t_transfer
        self.t_forward += t_forward
        self.t_backward += t_backward
        self.t_val_eval += t_val
        self.t_overhead += t_overhead
        self.t_total_train += t_total

        return rec

    def format_epoch_summary(self, rec: Dict[str, Any]) -> str:
        """Format single-epoch timing breakdown string for console output."""
        return (
            f"    [PROFILE] Epoch {rec['epoch']:02d} ({rec['total_epoch_sec']:.1f}s) Breakdown: "
            f"Data Loading (HDF5/Disk): {rec['data_loading_sec']:.1f}s ({rec['data_loading_pct']}%) | "
            f"H2D Transfer: {rec['h2d_transfer_sec']:.1f}s ({rec['h2d_transfer_pct']}%) | "
            f"GPU Fwd: {rec['forward_sec']:.1f}s ({rec['forward_pct']}%) | "
            f"GPU Bwd: {rec['backward_sec']:.1f}s ({rec['backward_pct']}%) | "
            f"Val: {rec['val_eval_sec']:.1f}s ({rec['val_eval_pct']}%)"
        )

    def format_overall_summary(self) -> str:
        """Format cumulative run timing breakdown with bottleneck identification."""
        denom = max(self.t_total_train, 1e-6)
        pct_data = (self.t_data_load / denom) * 100.0
        pct_xfer = (self.t_h2d_transfer / denom) * 100.0
        pct_fwd = (self.t_forward / denom) * 100.0
        pct_bwd = (self.t_backward / denom) * 100.0
        pct_val = (self.t_val_eval / denom) * 100.0
        pct_ovh = (self.t_overhead / denom) * 100.0

        breakdown = {
            "Data Loading (Disk / HDF5 / DataLoader)": pct_data,
            "Host-to-Device Transfer (PCIe)": pct_xfer,
            "GPU Forward Compute": pct_fwd,
            "GPU Backward & Optimizer": pct_bwd,
            "Validation Evaluation": pct_val,
        }
        bottleneck = max(breakdown.items(), key=lambda x: x[1])

        lines = [
            "\n" + "=" * 65,
            "  TIMING & PERFORMANCE PROFILING BREAKDOWN",
            "=" * 65,
            f"  Total Monitored Time:        {self.t_total_train:.1f}s ({self.t_total_train / 60:.2f} min)",
            f"  1. Data Loading (Disk/HDF5): {self.t_data_load:6.1f}s ({pct_data:5.1f}%)",
            f"  2. PCIe H2D Transfer:        {self.t_h2d_transfer:6.1f}s ({pct_xfer:5.1f}%)",
            f"  3. GPU Forward Pass:         {self.t_forward:6.1f}s ({pct_fwd:5.1f}%)",
            f"  4. GPU Loss & Backward:      {self.t_backward:6.1f}s ({pct_bwd:5.1f}%)",
            f"  5. Validation Passes:        {self.t_val_eval:6.1f}s ({pct_val:5.1f}%)",
            f"  6. Overhead & Logging:       {self.t_overhead:6.1f}s ({pct_ovh:5.1f}%)",
            "-" * 65,
            f"  PRIMARY BOTTLENECK: {bottleneck[0]} ({bottleneck[1]:.1f}%)",
        ]
        if pct_data > 40.0:
            lines.append("  --> TIP: Disk I/O is dominating! Use '--in-memory' to cache latents in RAM.")
        elif pct_val > 30.0:
            lines.append("  --> TIP: Validation pass is taking significant time. Increase '--eval-every' (e.g. 5).")
        lines.append("=" * 65 + "\n")
        return "\n".join(lines)

    def export(self, out_csv_path: Union[str, Path], out_plot_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Export timing profile to CSV and generate visualization plot."""
        df = pd.DataFrame(self.epoch_profiles)
        out_csv = Path(out_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[PROFILE] Saved timing profile log to: {out_csv}")

        if out_plot_path and not df.empty:
            self.generate_plot(out_plot_path)

        return df

    def generate_plot(self, out_plot_path: Union[str, Path]):
        """Generate stacked bar chart across epochs and cumulative pie chart."""
        df = pd.DataFrame(self.epoch_profiles)
        if df.empty:
            return

        out_path = Path(out_plot_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

        epochs = df["epoch"]
        ax1.bar(epochs, df["data_loading_sec"], label="Data Loading (HDF5)", color="#e377c2")
        ax1.bar(epochs, df["h2d_transfer_sec"], bottom=df["data_loading_sec"], label="PCIe Transfer", color="#bcbd22")
        bottom_fwd = df["data_loading_sec"] + df["h2d_transfer_sec"]
        ax1.bar(epochs, df["forward_sec"], bottom=bottom_fwd, label="GPU Forward", color="#1f77b4")
        bottom_bwd = bottom_fwd + df["forward_sec"]
        ax1.bar(epochs, df["backward_sec"], bottom=bottom_bwd, label="GPU Backward", color="#ff7f0e")
        bottom_val = bottom_bwd + df["backward_sec"]
        ax1.bar(epochs, df["val_eval_sec"], bottom=bottom_val, label="Validation", color="#2ca02c")

        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Seconds per Epoch", fontsize=10)
        ax1.set_title("Epoch Duration Breakdown Across Training", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax1.legend(loc="upper right", fontsize=8)

        labels = ["Data Loading", "PCIe Transfer", "GPU Forward", "GPU Backward", "Validation", "Overhead"]
        sizes = [self.t_data_load, self.t_h2d_transfer, self.t_forward, self.t_backward, self.t_val_eval, self.t_overhead]
        colors = ["#e377c2", "#bcbd22", "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"]

        valid = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0.05]
        if valid:
            v_labels, v_sizes, v_colors = zip(*valid)
            ax2.pie(
                v_sizes,
                labels=v_labels,
                autopct="%1.1f%%",
                colors=v_colors,
                startangle=140,
                wedgeprops={"edgecolor": "w", "linewidth": 1.5},
            )
            ax2.set_title(
                f"Cumulative Run Time Distribution (Total: {self.t_total_train:.1f}s)",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Generated Timing Profile plot: {out_path}")
