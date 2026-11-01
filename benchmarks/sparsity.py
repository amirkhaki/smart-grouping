"""Channel sparsity and activation density analysis for quantized ELIC latents."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np

from benchmarks.config import DEFAULT_CACHE_FILENAME, SLICE_SPECS


def get_channel_slice(ch: int) -> str:
    """Map a channel index (0..319) to its originating ELIC slice view."""
    if 0 <= ch < 16:
        return "S0"
    elif 16 <= ch < 32:
        return "S1"
    elif 32 <= ch < 64:
        return "S2"
    elif 64 <= ch < 128:
        return "S3"
    elif 128 <= ch < 320:
        return "S4"
    return "Unknown"


def get_channel_slice_info(ch: int) -> Tuple[str, int]:
    """Return originating slice name and relative index within that slice."""
    slice_name = get_channel_slice(ch)
    if slice_name in SLICE_SPECS:
        start_ch = SLICE_SPECS[slice_name].start_ch
        return slice_name, ch - start_ch
    return slice_name, ch


def compute_channel_densities(
    h5_source: Union[str, Path, h5py.File],
    split: str = "train",
    batch_size: int = 1024,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute empirical non-zero activation density for each of the 320 channels on quantized latents.

    Density = (Count of non-zero symbols in channel c) / (Total elements in channel c).
    Sparsity = 1.0 - Density.

    Args:
        h5_source: Path to HDF5 file or open h5py.File object.
        split: Dataset split ('train' or 'test').
        batch_size: Batch size for memory-efficient streaming.
        max_samples: Optional upper limit on samples for faster approximation.

    Returns:
        np.ndarray of shape (320,) containing non-zero densities in [0.0, 1.0].
    """
    should_close = False
    if isinstance(h5_source, (str, Path)):
        f = h5py.File(h5_source, "r", swmr=True)
        should_close = True
    else:
        f = h5_source

    try:
        ds = f[f"{split}/y_quantized"]
        total_samples = ds.shape[0]
        if max_samples:
            total_samples = min(total_samples, max_samples)

        num_channels = ds.shape[1]  # 320
        spatial_elements = ds.shape[2] * ds.shape[3]  # 16 * 16 = 256

        non_zero_counts = np.zeros(num_channels, dtype=np.int64)
        processed = 0

        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch = ds[i:end_idx]  # (B, 320, 16, 16)
            # Count non-zero symbols per channel
            non_zero_counts += np.count_nonzero(batch, axis=(0, 2, 3))
            processed += (end_idx - i)

        total_elements_per_ch = processed * spatial_elements
        densities = non_zero_counts.astype(np.float64) / max(total_elements_per_ch, 1)
        return densities

    finally:
        if should_close:
            f.close()


def rank_channels_by_density(densities: np.ndarray) -> List[Dict[str, Any]]:
    """Rank channels from highest activation density (least sparse) to lowest.

    Returns:
        List of dicts with keys: rank, channel, slice, rel_idx, density, sparsity.
    """
    sorted_indices = np.argsort(-densities)
    ranked = []
    for rank, ch_idx in enumerate(sorted_indices, start=1):
        slice_name, rel_idx = get_channel_slice_info(ch_idx)
        dens = float(densities[ch_idx])
        ranked.append({
            "rank": rank,
            "channel": int(ch_idx),
            "slice": slice_name,
            "rel_idx": int(rel_idx),
            "density": round(dens, 4),
            "density_pct": round(dens * 100.0, 2),
            "sparsity_pct": round((1.0 - dens) * 100.0, 2),
        })
    return ranked


def format_density_summary(
    ranked_channels: List[Dict[str, Any]],
    top_k_list: List[int] = (8, 16, 32),
) -> str:
    """Format an informative terminal summary report of channel density rankings."""
    lines = []
    lines.append("=" * 80)
    lines.append("           ELIC QUANTIZED LATENT ACTIVATION DENSITY RANKING")
    lines.append("=" * 80)

    # 1. Summary of Top-K subsets
    lines.append("\n[TOP-K CHANNEL SUBSETS SUMMARY]")
    for k in top_k_list:
        sub = ranked_channels[:k]
        channels_list = [r["channel"] for r in sub]
        ch_str = ", ".join(f"ch_{c}({r['slice']})" for c, r in zip(channels_list, sub))

        # Count representation from each slice
        slice_counts: Dict[str, int] = {}
        for r in sub:
            slice_counts[r["slice"]] = slice_counts.get(r["slice"], 0) + 1

        dist_str = ", ".join(f"{s}: {c} ({c/k*100:.1f}%)" for s, c in sorted(slice_counts.items()))

        lines.append(f"\n  Top {k} Least-Sparse Channels (Total: {k}):")
        lines.append(f"    Slice Distribution: {dist_str}")
        lines.append(f"    Channel Indices:    {channels_list}")
        lines.append(f"    Details:            {ch_str}")

    # 2. Detailed Table for Top 32 channels
    max_table_rows = max(top_k_list) if top_k_list else 32
    lines.append(f"\n[DETAILED RANKING (Top {max_table_rows} Channels)]")
    lines.append(f"  {'Rank':<6} {'Channel':<9} {'Slice':<8} {'Slice Rel Idx':<15} {'Density (%)':<13} {'Sparsity (%)':<13}")
    lines.append("  " + "-" * 70)

    for r in ranked_channels[:max_table_rows]:
        lines.append(
            f"  #{r['rank']:<5} ch_{r['channel']:<6} {r['slice']:<8} "
            f"+{r['rel_idx']:<14} {r['density_pct']:>6.2f}%       {r['sparsity_pct']:>6.2f}%"
        )

    lines.append("=" * 80)
    return "\n".join(lines)
