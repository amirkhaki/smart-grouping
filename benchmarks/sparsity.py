"""Channel sparsity, activation density, and per-channel rate analysis for ELIC latents."""

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


def compute_channel_rates_from_latents(
    model: Any,
    h5_source: Union[str, Path, h5py.File],
    split: str = "train",
    batch_size: int = 128,
    max_samples: Optional[int] = None,
    device: Optional[Union[str, Any]] = None,
) -> np.ndarray:
    """Compute empirical per-channel transmission bitrate from continuous latents and frozen ELIC model.

    Evaluates the entropy model likelihood p(y_c) for each of the 320 channels on-the-fly
    without needing raw RGB input images.
    Rate_c = E[-log2 p(y_c)] / (256 * 256) in bits per pixel (bpp).

    Args:
        model: Pretrained TestModel or ELICFeatureExtractor in eval mode.
        h5_source: Path to HDF5 file or open h5py.File containing 'latents_continuous'.
        split: Dataset split ('train', 'test', or 'all').
        batch_size: Batch size for GPU streaming.
        max_samples: Optional sample cap for faster estimation.
        device: Device to run likelihood estimation on.

    Returns:
        np.ndarray of shape (320,) containing mean bpp per channel.
    """
    import torch

    inner_model = getattr(model, "model", model)
    inner_model.eval()

    if device is None:
        try:
            device = next(inner_model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    should_close = False
    if isinstance(h5_source, (str, Path)):
        f = h5py.File(str(h5_source), "r", swmr=True)
        should_close = True
    else:
        f = h5_source

    try:
        if "latents_continuous" not in f:
            raise KeyError(f"Archive missing 'latents_continuous'. Keys present: {list(f.keys())}")

        ds = f["latents_continuous"]
        is_train = f["is_train"][:] if "is_train" in f else np.ones(ds.shape[0], dtype=bool)

        if split == "train":
            split_mask = is_train
        elif split == "test":
            split_mask = ~is_train
        else:
            split_mask = np.ones(len(is_train), dtype=bool)

        indices = np.where(split_mask)[0]
        if max_samples is not None and max_samples < len(indices):
            indices = indices[:max_samples]

        N_eval = len(indices)
        if N_eval == 0:
            raise ValueError(f"No samples matched split='{split}' in {h5_source}")

        num_channels = ds.shape[1]  # 320
        rates_sum = np.zeros(num_channels, dtype=np.float64)
        processed = 0

        num_pixels = 256 * 256  # Standard Caltech-256 evaluation crop area

        with torch.no_grad():
            for i in range(0, N_eval, batch_size):
                batch_idx = indices[i : i + batch_size]
                # Slice HDF5 data
                # Sort indices for h5py indexing compatibility if necessary
                if np.all(np.diff(batch_idx) == 1):
                    raw_batch = ds[batch_idx[0] : batch_idx[-1] + 1]
                else:
                    sorted_order = np.argsort(batch_idx)
                    sorted_idx = batch_idx[sorted_order]
                    raw_batch = ds[sorted_idx]
                    inv_order = np.empty_like(sorted_order)
                    inv_order[sorted_order] = np.arange(len(sorted_order))
                    raw_batch = raw_batch[inv_order]

                y = torch.from_numpy(raw_batch).float().to(device)
                B, C, H_y, W_y = y.size()

                # 1. Hyperprior forward pass and deterministic rounding
                z = inner_model.h_a(y)
                z_offset = inner_model.entropy_bottleneck._get_medians()
                z_hat = torch.round(z - z_offset) + z_offset

                # 2. Spatial parameter prior synthesis
                latent_means, latent_scales = inner_model.h_s(z_hat).chunk(2, 1)

                # 3. Checkerboard pattern masks
                anchor = torch.zeros_like(y)
                non_anchor = torch.zeros_like(y)
                anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
                anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
                non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
                non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

                y_slices = torch.split(y, inner_model.groups[1:], 1)
                anchor_split = torch.split(anchor, inner_model.groups[1:], 1)
                non_anchor_split = torch.split(non_anchor, inner_model.groups[1:], 1)
                ctx_params_anchor_split = torch.split(
                    torch.zeros(B, C * 2, H_y, W_y, device=device),
                    [2 * g for g in inner_model.groups[1:]],
                    1,
                )

                y_hat_slices = []
                batch_channel_rates = []

                # 4. Contextual slices
                for slice_index, y_slice in enumerate(y_slices):
                    if slice_index == 0:
                        support = torch.concat([latent_means, latent_scales], dim=1)
                    elif slice_index == 1:
                        support_slices = y_hat_slices[0]
                        support_slices_ch = inner_model.cc_transforms[slice_index - 1](support_slices)
                        mean_ch, scale_ch = support_slices_ch.chunk(2, 1)
                        support = torch.concat([mean_ch, scale_ch, latent_means, latent_scales], dim=1)
                    else:
                        support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                        support_slices_ch = inner_model.cc_transforms[slice_index - 1](support_slices)
                        mean_ch, scale_ch = support_slices_ch.chunk(2, 1)
                        support = torch.concat([mean_ch, scale_ch, latent_means, latent_scales], dim=1)

                    # Step 1: Anchor decoding
                    means_anchor, scales_anchor = inner_model.ParamAggregation[slice_index](
                        torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
                    ).chunk(2, 1)

                    y_a_quant = torch.round(anchor_split[slice_index] - means_anchor) + means_anchor
                    y_a_quant[:, :, 0::2, 1::2] = 0
                    y_a_quant[:, :, 1::2, 0::2] = 0

                    # Step 2: Non-anchor decoding
                    masked_ctx = inner_model.context_prediction[slice_index](y_a_quant)
                    means_non_anchor, scales_non_anchor = inner_model.ParamAggregation[slice_index](
                        torch.concat([masked_ctx, support], dim=1)
                    ).chunk(2, 1)

                    y_na_quant = torch.round(non_anchor_split[slice_index] - means_non_anchor) + means_non_anchor
                    y_na_quant[:, :, 0::2, 0::2] = 0
                    y_na_quant[:, :, 1::2, 1::2] = 0

                    y_hat_slice = y_a_quant + y_na_quant
                    y_hat_slices.append(y_hat_slice)

                    # Full scales and means
                    means_full = torch.zeros_like(y_slice)
                    scales_full = torch.zeros_like(y_slice)
                    means_full[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
                    means_full[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
                    means_full[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
                    means_full[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

                    scales_full[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
                    scales_full[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
                    scales_full[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
                    scales_full[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]

                    _, slice_likelihood = inner_model.gaussian_conditional(
                        y_slice, scales_full, means=means_full
                    )
                    # Sum spatial bits across (H_y, W_y) divided by input pixel count: (B, C_slice)
                    ch_rate_slice = -torch.log2(slice_likelihood).sum(dim=(2, 3)) / num_pixels
                    batch_channel_rates.append(ch_rate_slice.detach().cpu())

                batch_rates = torch.cat(batch_channel_rates, dim=1).numpy()  # (B, 320)
                rates_sum += batch_rates.sum(axis=0)
                processed += B

        mean_channel_rates = rates_sum / max(processed, 1)
        return mean_channel_rates

    finally:
        if should_close:
            f.close()


def rank_channels_by_rate(channel_rates: np.ndarray) -> List[Dict[str, Any]]:
    """Rank channels from highest transmission bitrate (bpp) to lowest.

    Returns:
        List of dicts with keys: rank, channel, slice, rel_idx, rate_bpp.
    """
    sorted_indices = np.argsort(-channel_rates)
    ranked = []
    for rank, ch_idx in enumerate(sorted_indices, start=1):
        slice_name, rel_idx = get_channel_slice_info(ch_idx)
        rate = float(channel_rates[ch_idx])
        ranked.append({
            "rank": rank,
            "channel": int(ch_idx),
            "slice": slice_name,
            "rel_idx": int(rel_idx),
            "rate_bpp": round(rate, 6),
        })
    return ranked


def format_rate_summary(
    ranked_channels: List[Dict[str, Any]],
    top_k_list: List[int] = (8, 16, 32, 64, 128),
) -> str:
    """Format an informative terminal summary report of channel rate rankings."""
    lines = []
    lines.append("=" * 80)
    lines.append("              ELIC PER-CHANNEL TRANSMISSION RATE RANKING")
    lines.append("=" * 80)

    # 1. Summary of Top-K subsets
    lines.append("\n[TOP-K RATE CHANNEL SUBSETS SUMMARY]")
    for k in top_k_list:
        sub = ranked_channels[:k]
        channels_list = [r["channel"] for r in sub]
        total_sub_rate = sum(r["rate_bpp"] for r in sub)

        slice_counts: Dict[str, int] = {}
        for r in sub:
            slice_counts[r["slice"]] = slice_counts.get(r["slice"], 0) + 1

        dist_str = ", ".join(f"{s}: {c} ({c/k*100:.1f}%)" for s, c in sorted(slice_counts.items()))

        lines.append(f"\n  Top {k} Highest-Rate Channels (Total: {k}):")
        lines.append(f"    Total Decodable Rate: {total_sub_rate:.4f} bpp")
        lines.append(f"    Slice Distribution:   {dist_str}")
        lines.append(f"    Channel Indices:      {channels_list}")

    # 2. Detailed Table for Top 32 channels
    max_table_rows = min(len(ranked_channels), 32)
    lines.append(f"\n[DETAILED RANKING (Top {max_table_rows} Channels)]")
    lines.append(f"  {'Rank':<6} {'Channel':<9} {'Slice':<8} {'Slice Rel Idx':<15} {'Rate (bpp)':<14}")
    lines.append("  " + "-" * 60)

    for r in ranked_channels[:max_table_rows]:
        lines.append(
            f"  #{r['rank']:<5} ch_{r['channel']:<6} {r['slice']:<8} "
            f"+{r['rel_idx']:<14} {r['rate_bpp']:>10.6f} bpp"
        )

    lines.append("=" * 80)
    return "\n".join(lines)



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
        split: Dataset split ('train', 'test', or 'all').
        batch_size: Batch size for memory-efficient streaming.
        max_samples: Optional upper limit on samples for faster approximation.

    Returns:
        np.ndarray of shape (320,) containing non-zero densities in [0.0, 1.0].
    """
    should_close = False
    if isinstance(h5_source, (str, Path)):
        f = h5py.File(str(h5_source), "r", swmr=True)
        should_close = True
    else:
        f = h5_source

    try:
        if "latents_quantized" not in f:
            raise KeyError(f"Archive missing 'latents_quantized'. Keys present: {list(f.keys())}")

        ds = f["latents_quantized"]
        is_train = f["is_train"][:] if "is_train" in f else np.ones(ds.shape[0], dtype=bool)

        if split == "train":
            split_mask = is_train
        elif split == "test":
            split_mask = ~is_train
        else:
            split_mask = np.ones(len(is_train), dtype=bool)

        num_channels = ds.shape[1]  # 320
        spatial_elements = ds.shape[2] * ds.shape[3]  # 16 * 16 = 256

        non_zero_counts = np.zeros(num_channels, dtype=np.int64)
        processed = 0
        N_total = len(is_train)

        # Stream in contiguous chunks for optimal HDF5 I/O throughput
        for i in range(0, N_total, batch_size):
            chunk_slice = slice(i, min(i + batch_size, N_total))
            chunk_mask = split_mask[chunk_slice]
            if not np.any(chunk_mask):
                continue

            chunk_data = ds[chunk_slice]  # (B, 320, 16, 16)
            valid_data = chunk_data[chunk_mask]

            if max_samples and (processed + len(valid_data) > max_samples):
                valid_data = valid_data[: max_samples - processed]

            non_zero_counts += np.count_nonzero(valid_data, axis=(0, 2, 3))
            processed += len(valid_data)

            if max_samples and processed >= max_samples:
                break

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
