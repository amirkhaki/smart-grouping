"""Tier-2 Entrypoint: Parallel joint training of all 36 probe configurations."""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarks.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILENAME,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    NUM_CLASSES,
    SLICE_SPECS,
    SliceSpec,
)
from benchmarks.datasets.hdf5_dataset import HDF5LatentDataset
from benchmarks.evaluate import export_master_results, generate_pareto_plot
from benchmarks.models.multi_head_container import MultiHeadClassifierContainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tier-2 Evaluator: Train all 36 probe heads in parallel on cached ELIC latents."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("TIER1_CACHE_DIR", None),
        help="Directory holding caltech256_elic_latents.h5 (or path to the .h5 file itself)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/kaggle/working",
        help="Output directory for results and plots (default: /kaggle/working)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per GPU (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Initial learning rate for AdamW (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute single epoch on small subset to test end-to-end evaluation flow",
    )
    return parser.parse_args(argv)


def find_h5_cache(cache_hint: Optional[str] = None) -> Path:
    """Locate caltech256_elic_latents.h5 across provided path, env vars, and standard Kaggle mounts."""
    candidates = []
    if cache_hint:
        p = Path(cache_hint)
        if p.is_file():
            return p
        if p.is_dir():
            candidates.append(p / DEFAULT_CACHE_FILENAME)
            candidates.extend(list(p.glob(f"**/{DEFAULT_CACHE_FILENAME}")))

    # Standard search directories
    search_dirs = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("."),
    ]
    for d in search_dirs:
        if d.is_dir():
            candidates.append(d / DEFAULT_CACHE_FILENAME)
            candidates.extend(list(d.glob(f"**/{DEFAULT_CACHE_FILENAME}")))

    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 1024 * 1024:
            return cand

    raise FileNotFoundError(
        f"Could not locate {DEFAULT_CACHE_FILENAME} in {cache_hint} or /kaggle/input. "
        "Ensure Tier-1 cache kernel is attached as a kernel source."
    )


def calculate_slice_bitrates(
    test_bitrates: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    """Compute empirical mean decodable bitrates for all slice views from test set.

    Args:
        test_bitrates: Array of shape (N_test, 6) holding [R(z), R(y0), R(y1), R(y2), R(y3), R(y4)]

    Returns:
        slice_bpp_dict: Mapping from view name to empirical decodable bpp
        bpp_z: Mean hyperprior bitrate
    """
    mean_rates = np.mean(test_bitrates, axis=0)  # (6,)
    bpp_z = float(mean_rates[0])
    bpp_slices = [float(x) for x in mean_rates[1:]]  # [s0, s1, s2, s3, s4]

    slice_bpp: Dict[str, float] = {}

    # Isolated slices
    slice_bpp["S0"] = bpp_slices[0]
    slice_bpp["S1"] = bpp_slices[1]
    slice_bpp["S2"] = bpp_slices[2]
    slice_bpp["S3"] = bpp_slices[3]
    slice_bpp["S4"] = bpp_slices[4]

    # Cumulative progressive slices (including hyperprior R(z))
    slice_bpp["C1"] = bpp_z + sum(bpp_slices[:2])  # S0 + S1
    slice_bpp["C2"] = bpp_z + sum(bpp_slices[:3])  # S0 + S1 + S2
    slice_bpp["C3"] = bpp_z + sum(bpp_slices[:4])  # S0 + S1 + S2 + S3
    slice_bpp["C4"] = bpp_z + sum(bpp_slices[:5])  # S0 + S1 + S2 + S3 + S4

    return slice_bpp, bpp_z


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate HDF5 Cache
    h5_path = find_h5_cache(args.cache_dir)
    print(f"[INIT] Located Tier-1 HDF5 Cache: {h5_path} ({h5_path.stat().st_size / (1024**3):.2f} GB)")

    # 2. Instantiate Datasets and DataLoaders
    train_dataset = HDF5LatentDataset(h5_path, split="train")
    test_dataset = HDF5LatentDataset(h5_path, split="test")

    epochs = args.epochs
    if args.dry_run:
        print("[DRY-RUN] Setting epochs=1 and using minimal sample subsets")
        epochs = 1
        train_dataset.indices = train_dataset.indices[:20]
        test_dataset.indices = test_dataset.indices[:20]

    print(f"[INIT] Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # 3. Instantiate Multi-Head Container
    print(f"[INIT] Instantiating MultiHeadClassifierContainer (36 heads) on {args.device}...")
    container = MultiHeadClassifierContainer(num_classes=NUM_CLASSES).to(args.device)

    # DataParallel wrapper for 2x Tesla T4 GPUs
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INIT] Distributing container across {torch.cuda.device_count()} GPUs using DataParallel")
        model = nn.DataParallel(container)
    else:
        model = container

    # Optimizer and Cosine Annealing Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    t_start = time.time()
    print(f"\n[TRAIN] Beginning {epochs} epochs of multi-head training...")

    raw_container = model.module if isinstance(model, nn.DataParallel) else model

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            y_q = batch["y_quantized"].to(args.device)
            y_c = batch["y_continuous"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            logits_dict = model(y_q, y_c)

            total_loss, _ = raw_container.compute_loss(logits_dict, labels)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(num_batches, 1)
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    train_time = time.time() - t_start
    print(f"[TRAIN] Finished training in {train_time:.1f}s ({train_time / 60:.2f} min)")

    # 5. Evaluation Loop on Test Split
    print("\n[EVAL] Evaluating all 36 heads on test split...")
    model.eval()

    head_correct_top1: Dict[str, int] = {k: 0 for k in raw_container.head_keys}
    head_correct_top5: Dict[str, int] = {k: 0 for k in raw_container.head_keys}
    total_test_samples = 0
    all_bitrates_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            y_q = batch["y_quantized"].to(args.device)
            y_c = batch["y_continuous"].to(args.device)
            labels = batch["label"].to(args.device)
            B = labels.size(0)
            total_test_samples += B

            all_bitrates_list.append(batch["bitrates_bpp"].numpy())
            logits_dict = model(y_q, y_c)

            for head_key, logits in logits_dict.items():
                _, pred = logits.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))

                head_correct_top1[head_key] += correct[:1].reshape(-1).int().sum().item()
                head_correct_top5[head_key] += correct[:5].reshape(-1).int().sum().item()

    test_bitrates = np.concatenate(all_bitrates_list, axis=0) if all_bitrates_list else np.zeros((1, 6))
    slice_bpp_map, bpp_z = calculate_slice_bitrates(test_bitrates)
    lmbda_val = train_dataset.attrs.get("lambda", 0.032)

    # 6. Format Results Table
    records: List[Dict[str, Any]] = []
    for head_key in raw_container.head_keys:
        meta = raw_container.head_metadata[head_key]
        spec: SliceSpec = meta["slice_spec"]
        arch = meta["arch"]
        quant = meta["quant"]
        view = spec.name

        top1 = (head_correct_top1[head_key] / max(total_test_samples, 1)) * 100.0
        top5 = (head_correct_top5[head_key] / max(total_test_samples, 1)) * 100.0
        bpp = slice_bpp_map.get(view, 0.0)

        records.append({
            "head_key": head_key,
            "view": view,
            "category": spec.category,
            "arch": arch,
            "quant": quant,
            "in_channels": spec.num_channels,
            "bpp": round(bpp, 4),
            "top1_acc": round(top1, 2),
            "top5_acc": round(top5, 2),
            "lmbda": lmbda_val,
        })

        # Also add S0 as cumulative C0 for Pareto plotting consistency
        if view == "S0":
            records.append({
                "head_key": f"C0_{arch}_{quant}",
                "view": "C0",
                "category": "cumulative",
                "arch": arch,
                "quant": quant,
                "in_channels": spec.num_channels,
                "bpp": round(bpp_z + bpp, 4),
                "top1_acc": round(top1, 2),
                "top5_acc": round(top5, 2),
                "lmbda": lmbda_val,
            })

    # 7. Serialize Metrics and Plots
    out_csv = out_dir / "master_results.csv"
    out_json = out_dir / "metrics.json"
    out_plot = out_dir / "pareto_curve.png"

    df = export_master_results(records, out_csv, out_json)
    generate_pareto_plot(df, out_plot)

    print("\n[BENCHMARK COMPLETE] Summary of Top-1 Accuracies:")
    print(df[["view", "category", "arch", "quant", "bpp", "top1_acc"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
