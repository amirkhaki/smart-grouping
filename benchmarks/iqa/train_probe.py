"""Concurrent Multi-Head Trainer and Evaluator for IQA Latent Regression."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from benchmarks.iqa.metrics import compute_iqa_metrics
from benchmarks.iqa.models import MultiHeadIQARegressionContainer


class CachedIQADataset(Dataset):
    """In-memory or streamed PyTorch dataset over cached HDF5 latents and MOS scores."""

    def __init__(
        self,
        h5_path: Path,
        split: str = "train",
        quant_mode: str = "quantized",
        in_memory: bool = True,
    ):
        super().__init__()
        self.h5_path = str(h5_path)
        self.in_memory = in_memory
        dataset_key = "latents_quantized" if quant_mode == "quantized" else "latents_continuous"

        with h5py.File(self.h5_path, "r") as f:
            mask_key = f"is_{split}"
            if mask_key not in f:
                raise KeyError(f"HDF5 archive missing split mask '{mask_key}'. Keys: {list(f.keys())}")
            split_mask = f[mask_key][:]
            self.indices = np.where(split_mask)[0]
            self.total_samples = len(self.indices)

            if self.in_memory:
                print(f"[CACHE] Loading {self.total_samples} '{split}' samples into RAM ({dataset_key})...")
                # Read only split samples
                raw_latents = f[dataset_key][:][self.indices]
                self.latents = torch.from_numpy(raw_latents).float()
                raw_scores = f["mos_scores"][:][self.indices]
                self.scores = torch.from_numpy(raw_scores).float()
            else:
                self.latents = None
                self.scores = None

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            return self.latents[idx], self.scores[idx]

        # File-based fallback
        h5_idx = self.indices[idx]
        with h5py.File(self.h5_path, "r", swmr=True) as f:
            dataset_key = "latents_quantized"
            latent = torch.from_numpy(f[dataset_key][h5_idx]).float()
            score = torch.tensor(float(f["mos_scores"][h5_idx]), dtype=torch.float32)
            return latent, score


class SyntheticInMemoryDataset(Dataset):
    """Synthetic dataset for Tier-1 smoke testing."""

    def __init__(self, num_samples: int = 50, channels: int = 320, spatial: int = 16, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.latents = torch.randn(num_samples, channels, spatial, spatial)
        self.scores = torch.from_numpy(rng.uniform(0.1, 0.9, size=(num_samples,)).astype(np.float32))

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.scores[idx]


def evaluate_loader(
    model: MultiHeadIQARegressionContainer,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate multi-head predictions over a dataloader and compute per-head metrics."""
    model.eval()
    all_preds: Dict[str, List[float]] = {v: [] for v in model.views}
    all_targets: List[float] = []

    with torch.no_grad():
        for latents, targets in loader:
            latents = latents.to(device)
            preds = model(latents)
            targets_list = targets.view(-1).tolist()
            all_targets.extend(targets_list)

            for v in model.views:
                p_list = preds[v].detach().view(-1).tolist()
                all_preds[v].extend(p_list)

    results = {}
    for v in model.views:
        results[v] = compute_iqa_metrics(all_preds[v], all_targets)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multi-Head IQA Probe Trainer and Evaluator.")
    parser.add_argument("--h5-cache", type=str, default=None, help="Path to HDF5 cache file")
    parser.add_argument("--views", nargs="+", default=["S0", "S1", "S2", "S3", "S4", "C4"], help="Slice views to train")
    parser.add_argument("--stem-channels", type=int, default=256, help="Stem channel count (default: 256)")
    parser.add_argument("--quant-mode", type=str, default="quantized", choices=["quantized", "continuous"])
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--loss-alpha", type=float, default=1.0, help="Weight for Pearson correlation loss")
    parser.add_argument("--loss-beta", type=float, default=0.5, help="Weight for L1 regression loss")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs on validation SRCC")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working", help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true", help="Run in mock synthetic mode for Tier-1 smoke testing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Prepare Datasets & Loaders
    if args.synthetic:
        print("[INIT] Running with SyntheticInMemoryDataset for Tier-1 smoke testing")
        train_ds = SyntheticInMemoryDataset(num_samples=40, seed=args.seed)
        val_ds = SyntheticInMemoryDataset(num_samples=15, seed=args.seed + 1)
        test_ds = SyntheticInMemoryDataset(num_samples=20, seed=args.seed + 2)
    else:
        h5_path = Path(args.h5_cache)
        if not h5_path.is_file():
            # Search /kaggle/input for any .h5 file if omitted
            found = list(Path("/kaggle/input").rglob("*.h5"))
            if found:
                h5_path = found[0]
                print(f"[INIT] Discovered HDF5 cache: {h5_path}")
            else:
                raise FileNotFoundError(f"HDF5 cache file not found at {args.h5_cache}")

        train_ds = CachedIQADataset(h5_path, split="train", quant_mode=args.quant_mode, in_memory=True)
        val_ds = CachedIQADataset(h5_path, split="val", quant_mode=args.quant_mode, in_memory=True)
        test_ds = CachedIQADataset(h5_path, split="test", quant_mode=args.quant_mode, in_memory=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Build Multi-Head Model
    print(f"[INIT] Initializing MultiHeadIQARegressionContainer with views: {args.views}")
    model = MultiHeadIQARegressionContainer(
        views=args.views,
        stem_channels=args.stem_channels,
        loss_alpha=args.loss_alpha,
        loss_beta=args.loss_beta,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 3. Training Loop with Per-Head Early Stopping on Validation SRCC
    best_val_srcc = {v: -1.0 for v in model.views}
    best_head_weights = {v: copy.deepcopy(model.heads[v].state_dict()) for v in model.views}
    patience_counters = {v: 0 for v in model.views}
    history = []

    print(f"[TRAIN] Starting multi-head training for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_head_loss_acc = {v: 0.0 for v in model.views}
        n_batches = 0

        for latents, targets in train_loader:
            latents = latents.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(latents)
            loss, per_head_losses = model.compute_losses(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += loss.item()
            for v in model.views:
                train_head_loss_acc[v] += per_head_losses[v]
            n_batches += 1

        scheduler.step()

        # Validation Step
        val_metrics = evaluate_loader(model, val_loader, device)

        # Track per-head validation progress
        epoch_record = {"epoch": epoch, "train_loss": round(train_loss_acc / max(1, n_batches), 4)}
        for v in model.views:
            v_srcc = val_metrics[v]["srcc"]
            v_plcc = val_metrics[v]["plcc"]
            epoch_record[f"{v}_val_srcc"] = v_srcc
            epoch_record[f"{v}_val_plcc"] = v_plcc

            if v_srcc > best_val_srcc[v]:
                best_val_srcc[v] = v_srcc
                best_head_weights[v] = copy.deepcopy(model.heads[v].state_dict())
                patience_counters[v] = 0
            else:
                patience_counters[v] += 1

        history.append(epoch_record)
        status_line = " | ".join([f"{v}: SRCC={val_metrics[v]['srcc']:.3f}" for v in model.views[:4]])
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | TrainLoss: {epoch_record['train_loss']:.4f} | {status_line}")

        # Early stopping check
        if all(patience_counters[v] >= args.patience for v in model.views):
            print(f"[EARLY-STOP] All heads exceeded patience ({args.patience}) at epoch {epoch}. Stopping.")
            break

    # 4. Final Evaluation on Held-out 20% Test Split using Best Validation Checkpoints
    print("[EVAL] Restoring best validation checkpoints for final test evaluation...")
    for v in model.views:
        model.heads[v].load_state_dict(best_head_weights[v])

    test_metrics = evaluate_loader(model, test_loader, device)
    print("\n" + "=" * 60)
    print(f"{'Slice View':<12} | {'Channels':<8} | {'Val SRCC':<10} | {'Test SRCC':<10} | {'Test PLCC':<10} | {'RMSE':<8}")
    print("-" * 60)

    master_records = []
    for v in model.views:
        meta = model.head_metadata[v]
        val_s = best_val_srcc[v]
        test_s = test_metrics[v]["srcc"]
        test_p = test_metrics[v]["plcc"]
        rmse = test_metrics[v]["rmse"]

        print(f"{v:<12} | {meta['in_channels']:<8} | {val_s:<10.4f} | {test_s:<10.4f} | {test_p:<10.4f} | {rmse:<8.4f}")

        master_records.append({
            "head_key": f"{v}_iqa_latent_head_w{args.stem_channels}",
            "view": v,
            "in_channels": meta["in_channels"],
            "stem_channels": meta["stem_channels"],
            "num_params": meta["num_params"],
            "quant_mode": args.quant_mode,
            "seed": args.seed,
            "val_srcc": val_s,
            "test_srcc": test_s,
            "test_plcc": test_p,
            "test_rmse": rmse,
        })
    print("=" * 60 + "\n")

    # 5. Export Output Artifacts
    csv_path = out_dir / "iqa_master_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_records[0].keys()))
        writer.writeheader()
        writer.writerows(master_records)

    json_path = out_dir / "iqa_metrics.json"
    json_path.write_text(json.dumps(master_records, indent=2))

    hist_path = out_dir / "iqa_training_history.csv"
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"[COMPLETE] Results exported to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
