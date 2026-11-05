"""Concurrent Multi-Head Trainer and Evaluator for IQA Latent Regression.

Supports:
  - Soft Pairwise Logistic (RankNet-style) loss with label-noise filtering and anchor MSE
  - Independent per-head AdamW optimizers and independent gradient norm clipping
  - Multi-seed outer evaluation loop with Mean +/- Std reporting
  - Prediction vector saving and paired bootstrap hypothesis testing (B=2,000)
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
                print(f"[CACHE] Loading {self.total_samples} '{split}' samples into RAM (fp16) from {dataset_key}...")
                raw_latents = f[dataset_key][:][self.indices]
                # Store as fp16 in RAM to halve memory consumption (~13.8 GB vs 27.7 GB)
                self.latents = torch.from_numpy(raw_latents)
                if self.latents.dtype != torch.float16:
                    self.latents = self.latents.half()
                raw_scores = f["mos_scores"][:][self.indices]
                self.scores = torch.from_numpy(raw_scores).float()
            else:
                self.latents = None
                self.scores = None

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            return self.latents[idx].float(), self.scores[idx]

        # Streaming fallback
        h5_idx = self.indices[idx]
        with h5py.File(self.h5_path, "r", swmr=True) as f:
            dataset_key = "latents_quantized"
            latent = torch.from_numpy(f[dataset_key][h5_idx]).float()
            score = torch.tensor(float(f["mos_scores"][h5_idx]), dtype=torch.float32)
            return latent, score


class SyntheticInMemoryDataset(Dataset):
    """Synthetic dataset for Tier-1 smoke testing."""

    def __init__(
        self,
        num_samples: int = 50,
        channels: int = 320,
        spatial: Tuple[int, int] = (48, 64),
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        H_s, W_s = spatial if isinstance(spatial, (tuple, list)) else (spatial, spatial)
        self.latents = torch.randn(num_samples, channels, H_s, W_s)
        self.scores = torch.from_numpy(rng.uniform(0.1, 0.9, size=(num_samples,)).astype(np.float32))

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.scores[idx]


def evaluate_loader(
    model: MultiHeadIQARegressionContainer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], np.ndarray]:
    """Evaluate multi-head predictions over a dataloader and return metrics and predictions."""
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
    preds_arrays = {}
    targets_array = np.array(all_targets, dtype=np.float32)

    for v in model.views:
        pv = np.array(all_preds[v], dtype=np.float32)
        preds_arrays[v] = pv
        results[v] = compute_iqa_metrics(pv, targets_array)

    return results, preds_arrays, targets_array


def paired_bootstrap_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    targets: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap hypothesis test comparing predictions A vs B against ground truth targets.

    Computes delta_srcc = SRCC(A) - SRCC(B), 95% bootstrap confidence intervals,
    and two-sided empirical p-values.
    """
    from scipy.stats import spearmanr
    rng = np.random.RandomState(seed)
    N = len(targets)
    deltas = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, size=N)
        t_sample = targets[idx]
        a_sample = preds_a[idx]
        b_sample = preds_b[idx]

        if np.ptp(a_sample) < 1e-7 or np.ptp(b_sample) < 1e-7:
            continue
        srcc_a = spearmanr(a_sample, t_sample).correlation
        srcc_b = spearmanr(b_sample, t_sample).correlation
        if np.isnan(srcc_a) or np.isnan(srcc_b):
            continue
        deltas.append(srcc_a - srcc_b)

    if not deltas:
        return {"delta_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "n_bootstraps": 0}

    deltas_arr = np.array(deltas)
    delta_mean = float(np.mean(deltas_arr))
    ci_lower = float(np.percentile(deltas_arr, 2.5))
    ci_upper = float(np.percentile(deltas_arr, 97.5))
    p_val = float(2.0 * min(np.mean(deltas_arr <= 0), np.mean(deltas_arr >= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "delta_mean": round(delta_mean, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "significant_p05": bool(ci_lower > 0 or ci_upper < 0),
        "n_bootstraps": len(deltas_arr),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multi-Head IQA Probe Trainer and Evaluator.")
    parser.add_argument("--h5-cache", type=str, default=None, help="Path to HDF5 cache file")
    parser.add_argument("--views", nargs="+", default=["S0", "C1", "C2", "C3", "C4"], help="Slice views to train")
    parser.add_argument("--symbol-density", nargs="+", type=int, default=None, help="Top-K symbol-density channel subsets")
    parser.add_argument("--symbol-density-samples", type=int, default=2000, help="Training latents for symbol density")
    parser.add_argument("--checkpoint", type=str, default="0.45", help="ELIC checkpoint lambda")
    parser.add_argument("--stem-channels", type=int, default=128, help="Stem channel count (default: 128)")
    parser.add_argument("--num-blocks", type=int, default=2, help="Number of PreAct blocks (default: 2)")
    parser.add_argument("--channel-dropout", type=float, default=0.1, help="Input channel dropout (default: 0.1)")
    parser.add_argument("--quant-mode", type=str, default="quantized", choices=["quantized", "continuous"])
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay (default: 0.02)")
    parser.add_argument("--loss-type", type=str, default="pairwise", choices=["pairwise", "joint"], help="Loss type (default: pairwise)")
    parser.add_argument("--loss-tau", type=float, default=0.03, help="Noise threshold tau for pairwise loss (default: 0.03)")
    parser.add_argument("--loss-anchor", type=float, default=0.1, help="MSE anchor weight for pairwise loss (default: 0.1)")
    parser.add_argument("--loss-alpha", type=float, default=1.0, help="Weight for Pearson correlation loss (if joint)")
    parser.add_argument("--loss-beta", type=float, default=0.5, help="Weight for L1 regression loss (if joint)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience epochs on validation SRCC (default: 7)")
    parser.add_argument("--seed", type=int, default=42, help="Single random seed (default: 42)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of random seeds (e.g. 42 43 44 45 46)")
    parser.add_argument("--multi-seed", action="store_true", help="Run across 5 seeds: 42, 43, 44, 45, 46")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working", help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true", help="Run in mock synthetic mode for Tier-1 smoke testing")
    return parser.parse_args(argv)


def train_single_seed(
    seed: int,
    args: Any,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    custom_subsets: Dict[str, List[int]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], np.ndarray, List[Dict[str, Any]]]:
    """Train and evaluate multi-head probe on a single deterministic seed with independent optimizers."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MultiHeadIQARegressionContainer(
        views=args.views,
        custom_subsets=custom_subsets,
        stem_channels=args.stem_channels,
        num_blocks=args.num_blocks,
        channel_dropout=args.channel_dropout,
        loss_type=args.loss_type,
        loss_tau=args.loss_tau,
        loss_anchor=args.loss_anchor,
        loss_alpha=args.loss_alpha,
        loss_beta=args.loss_beta,
    ).to(device)

    # Independent per-head AdamW optimizers and Cosine Annealing schedulers
    optimizers = {
        v: torch.optim.AdamW(model.heads[v].parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for v in model.views
    }
    schedulers = {
        v: torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[v], T_max=args.epochs, eta_min=1e-5)
        for v in model.views
    }

    best_val_srcc = {v: -1.0 for v in model.views}
    best_head_weights = {v: copy.deepcopy(model.heads[v].state_dict()) for v in model.views}
    patience_counters = {v: 0 for v in model.views}
    seed_history = []

    print(f"\n[Seed {seed}] Starting training for {args.epochs} epochs with independent per-head optimizers...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_head_loss_acc = {v: 0.0 for v in model.views}
        n_batches = 0

        for latents, targets in train_loader:
            latents = latents.to(device)
            targets = targets.to(device)

            preds = model(latents)
            head_losses = model.compute_head_losses(preds, targets)

            # Decoupled backward pass and per-head gradient norm clipping (1.0)
            for idx, v in enumerate(model.views):
                opt = optimizers[v]
                opt.zero_grad()
                h_loss = head_losses[v]
                # Retain computation graph across multiple heads
                h_loss.backward(retain_graph=(idx < len(model.views) - 1))
                torch.nn.utils.clip_grad_norm_(model.heads[v].parameters(), max_norm=1.0)
                opt.step()

                train_head_loss_acc[v] += float(h_loss.item())

            total_loss = sum(head_losses.values())
            train_loss_acc += float(total_loss.item())
            n_batches += 1

        for v in model.views:
            schedulers[v].step()

        # Validation Step
        val_metrics, _, _ = evaluate_loader(model, val_loader, device)

        epoch_record = {"seed": seed, "epoch": epoch, "train_loss": round(train_loss_acc / max(1, n_batches), 4)}
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

        seed_history.append(epoch_record)
        status_line = " | ".join([f"{v}: ValSRCC={val_metrics[v]['srcc']:.3f}" for v in model.views[:4]])
        print(f"[Seed {seed}] Epoch {epoch:02d}/{args.epochs:02d} | TrainLoss: {epoch_record['train_loss']:.4f} | {status_line}")

        if all(patience_counters[v] >= args.patience for v in model.views):
            print(f"[Seed {seed}] All heads reached patience ({args.patience}) at epoch {epoch}. Stopping.")
            break

    # Restore best validation checkpoints for test evaluation
    for v in model.views:
        model.heads[v].load_state_dict(best_head_weights[v])

    test_metrics, test_preds, test_targets = evaluate_loader(model, test_loader, device)

    seed_records = []
    for v in model.views:
        meta = model.head_metadata[v]
        seed_records.append({
            "head_key": f"{v}_iqa_latent_head_w{args.stem_channels}",
            "view": v,
            "seed": seed,
            "in_channels": meta["in_channels"],
            "stem_channels": meta["stem_channels"],
            "num_params": meta["num_params"],
            "quant_mode": args.quant_mode,
            "loss_type": args.loss_type,
            "val_srcc": best_val_srcc[v],
            "test_srcc": test_metrics[v]["srcc"],
            "test_plcc": test_metrics[v]["plcc"],
            "test_rmse": test_metrics[v]["rmse"],
        })

    return seed_records, test_preds, test_targets, seed_history


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Determine seed list
    if args.multi_seed:
        seeds = [42, 43, 44, 45, 46]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.seed]

    print(f"[INIT] Evaluating seeds: {seeds}")

    # 1. Prepare Datasets
    if args.synthetic:
        print("[INIT] Running with SyntheticInMemoryDataset for Tier-1 smoke testing")
        train_ds = SyntheticInMemoryDataset(num_samples=40, spatial=(48, 64), seed=seeds[0])
        val_ds = SyntheticInMemoryDataset(num_samples=15, spatial=(48, 64), seed=seeds[0] + 1)
        test_ds = SyntheticInMemoryDataset(num_samples=20, spatial=(48, 64), seed=seeds[0] + 2)
    else:
        if args.h5_cache and Path(args.h5_cache).is_file():
            h5_path = Path(args.h5_cache)
        else:
            # Search /kaggle/input and current directory for full-frame or standard HDF5
            found = list(Path("/kaggle/input").rglob("*fullframe*.h5"))
            if not found:
                found = list(Path("/kaggle/input").rglob("*.h5"))
            if not found and Path(".").is_dir():
                found = list(Path(".").rglob("*fullframe*.h5")) or list(Path(".").rglob("*.h5"))
            if found:
                h5_path = found[0]
                print(f"[INIT] Discovered HDF5 cache: {h5_path}")
            else:
                raise FileNotFoundError(f"HDF5 cache file not found (specified: {args.h5_cache})")

        train_ds = CachedIQADataset(h5_path, split="train", quant_mode=args.quant_mode, in_memory=True)
        val_ds = CachedIQADataset(h5_path, split="val", quant_mode=args.quant_mode, in_memory=True)
        test_ds = CachedIQADataset(h5_path, split="test", quant_mode=args.quant_mode, in_memory=True)

    # 2. Handle Custom Subsets (Symbol Density) if requested
    custom_subsets = {}
    if args.symbol_density:
        if args.synthetic:
            ranked_by_symbol_density = [
                {"channel": c, "symbol_density": float(1.0 / (c + 1)), "rank": c + 1} for c in range(320)
            ]
        else:
            from benchmarks.cache_latents import resolve_checkpoint
            from benchmarks.elic_extractor import ELICFeatureExtractor
            from benchmarks.sparsity import (
                compute_channel_symbol_density_from_latents,
                rank_channels_by_symbol_density,
                format_symbol_density_summary,
            )

            ckpt_dir = Path("/tmp/checkpoints") if Path("/tmp").is_dir() else (out_dir / "checkpoints")
            ckpt_path, _ = resolve_checkpoint(args.checkpoint, ckpt_dir)
            extractor = ELICFeatureExtractor.from_checkpoint(ckpt_path, device=str(device))
            densities = compute_channel_symbol_density_from_latents(
                extractor=extractor,
                h5_source=h5_path,
                split="train",
                batch_size=args.batch_size,
                max_samples=args.symbol_density_samples,
                device=str(device),
            )
            ranked_by_symbol_density = rank_channels_by_symbol_density(densities)

        for k in args.symbol_density:
            indices = [r["channel"] for r in ranked_by_symbol_density[:k]]
            custom_subsets[f"SD{k}"] = indices

    # 3. Multi-Seed Training Loop
    all_master_records = []
    all_seed_predictions: Dict[int, Dict[str, np.ndarray]] = {}
    all_history = []
    ground_truth_test_targets = None

    for seed in seeds:
        seed_records, seed_test_preds, test_targets, seed_hist = train_single_seed(
            seed=seed,
            args=args,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            custom_subsets=custom_subsets,
            device=device,
        )
        all_master_records.extend(seed_records)
        all_seed_predictions[seed] = seed_test_preds
        all_history.extend(seed_hist)
        if ground_truth_test_targets is None:
            ground_truth_test_targets = test_targets

    # 4. Compute Aggregate Statistics Across Seeds (Mean +/- Std)
    views_present = list(dict.fromkeys([r["view"] for r in all_master_records]))
    summary_records = []

    print("\n" + "=" * 80)
    print(f"{'Slice View':<12} | {'Channels':<8} | {'Seeds':<5} | {'Val SRCC (Mean±Std)':<20} | {'Test SRCC (Mean±Std)':<20} | {'Test PLCC (Mean±Std)':<20}")
    print("-" * 80)

    for v in views_present:
        v_records = [r for r in all_master_records if r["view"] == v]
        n_seeds = len(v_records)
        val_srccs = [r["val_srcc"] for r in v_records]
        test_srccs = [r["test_srcc"] for r in v_records]
        test_plccs = [r["test_plcc"] for r in v_records]
        test_rmses = [r["test_rmse"] for r in v_records]
        ch = v_records[0]["in_channels"]

        mean_val_s, std_val_s = float(np.mean(val_srccs)), float(np.std(val_srccs))
        mean_test_s, std_test_s = float(np.mean(test_srccs)), float(np.std(test_srccs))
        mean_test_p, std_test_p = float(np.mean(test_plccs)), float(np.std(test_plccs))
        mean_rmse, std_rmse = float(np.mean(test_rmses)), float(np.std(test_rmses))

        print(
            f"{v:<12} | {ch:<8} | {n_seeds:<5} | "
            f"{mean_val_s:.4f} ± {std_val_s:.4f}       | "
            f"{mean_test_s:.4f} ± {std_test_s:.4f}       | "
            f"{mean_test_p:.4f} ± {std_test_p:.4f}"
        )

        summary_records.append({
            "view": v,
            "in_channels": ch,
            "n_seeds": n_seeds,
            "mean_val_srcc": round(mean_val_s, 4),
            "std_val_srcc": round(std_val_s, 4),
            "mean_test_srcc": round(mean_test_s, 4),
            "std_test_srcc": round(std_test_s, 4),
            "mean_test_plcc": round(mean_test_p, 4),
            "std_test_plcc": round(std_test_p, 4),
            "mean_test_rmse": round(mean_rmse, 4),
            "std_test_rmse": round(std_rmse, 4),
        })
    print("=" * 80 + "\n")

    # 5. Paired Bootstrap Hypothesis Testing on Test Split
    bootstrap_results = {}
    if len(views_present) >= 2 and ground_truth_test_targets is not None:
        print("[BOOTSTRAP] Performing paired bootstrap hypothesis testing (B=2,000)...")
        # Build ensemble predictions across seeds for each view
        ensemble_preds = {}
        for v in views_present:
            preds_list = [all_seed_predictions[s][v] for s in seeds if v in all_seed_predictions[s]]
            ensemble_preds[v] = np.mean(preds_list, axis=0)

        # Standard comparisons: C3 vs S0, C3 vs C4, C3 vs C2, C4 vs S0
        pairs_to_test = [("C3", "S0"), ("C3", "C4"), ("C3", "C2"), ("C4", "S0")]
        for va, vb in pairs_to_test:
            if va in ensemble_preds and vb in ensemble_preds:
                pair_key = f"{va}_vs_{vb}"
                boot = paired_bootstrap_test(
                    preds_a=ensemble_preds[va],
                    preds_b=ensemble_preds[vb],
                    targets=ground_truth_test_targets,
                    n_bootstraps=2000,
                    seed=42,
                )
                bootstrap_results[pair_key] = boot
                print(
                    f"  {pair_key:<12}: Delta_SRCC={boot['delta_mean']:+.4f} "
                    f"(95% CI: [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}], p={boot['p_value']:.4f}, sig={boot['significant_p05']})"
                )

    # 6. Export Artifacts
    # All raw seed records
    csv_path = out_dir / "iqa_master_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_master_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_master_records)

    # Summary Mean +/- Std table (master_results.csv for Job Summary)
    summary_csv_path = out_dir / "master_results.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)

    # JSON outputs
    (out_dir / "iqa_metrics.json").write_text(json.dumps(all_master_records, indent=2))
    (out_dir / "iqa_summary.json").write_text(json.dumps(summary_records, indent=2))
    (out_dir / "paired_bootstrap_results.json").write_text(json.dumps(bootstrap_results, indent=2))

    # Save raw test prediction arrays for replication
    test_pred_save_dict = {"targets": ground_truth_test_targets}
    for s in seeds:
        for v in views_present:
            if s in all_seed_predictions and v in all_seed_predictions[s]:
                test_pred_save_dict[f"pred_{v}_seed{s}"] = all_seed_predictions[s][v]
    np.savez_compressed(out_dir / "test_predictions.npz", **test_pred_save_dict)

    # History
    with open(out_dir / "iqa_training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_history[0].keys()))
        writer.writeheader()
        writer.writerows(all_history)

    print(f"\n[COMPLETE] All artifacts exported to {out_dir}:")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_csv_path.name}")
    print(f"  - paired_bootstrap_results.json")
    print(f"  - test_predictions.npz")


if __name__ == "__main__":
    main()
