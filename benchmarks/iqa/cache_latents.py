"""Tier-1 Entrypoint: Extracts continuous & quantized ELIC latents and bitrates for IQA to HDF5."""

import argparse
import datetime
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from benchmarks.config import (
    CHECKPOINT_REGISTRY,
    DEFAULT_BATCH_SIZE,
    HDF5_CHUNK_SHAPE,
    LATENT_SPATIAL_DIM,
    LATENT_TOTAL_CHANNELS,
)
from benchmarks.download_checkpoints import download_checkpoint
from benchmarks.elic_extractor import ELICFeatureExtractor
from benchmarks.iqa.dataset import (
    IQADatasetWrapper,
    create_deterministic_split,
    get_iqa_transform,
)


DEFAULT_IQA_CACHE_FILENAME = "koniq10k_elic0450_latents.h5"
DEFAULT_IQA_FULLFRAME_CACHE_FILENAME = "koniq10k_elic0450_fullframe_latents.h5"


class SyntheticIQADataset(Dataset):
    """Mock synthetic dataset for Tier-1 smoke testing without network/dataset download."""

    def __init__(self, num_samples: int = 20, image_size: Tuple[int, int] = (256, 256), seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        rng = np.random.RandomState(seed)
        self.scores = rng.uniform(0.1, 0.9, size=(num_samples,)).astype(np.float32)
        torch.manual_seed(seed)
        self.images = torch.rand(num_samples, 3, image_size[0], image_size[1])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "image": self.images[idx],
            "raw_score": self.scores[idx] * 5.0,
            "score": self.scores[idx],
            "index": idx,
        }


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access inner module methods and attributes."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def resolve_checkpoint(checkpoint_arg: str, download_dir: Path) -> Tuple[Path, float]:
    """Resolve a checkpoint path or lambda key from registry."""
    # 1. Direct path check
    direct_path = Path(checkpoint_arg)
    if direct_path.is_file():
        lmbda = 0.45
        for k, meta in CHECKPOINT_REGISTRY.items():
            if k in direct_path.name or meta.default_filename in str(direct_path):
                lmbda = meta.lmbda
                break
        return direct_path, lmbda

    # 2. Key lookup
    if checkpoint_arg in CHECKPOINT_REGISTRY:
        meta = CHECKPOINT_REGISTRY[checkpoint_arg]
        ckpt_path = download_checkpoint(meta, download_dir)
        return ckpt_path, meta.lmbda

    # 3. Normalized string lookup
    clean_key = str(float(checkpoint_arg)) if checkpoint_arg.replace(".", "", 1).isdigit() else checkpoint_arg
    for k, meta in CHECKPOINT_REGISTRY.items():
        if k == clean_key or str(meta.lmbda) == clean_key:
            ckpt_path = download_checkpoint(meta, download_dir)
            return ckpt_path, meta.lmbda

    raise ValueError(
        f"Checkpoint argument '{checkpoint_arg}' is neither an existing file nor a valid lambda key in {list(CHECKPOINT_REGISTRY.keys())}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tier-1 IQA Cacher: Extract deterministic ELIC latents and MOS scores into HDF5."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="koniq10k",
        help="IQA dataset name supported by iqadataset (default: koniq10k)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/tmp/data" if Path("/tmp").is_dir() else "data",
        help="Directory to download or load IQA dataset (default: /tmp/data to keep /kaggle/working clean)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="0.45",
        help="ELIC checkpoint lambda (default: 0.45 for maximum fidelity)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Center crop size in pixels (default: 256; ignored if --no-crop is passed)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Extract from native full-frame images without spatial cropping (e.g. 1024x768 -> 64x48)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/kaggle/working",
        help="Output directory for HDF5 cache (default: /kaggle/working)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="HDF5 archive filename (defaults to koniq10k_elic0450_fullframe_latents.h5 if --no-crop, else koniq10k_elic0450_latents.h5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for feature extraction (default: 16)",
    )
    parser.add_argument(
        "--save-continuous",
        action="store_true",
        help="Also store continuous latents (warning: high-entropy floats do not compress under LZF and add ~18 GB)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run extraction on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 70/10/20 train/val/test splitting (default: 42)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on sample count for fast debugging or Tier-2 signals",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for smoke tests without downloading real images",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.filename is None:
        filename = (
            DEFAULT_IQA_FULLFRAME_CACHE_FILENAME
            if args.no_crop
            else DEFAULT_IQA_CACHE_FILENAME
        )
    else:
        filename = args.filename

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # 1. Resolve and load dataset
    if args.synthetic:
        img_size = (768, 1024) if args.no_crop else (args.crop_size, args.crop_size)
        print(f"[INIT] Using SyntheticIQADataset (max_samples={args.max_samples or 20}, image_size={img_size})")
        dataset = SyntheticIQADataset(num_samples=args.max_samples or 20, image_size=img_size, seed=args.seed)
    else:
        print(f"[INIT] Loading IQA dataset '{args.dataset}' via iqadataset (root: {args.data_root}, no_crop={args.no_crop})...")
        dataset = IQADatasetWrapper(
            dataset_name=args.dataset,
            dataset_root=args.data_root,
            crop_size=args.crop_size,
            no_crop=args.no_crop,
            download=True,
        )

    N_total = len(dataset)
    if args.max_samples and args.max_samples < N_total:
        print(f"[INIT] Capping total dataset samples from {N_total} to {args.max_samples}")
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(args.max_samples)))
        N_total = args.max_samples

    print(f"[INIT] Dataset prepared with {N_total} total samples")

    # Determine dynamic spatial dimensions from the first sample
    sample_img = dataset[0]["image"]
    H_in, W_in = sample_img.shape[-2], sample_img.shape[-1]
    H_y, W_y = H_in // 16, W_in // 16
    print(f"[INIT] Input image resolution: ({H_in}, {W_in}) -> Latent spatial dim: ({H_y}, {W_y})")

    # 2. Compute deterministic 70/10/20 train/val/test split
    split_info = create_deterministic_split(
        total_samples=N_total,
        train_ratio=0.70,
        val_ratio=0.10,
        test_ratio=0.20,
        seed=args.seed,
    )
    print(
        f"[INIT] Split: {split_info['is_train'].sum()} train (70%), "
        f"{split_info['is_val'].sum()} val (10%), {split_info['is_test'].sum()} test (20%)"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # 3. Resolve & download checkpoint
    ckpt_dir = Path("/tmp/checkpoints") if Path("/tmp").is_dir() else (out_dir / "checkpoints")
    ckpt_path, lmbda_val = resolve_checkpoint(args.checkpoint, ckpt_dir)
    print(f"[INIT] Loaded ELIC checkpoint: {ckpt_path} (lambda={lmbda_val})")

    # 4. Initialize Feature Extractor
    print(f"[INIT] Initializing ELICFeatureExtractor on {args.device}...")
    extractor = ELICFeatureExtractor.from_checkpoint(ckpt_path, device=args.device)
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INIT] Multi-GPU extraction across {torch.cuda.device_count()} GPUs")
        extractor = CustomDataParallel(extractor)

    # 5. Initialize HDF5 Archive with LZF chunked compression
    chunk_b = min(16, N_total)
    chunk_shape = (chunk_b, LATENT_TOTAL_CHANNELS, H_y, W_y)
    print(f"[INIT] Creating HDF5 cache: {out_path} ({N_total} samples, chunks={chunk_shape}, compression=lzf)")
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset(
            "latents_quantized",
            shape=(N_total, LATENT_TOTAL_CHANNELS, H_y, W_y),
            dtype=np.float16,
            chunks=chunk_shape,
            compression="lzf",
        )
        if args.save_continuous:
            f.create_dataset(
                "latents_continuous",
                shape=(N_total, LATENT_TOTAL_CHANNELS, H_y, W_y),
                dtype=np.float16,
                chunks=chunk_shape,
                compression="lzf",
            )
        f.create_dataset("mos_scores", shape=(N_total,), dtype=np.float32)
        f.create_dataset("raw_scores", shape=(N_total,), dtype=np.float32)
        f.create_dataset("is_train", shape=(N_total,), dtype=bool)
        f.create_dataset("is_val", shape=(N_total,), dtype=bool)
        f.create_dataset("is_test", shape=(N_total,), dtype=bool)
        f.create_dataset("bitrates_bpp", shape=(N_total, 6), dtype=np.float32)

        # Write split masks
        f["is_train"][:] = split_info["is_train"]
        f["is_val"][:] = split_info["is_val"]
        f["is_test"][:] = split_info["is_test"]

        # Metadata attributes
        f.attrs["checkpoint"] = str(ckpt_path)
        f.attrs["lambda"] = lmbda_val
        f.attrs["dataset"] = args.dataset
        f.attrs["crop_size"] = -1 if args.no_crop else args.crop_size
        f.attrs["no_crop"] = bool(args.no_crop)
        f.attrs["spatial_dim"] = (H_y, W_y)
        f.attrs["compression"] = "lzf"
        f.attrs["total_samples"] = N_total
        f.attrs["seed"] = args.seed
        f.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 6. Extraction Loop
        cursor = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting IQA latents"):
                imgs = batch["image"].to(args.device)
                scores = batch["score"].cpu().numpy()
                raw_s = batch["raw_score"].cpu().numpy()
                B = imgs.size(0)

                # Extract latents and per-slice bitrates
                outputs = extractor(imgs)
                y_hat_tensor = outputs["y_hat"] if "y_hat" in outputs else outputs["latents_quantized"]
                y_hat = y_hat_tensor.cpu().numpy().astype(np.float16)
                bpp = outputs["bitrates_bpp"].cpu().numpy().astype(np.float32)

                f["latents_quantized"][cursor:cursor + B] = y_hat
                if args.save_continuous:
                    y_cont_tensor = outputs["y"] if "y" in outputs else outputs["latents_continuous"]
                    y_cont = y_cont_tensor.cpu().numpy().astype(np.float16)
                    f["latents_continuous"][cursor:cursor + B] = y_cont
                f["mos_scores"][cursor:cursor + B] = scores
                f["raw_scores"][cursor:cursor + B] = raw_s
                f["bitrates_bpp"][cursor:cursor + B] = bpp

                cursor += B
                f.flush()

        elapsed = time.time() - t0
        print(f"[COMPLETE] Extracted {cursor} samples in {elapsed:.1f}s ({elapsed / max(1, cursor):.3f}s/img)")
        print(f"[COMPLETE] Final cache size: {out_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
