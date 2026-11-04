"""Tier-1 Entrypoint: Extracts continuous & quantized ELIC latents and bitrates to HDF5."""

import argparse
import datetime
import os
from pathlib import Path
import sys
import time
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarks.config import (
    CHECKPOINT_REGISTRY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_FILENAME,
    HDF5_CHUNK_SHAPE,
    LATENT_SPATIAL_DIM,
    LATENT_TOTAL_CHANNELS,
)
from benchmarks.datasets.caltech256 import (
    Caltech256PreparedDataset,
    build_caltech256_sample_manifest,
    find_caltech256_root,
)
from benchmarks.download_checkpoints import download_checkpoint
from benchmarks.elic_extractor import ELICFeatureExtractor


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access inner module methods and attributes."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tier-1 Cacher: Extract deterministic ELIC latents into an indexed HDF5 archive."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to Caltech-256 dataset root. If omitted, searches /kaggle/input",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="0.032",
        help="Path to checkpoint file OR rate lambda key (e.g. '0.032', '0.004')",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/kaggle/working",
        help="Directory to store HDF5 cache (default: /kaggle/working)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_CACHE_FILENAME,
        help=f"HDF5 archive filename (default: {DEFAULT_CACHE_FILENAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=150,
        help="Balanced target sample count per class in train split (default: 150)",
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
        help="Process only 10 samples to test pipeline and archive schema",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(checkpoint_arg: str, download_dir: Path) -> Tuple[Path, float]:
    """Resolve checkpoint file path and lambda value from CLI argument, downloading if needed."""
    # 1. Check if argument is a registered lambda key (e.g. '0.032')
    if checkpoint_arg in CHECKPOINT_REGISTRY:
        meta = CHECKPOINT_REGISTRY[checkpoint_arg]
        ckpt_path = download_checkpoint(meta, download_dir)
        return ckpt_path, meta.lmbda

    p = Path(checkpoint_arg)
    # 2. Check if file exists directly on disk
    if p.is_file():
        lmbda = 0.032
        for k, meta in CHECKPOINT_REGISTRY.items():
            if k in p.name or Path(meta.default_filename).name == p.name:
                lmbda = meta.lmbda
                break
        return p, lmbda

    # 3. Check candidate local directories if the target already exists
    filename = p.name
    search_locations = [
        download_dir / filename,
        Path("checkpoints") / filename,
        Path("/kaggle/working/checkpoints") / filename,
        Path("/kaggle/input") / filename,
    ]
    for loc in search_locations:
        if loc.is_file():
            lmbda = 0.032
            for k, meta in CHECKPOINT_REGISTRY.items():
                if k in loc.name or Path(meta.default_filename).name == loc.name:
                    lmbda = meta.lmbda
                    break
            return loc, lmbda

    # 4. If not on disk, match filename or string against registered checkpoints and download
    matched_meta = None
    for k, meta in CHECKPOINT_REGISTRY.items():
        reg_filename = Path(meta.default_filename).name
        if filename == reg_filename or k == checkpoint_arg:
            matched_meta = meta
            break

    if matched_meta is None:
        for k, meta in CHECKPOINT_REGISTRY.items():
            k_nodot = k.replace(".", "")
            if k in filename or k_nodot in filename or k in checkpoint_arg:
                matched_meta = meta
                break

    if matched_meta is not None:
        ckpt_path = download_checkpoint(matched_meta, download_dir)
        return ckpt_path, matched_meta.lmbda

    # 5. Failed to resolve
    raise ValueError(
        f"Checkpoint argument '{checkpoint_arg}' is neither an existing file nor a valid lambda key in {list(CHECKPOINT_REGISTRY.keys())}"
    )


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename

    # 1. Locate Caltech-256 data root
    if args.data_root:
        data_root = Path(args.data_root)
        if not data_root.exists():
            data_root = find_caltech256_root(args.data_root)
    else:
        data_root = find_caltech256_root()
    print(f"[INIT] Using Caltech-256 data root: {data_root}")

    # 2. Resolve & download checkpoint
    ckpt_dir = out_dir / "checkpoints" if not Path("checkpoints").exists() else Path("checkpoints")
    ckpt_path, lmbda_val = resolve_checkpoint(args.checkpoint, ckpt_dir)
    print(f"[INIT] Loaded checkpoint: {ckpt_path} (lambda={lmbda_val})")

    # 3. Build sample manifest and dataset
    samples, class_to_name = build_caltech256_sample_manifest(data_root, target_k=args.target_k)
    if args.dry_run:
        print("[DRY-RUN] Restricting extraction to 10 samples")
        samples = samples[:10]

    N = len(samples)
    dataset = Caltech256PreparedDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # 4. Initialize Feature Extractor
    print(f"[INIT] Initializing ELICFeatureExtractor on {args.device}...")
    extractor = ELICFeatureExtractor.from_checkpoint(ckpt_path, device=args.device)

    # Multi-GPU support with CustomDataParallel
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INIT] Accelerating extraction across {torch.cuda.device_count()} GPUs using CustomDataParallel")
        extractor = CustomDataParallel(extractor)

    # 5. Initialize HDF5 Archive
    print(f"[INIT] Creating HDF5 cache: {out_path} ({N} samples)")
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset(
            "latents_quantized",
            shape=(N, LATENT_TOTAL_CHANNELS, LATENT_SPATIAL_DIM[0], LATENT_SPATIAL_DIM[1]),
            dtype=np.float16,
            chunks=HDF5_CHUNK_SHAPE,
        )
        f.create_dataset(
            "latents_continuous",
            shape=(N, LATENT_TOTAL_CHANNELS, LATENT_SPATIAL_DIM[0], LATENT_SPATIAL_DIM[1]),
            dtype=np.float16,
            chunks=HDF5_CHUNK_SHAPE,
        )
        f.create_dataset("labels", shape=(N,), dtype=np.int64)
        f.create_dataset("is_train", shape=(N,), dtype=bool)
        f.create_dataset("is_augmented", shape=(N,), dtype=bool)
        f.create_dataset("source_id", shape=(N,), dtype=np.int32)
        f.create_dataset("bitrates_bpp", shape=(N, 6), dtype=np.float32)

        # Derive source_ids and is_augmented directly from samples manifest
        unique_paths = sorted(list({str(s.filepath) for s in samples}))
        path_to_id = {p: idx for idx, p in enumerate(unique_paths)}
        f["is_augmented"][:] = np.array([s.is_augmented for s in samples], dtype=bool)
        f["source_id"][:] = np.array([path_to_id[str(s.filepath)] for s in samples], dtype=np.int32)

        # Metadata attributes
        f.attrs["checkpoint"] = str(ckpt_path)
        f.attrs["lambda"] = lmbda_val
        f.attrs["total_samples"] = N
        f.attrs["target_k"] = args.target_k
        f.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 6. Extraction Loop
        cursor = 0
        t0 = time.time()
        with torch.no_grad():
            for batch_imgs, batch_labels, batch_is_train, _ in tqdm(dataloader, desc="Extracting latents"):
                B = batch_imgs.size(0)
                batch_imgs = batch_imgs.to(args.device)

                out = extractor(batch_imgs)

                # Store into HDF5
                f["latents_quantized"][cursor : cursor + B] = out["y_hat"].cpu().half().numpy()
                f["latents_continuous"][cursor : cursor + B] = out["y"].cpu().half().numpy()
                f["labels"][cursor : cursor + B] = batch_labels.numpy()
                f["is_train"][cursor : cursor + B] = batch_is_train.numpy()
                f["bitrates_bpp"][cursor : cursor + B] = out["bitrates_bpp"].cpu().float().numpy()

                cursor += B
                f.flush()

        elapsed = time.time() - t0
        print(f"[DONE] Extracted {N} samples in {elapsed:.1f}s ({N / max(elapsed, 0.001):.1f} samples/s)")

    # 7. Verify file on disk
    file_size_gb = out_path.stat().st_size / (1024**3)
    print(f"[SUCCESS] HDF5 Cache created: {out_path} ({file_size_gb:.2f} GB)")


if __name__ == "__main__":
    main(sys.argv[1:])
