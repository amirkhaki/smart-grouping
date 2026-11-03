"""Generate deterministic, leak-free validation split for Caltech-256 latents.

Validation samples are held out at the level of SOURCE IMAGES:
If a clean train image is held out, it and all augmented copies derived from it
leave the training set. The validation set contains only the clean copies of
the held-out sources.
"""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from benchmarks.datasets.caltech256 import (
    build_caltech256_sample_manifest,
    find_caltech256_root,
)

ROLE_TRAIN = np.int8(0)
ROLE_VAL = np.int8(1)
ROLE_TEST = np.int8(2)
ROLE_EXCLUDED = np.int8(3)

ROLE_NAMES = {
    0: "train",
    1: "val",
    2: "test",
    3: "excluded",
}


def assign_roles(
    labels: np.ndarray,
    is_train: np.ndarray,
    is_augmented: np.ndarray,
    source_id: np.ndarray,
    val_frac: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Assign sample roles ensuring leak-free source-level validation holdout.

    Args:
        labels: 1D array of integer class labels (length N)
        is_train: 1D boolean array indicating training split membership (length N)
        is_augmented: 1D boolean array indicating augmented copies (length N)
        source_id: 1D array of integer source image identifiers (length N)
        val_frac: Fraction of clean source images per class to hold out for validation
        seed: Random seed for deterministic SHA256-based ranking

    Returns:
        roles: 1D array of np.int8 with values:
            0 = train
            1 = val
            2 = test
            3 = excluded (augmented copies of held-out sources)
    """
    labels = np.asarray(labels)
    is_train = np.asarray(is_train, dtype=bool)
    is_augmented = np.asarray(is_augmented, dtype=bool)
    source_id = np.asarray(source_id)

    N = len(labels)
    if not (len(is_train) == N and len(is_augmented) == N and len(source_id) == N):
        raise ValueError(
            f"Input array lengths mismatch: labels={N}, is_train={len(is_train)}, "
            f"is_augmented={len(is_augmented)}, source_id={len(source_id)}"
        )

    roles = np.full(N, -1, dtype=np.int8)

    # 1. Test samples (~is_train) are always assigned to test (role 2)
    roles[~is_train] = ROLE_TEST

    classes = np.unique(labels)
    for c in classes:
        # Identify clean train samples for this class
        mask_clean_train = (labels == c) & is_train & (~is_augmented)
        unique_sources = np.unique(source_id[mask_clean_train])
        n_sources = len(unique_sources)

        if n_sources < 2:
            raise ValueError(
                f"Class {c} has {n_sources} clean train source(s). "
                "At least 2 clean train sources are required to create a validation split."
            )

        n_val = max(1, round(val_frac * n_sources))
        if n_val >= n_sources:
            n_val = n_sources - 1

        # Deterministic SHA256 ranking of sources
        ranked_sources = sorted(
            unique_sources,
            key=lambda s: hashlib.sha256(f"{seed}:{c}:{s}".encode("utf-8")).hexdigest(),
        )
        held_out_sources = set(ranked_sources[:n_val])

        # Assign roles for train samples in class c
        class_train_mask = (labels == c) & is_train
        class_indices = np.where(class_train_mask)[0]

        for idx in class_indices:
            src = source_id[idx]
            if src in held_out_sources:
                if not is_augmented[idx]:
                    roles[idx] = ROLE_VAL
                else:
                    roles[idx] = ROLE_EXCLUDED
            else:
                roles[idx] = ROLE_TRAIN

    if np.any(roles < 0):
        unassigned = np.where(roles < 0)[0]
        raise RuntimeError(f"Internal error: {len(unassigned)} samples were not assigned a role.")

    # Leak assertions
    train_sources = set(source_id[roles == ROLE_TRAIN])
    val_sources = set(source_id[roles == ROLE_VAL])
    excluded_sources = set(source_id[roles == ROLE_EXCLUDED])

    if not train_sources.isdisjoint(val_sources):
        overlap = train_sources.intersection(val_sources)
        raise AssertionError(f"Leak detected: {len(overlap)} source_ids appear in both train and val: {overlap}")

    if np.any(is_augmented[roles == ROLE_VAL]):
        raise AssertionError("Leak detected: validation set contains augmented samples.")

    if not train_sources.isdisjoint(excluded_sources):
        overlap = train_sources.intersection(excluded_sources)
        raise AssertionError(f"Leak detected: {len(overlap)} source_ids appear in both train and excluded: {overlap}")

    if np.sum(roles == ROLE_TEST) != np.sum(~is_train):
        raise AssertionError(
            f"Test sample count mismatch: {np.sum(roles == ROLE_TEST)} vs expected {np.sum(~is_train)}"
        )

    # Class-level coverage check
    for c in classes:
        train_c = np.sum((labels == c) & (roles == ROLE_TRAIN))
        val_c = np.sum((labels == c) & (roles == ROLE_VAL))
        if train_c == 0:
            raise AssertionError(f"Class {c} has 0 training samples remaining after split.")
        if val_c == 0:
            raise AssertionError(f"Class {c} has 0 validation samples assigned.")

    return roles


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic, leak-free validation split for HDF5 cache.")
    parser.add_argument("--h5", type=Path, required=True, help="Path to caltech256_elic_latents.h5")
    parser.add_argument("--caltech-root", type=Path, default=None, help="Path to Caltech-256 root directory")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction of clean sources (default: 0.1)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic hashing (default: 0)")
    parser.add_argument("--out", type=Path, default=None, help="Output .npz path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for .npz file")
    return parser.parse_args(argv)


def derive_source_ids_and_augmented(
    h5_path: Path,
    caltech_root: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Read or reconstruct is_augmented and source_id arrays matching HDF5 archive.

    Returns:
        labels: 1D array of class labels (N,)
        is_train: 1D array of train flags (N,)
        is_augmented: 1D array of augmentation flags (N,)
        source_id: 1D array of integer source ids (N,)
        labels_sha256: Hex digest string of labels array
    """
    with h5py.File(str(h5_path), "r") as f:
        labels = f["labels"][:]
        is_train = f["is_train"][:]
        N = len(labels)
        target_k = int(f.attrs.get("target_k", 150))
        labels_sha256 = hashlib.sha256(labels.tobytes()).hexdigest()

        # 1. Direct read if already present in HDF5 archive
        if "is_augmented" in f and "source_id" in f:
            is_augmented = f["is_augmented"][:]
            source_id = f["source_id"][:]
            return labels, is_train, is_augmented, source_id, labels_sha256

    # 2. Reconstruct from Caltech-256 sample manifest
    if caltech_root is None:
        try:
            caltech_root = find_caltech256_root()
        except Exception as e:
            raise FileNotFoundError(
                f"HDF5 archive does not contain 'is_augmented'/'source_id', and Caltech-256 "
                f"root directory could not be located automatically: {e}. Provide --caltech-root."
            )

    print(f"[SPLIT] Rebuilding Caltech-256 sample manifest from: {caltech_root} (target_k={target_k})...")
    samples, _ = build_caltech256_sample_manifest(caltech_root, target_k=target_k)

    if len(samples) != N:
        raise ValueError(
            f"Manifest sample count ({len(samples)}) does not match HDF5 archive total ({N})."
        )

    manifest_labels = np.array([s.label for s in samples], dtype=np.int64)
    manifest_is_train = np.array([s.is_train for s in samples], dtype=bool)

    if not np.array_equal(manifest_labels, labels):
        raise ValueError("Manifest labels do not match HDF5 labels elementwise. Aborting.")
    if not np.array_equal(manifest_is_train, is_train):
        raise ValueError("Manifest is_train flags do not match HDF5 is_train elementwise. Aborting.")

    is_augmented = np.array([s.is_augmented for s in samples], dtype=bool)

    # Derive unique integer source_id from source filepaths
    unique_paths = sorted(list({str(s.filepath) for s in samples}))
    path_to_id = {p: idx for idx, p in enumerate(unique_paths)}
    source_id = np.array([path_to_id[str(s.filepath)] for s in samples], dtype=np.int32)

    return labels, is_train, is_augmented, source_id, labels_sha256


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    h5_path = Path(args.h5)
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 archive not found: {h5_path}")

    labels, is_train, is_augmented, source_id, labels_sha256 = derive_source_ids_and_augmented(
        h5_path=h5_path,
        caltech_root=args.caltech_root,
    )

    roles = assign_roles(
        labels=labels,
        is_train=is_train,
        is_augmented=is_augmented,
        source_id=source_id,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    # Determine output path
    if args.out is not None:
        out_path = Path(args.out)
    elif args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{h5_path.stem}_valsplit_s{args.seed}_f{args.val_frac}.npz"
    else:
        out_path = h5_path.parent / f"{h5_path.stem}_valsplit_s{args.seed}_f{args.val_frac}.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        role=roles.astype(np.int8),
        source_id=source_id.astype(np.int32),
        is_augmented=is_augmented.astype(bool),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        n_total=int(len(roles)),
        labels_sha256=labels_sha256,
    )
    print(f"[SPLIT] Saved validation split to: {out_path}")

    # Summary table
    n_train = int(np.sum(roles == ROLE_TRAIN))
    n_val = int(np.sum(roles == ROLE_VAL))
    n_test = int(np.sum(roles == ROLE_TEST))
    n_excluded = int(np.sum(roles == ROLE_EXCLUDED))
    n_total = len(roles)

    print("\n" + "=" * 60)
    print("VALIDATION SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total samples:     {n_total:,}")
    print(f"  Role 0 (Train):    {n_train:,} ({n_train / n_total * 100:.1f}%)")
    print(f"  Role 1 (Val):      {n_val:,} ({n_val / n_total * 100:.1f}%)")
    print(f"  Role 2 (Test):     {n_test:,} ({n_test / n_total * 100:.1f}%)")
    print(f"  Role 3 (Excluded): {n_excluded:,} ({n_excluded / n_total * 100:.1f}%)")
    print("-" * 60)

    classes = np.unique(labels)
    train_per_class = [np.sum((labels == c) & (roles == ROLE_TRAIN)) for c in classes]
    val_per_class = [np.sum((labels == c) & (roles == ROLE_VAL)) for c in classes]

    print(
        f"Train samples per class: min={np.min(train_per_class)}, "
        f"median={int(np.median(train_per_class))}, max={np.max(train_per_class)}"
    )
    print(
        f"Val samples per class:   min={np.min(val_per_class)}, "
        f"median={int(np.median(val_per_class))}, max={np.max(val_per_class)}"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
