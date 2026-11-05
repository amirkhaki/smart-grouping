"""Dataset loading, transforms, and deterministic 70/10/20 splitting for IQA benchmarks."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IQA_NAME_MAP = {
    "koniq10k": "KonIQ-10k",
    "koniq-10k": "KonIQ-10k",
    "koniq": "KonIQ-10k",
    "kadid10k": "KADID-10k",
    "kadid-10k": "KADID-10k",
    "kadid": "KADID-10k",
    "live": "LIVE",
    "csiq": "CSIQ",
    "tid2013": "TID2013",
    "pipal": "PIPAL",
}


def get_iqa_transform(crop_size: Optional[int] = 256) -> transforms.Compose:
    """Transform for IQA images.
    
    If crop_size is None or <= 0, returns transforms.ToTensor() for native full-frame
    loading without spatial cropping or interpolation distortion.
    """
    if crop_size is None or crop_size <= 0:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


def create_deterministic_split(
    total_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate deterministic, leak-free 70/10/20 train/val/test split masks.

    Args:
        total_samples: Total number of samples in the dataset.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation / early stopping (default 0.10).
        test_ratio: Fraction for final held-out test evaluation (default 0.20).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict containing boolean masks 'is_train', 'is_val', 'is_test' and index arrays.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = np.random.RandomState(seed)
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    n_train = int(round(train_ratio * total_samples))
    n_val = int(round(val_ratio * total_samples))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    is_train = np.zeros(total_samples, dtype=bool)
    is_val = np.zeros(total_samples, dtype=bool)
    is_test = np.zeros(total_samples, dtype=bool)

    is_train[train_indices] = True
    is_val[val_indices] = True
    is_test[test_indices] = True

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "is_train": is_train,
        "is_val": is_val,
        "is_test": is_test,
    }


def normalize_mos(scores: np.ndarray) -> np.ndarray:
    """Normalize MOS scores to strictly lie in [0, 1]."""
    s_min = float(np.nanmin(scores))
    s_max = float(np.nanmax(scores))
    if s_max - s_min < 1e-6:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


class IQADatasetWrapper(Dataset):
    """Wrapper around iqadataset to provide standardized sample dicts."""

    def __init__(
        self,
        dataset_name: str = "koniq10k",
        dataset_root: Union[str, Path] = "datasets",
        crop_size: Optional[int] = 256,
        no_crop: bool = False,
        download: bool = True,
    ):
        super().__init__()
        canonical_name = IQA_NAME_MAP.get(dataset_name.lower(), dataset_name)
        effective_crop = None if no_crop else crop_size
        self.transform = get_iqa_transform(crop_size=effective_crop)

        try:
            from iqadataset import load_dataset_pytorch
        except ImportError:
            import subprocess
            import sys
            print("[INIT] 'iqadataset' not found; installing via pip...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "iqadataset"])
            from iqadataset import load_dataset_pytorch
        self.ds = load_dataset_pytorch(
            canonical_name,
            dataset_root=str(dataset_root),
            attributes=["dis_img_path", "score"],
            transform=self.transform,
            download=download,
        )

        # Collect and normalize all scores
        raw_scores = []
        for i in range(len(self.ds)):
            item = self.ds[i]
            raw_scores.append(float(item["score"]))
        self.raw_scores = np.array(raw_scores, dtype=np.float32)
        self.normalized_scores = normalize_mos(self.raw_scores)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        img = item["dis_img"] if "dis_img" in item else item.get("img")
        return {
            "image": img,
            "raw_score": self.raw_scores[idx],
            "score": self.normalized_scores[idx],
            "index": idx,
        }
