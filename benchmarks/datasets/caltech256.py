"""Caltech-256 dataset mounting, stratified splitting, and balanced offline augmentation."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from benchmarks.config import (
    AUGMENTATION_TARGET_K,
    IMAGE_SIZE,
    NUM_CLASSES,
    TRAIN_SPLIT_RATIO,
)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp")


def find_caltech256_root(search_dir: Union[str, Path] = "/kaggle/input") -> Path:
    """Recursively search for the Caltech-256 categories folder (e.g. 256_ObjectCategories)."""
    search_path = Path(search_dir)
    if not search_path.exists():
        raise FileNotFoundError(f"Search directory does not exist: {search_path}")

    # Direct match check
    if (search_path / "256_ObjectCategories").is_dir():
        return search_path / "256_ObjectCategories"
    if (search_path / "001.ak47").is_dir():
        return search_path

    # Shallow breadth-first search
    for p in search_path.glob("**/256_ObjectCategories"):
        if p.is_dir():
            return p

    for p in search_path.glob("**/001.ak47"):
        if p.is_dir():
            return p.parent

    raise FileNotFoundError(
        f"Could not find 256_ObjectCategories directory under {search_path}. "
        "Ensure the dataset 'jessicali9530/caltech256' is properly attached."
    )


class CaltechSample:
    """Descriptor for a single prepared sample in the benchmarking archive."""

    def __init__(
        self,
        filepath: Path,
        label: int,
        is_train: bool,
        is_augmented: bool,
        sample_id: str,
    ):
        self.filepath = filepath
        self.label = label
        self.is_train = is_train
        self.is_augmented = is_augmented
        self.sample_id = sample_id


def build_caltech256_sample_manifest(
    data_root: Path,
    target_k: int = AUGMENTATION_TARGET_K,
    split_ratio: float = TRAIN_SPLIT_RATIO,
) -> Tuple[List[CaltechSample], Dict[int, str]]:
    """Scan Caltech-256 directory, compute deterministic hash split, and balance train split.

    Returns:
        samples: Complete ordered list of CaltechSample objects (train + test).
        class_to_name: Mapping of integer class index (0..256) to directory name.
    """
    category_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if len(category_dirs) < 256:
        raise ValueError(
            f"Expected at least 256 class directories in {data_root}, found {len(category_dirs)}"
        )

    class_to_name: Dict[int, str] = {i: d.name for i, d in enumerate(category_dirs)}
    all_samples: List[CaltechSample] = []

    train_count = 0
    test_count = 0

    for cls_idx, cat_dir in enumerate(category_dirs):
        img_paths = sorted([
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
        ])
        if not img_paths:
            continue

        train_base: List[Path] = []
        test_base: List[Path] = []

        # Deterministic hash partitioning
        for p in img_paths:
            hash_str = f"{cls_idx}_{p.name}"
            hash_val = int(hashlib.sha256(hash_str.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if hash_val < split_ratio:
                train_base.append(p)
            else:
                test_base.append(p)

        # Safeguard: ensure both partitions have at least 1 image
        if not test_base and train_base:
            test_base.append(train_base.pop())
        elif not train_base and test_base:
            train_base.append(test_base.pop())

        # 1. Base clean train samples
        for p in train_base:
            all_samples.append(CaltechSample(
                filepath=p,
                label=cls_idx,
                is_train=True,
                is_augmented=False,
                sample_id=f"cls_{cls_idx:03d}_{p.stem}_clean",
            ))
            train_count += 1

        # 2. Balanced offline augmentation for train split up to target_k
        if len(train_base) < target_k:
            needed = target_k - len(train_base)
            for i in range(needed):
                source_p = train_base[i % len(train_base)]
                all_samples.append(CaltechSample(
                    filepath=source_p,
                    label=cls_idx,
                    is_train=True,
                    is_augmented=True,
                    sample_id=f"cls_{cls_idx:03d}_{source_p.stem}_aug_{i:04d}",
                ))
                train_count += 1

        # 3. Clean test samples (strictly unaugmented)
        for p in test_base:
            all_samples.append(CaltechSample(
                filepath=p,
                label=cls_idx,
                is_train=False,
                is_augmented=False,
                sample_id=f"cls_{cls_idx:03d}_{p.stem}_test",
            ))
            test_count += 1

    print(
        f"[MANIFEST] Caltech-256 prepared: {train_count} train samples "
        f"(augmented to target_k={target_k}), {test_count} clean test samples. "
        f"Total: {len(all_samples)} samples."
    )
    return all_samples, class_to_name


class Caltech256PreparedDataset(Dataset):
    """PyTorch Dataset loading images and applying offline train-augmentation or test-crop."""

    def __init__(
        self,
        samples: List[CaltechSample],
        image_size: Tuple[int, int] = IMAGE_SIZE,
    ):
        self.samples = samples
        self.image_size = image_size

        # Unaugmented / Test transform: shortest edge resized to 256, center cropped to 256x256
        self.clean_transform = transforms.Compose([
            transforms.Resize(image_size[0]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        # Augmented transform for balancing the train split
        self.aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size[0], scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool, str]:
        item = self.samples[idx]
        with Image.open(item.filepath) as img:
            img = img.convert("RGB")
            if item.is_augmented:
                tensor = self.aug_transform(img)
            else:
                tensor = self.clean_transform(img)

        return tensor, item.label, item.is_train, item.sample_id
