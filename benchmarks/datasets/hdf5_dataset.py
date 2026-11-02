"""Thread-safe PyTorch Dataset reading cached ELIC latents from HDF5."""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5LatentDataset(Dataset):
    """PyTorch Dataset reading cached float16 ELIC latents from HDF5 in SWMR mode

    and casting on-the-fly to float32. Multi-worker safe via lazy file opening.
    """

    _archive_cache: Dict[Any, Any] = {}

    @classmethod
    def clear_cache(cls):
        """Free shared archive raw buffers from memory."""
        cls._archive_cache.clear()
        import gc
        gc.collect()

    def __init__(
        self,
        h5_path: Union[str, Path],
        split: Literal["train", "test", "all"] = "train",
        in_memory: bool = False,
        load_continuous: bool = True,
        load_quantized: bool = True,
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 archive not found: {self.h5_path}")

        self.split = split
        self.in_memory = in_memory
        self.load_continuous = load_continuous
        self.load_quantized = load_quantized
        self._h5_file: Optional[h5py.File] = None

        # Inspect metadata and partition indices without keeping handle open
        with h5py.File(str(self.h5_path), mode="r") as f:
            is_train = f["is_train"][:]
            self.total_archive_samples = len(is_train)
            if split == "train":
                self.indices = np.where(is_train)[0]
            elif split == "test":
                self.indices = np.where(~is_train)[0]
            elif split == "all":
                self.indices = np.arange(self.total_archive_samples)
            else:
                raise ValueError(f"Unknown split: {split}. Choose 'train', 'test', or 'all'.")

            # Cache global attributes
            self.attrs: Dict[str, Any] = dict(f.attrs)

        self.num_samples = len(self.indices)

        self._latents_q: Optional[torch.Tensor] = None
        self._latents_c: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._bitrates: Optional[torch.Tensor] = None

        if self.in_memory:
            cache_key = (str(self.h5_path.resolve()), load_quantized, load_continuous)
            if cache_key in HDF5LatentDataset._archive_cache:
                shared = HDF5LatentDataset._archive_cache[cache_key]
            else:
                with h5py.File(str(self.h5_path), mode="r") as f:
                    # Single contiguous C-level sequential read (avoids slow point selection over FUSE)
                    shared = {
                        "q": torch.from_numpy(f["latents_quantized"][:]) if load_quantized else None,
                        "c": torch.from_numpy(f["latents_continuous"][:]) if load_continuous else None,
                        "labels": torch.from_numpy(f["labels"][:]).long(),
                        "bitrates": torch.from_numpy(f["bitrates_bpp"][:]).float(),
                    }
                HDF5LatentDataset._archive_cache[cache_key] = shared

            self._latents_q = shared.get("q")
            self._latents_c = shared.get("c")
            self._labels = shared["labels"]
            self._bitrates = shared["bitrates"]

    def _get_handle(self) -> h5py.File:
        """Lazily obtain worker-local HDF5 file handle for disk streaming."""
        if self._h5_file is None:
            self._h5_file = h5py.File(str(self.h5_path), mode="r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample by index, casting float16 latents to float32 tensors."""
        real_idx = int(self.indices[idx])

        if self.in_memory:
            y_q = self._latents_q[real_idx].float() if self._latents_q is not None else torch.empty(0)
            y_c = self._latents_c[real_idx].float() if self._latents_c is not None else torch.empty(0)
            label = int(self._labels[real_idx].item())
            bitrates = self._bitrates[real_idx]
        else:
            h5 = self._get_handle()
            y_q = torch.from_numpy(h5["latents_quantized"][real_idx]).float() if self.load_quantized else torch.empty(0)
            y_c = torch.from_numpy(h5["latents_continuous"][real_idx]).float() if self.load_continuous else torch.empty(0)
            label = int(h5["labels"][real_idx])
            bitrates = torch.from_numpy(h5["bitrates_bpp"][real_idx]).float()

        return {
            "y_quantized": y_q,
            "y_continuous": y_c,
            "label": label,
            "bitrates_bpp": bitrates,
            "index": real_idx,
        }

    def close(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

    def __del__(self):
        self.close()
