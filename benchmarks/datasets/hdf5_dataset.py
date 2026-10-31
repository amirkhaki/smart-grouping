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

    def __init__(
        self,
        h5_path: Union[str, Path],
        split: Literal["train", "test", "all"] = "train",
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 archive not found: {self.h5_path}")

        self.split = split
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

    def _get_handle(self) -> h5py.File:
        """Lazily obtain worker-local HDF5 file handle."""
        if self._h5_file is None:
            self._h5_file = h5py.File(str(self.h5_path), mode="r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample by index, casting float16 latents to float32 tensors."""
        h5 = self._get_handle()
        real_idx = int(self.indices[idx])

        # Read float16 latent arrays
        y_q_np = h5["latents_quantized"][real_idx]
        y_c_np = h5["latents_continuous"][real_idx]
        label = int(h5["labels"][real_idx])
        bitrates_np = h5["bitrates_bpp"][real_idx]

        # Convert to float32 PyTorch tensors
        y_quantized = torch.from_numpy(y_q_np).float()
        y_continuous = torch.from_numpy(y_c_np).float()
        bitrates = torch.from_numpy(bitrates_np).float()

        return {
            "y_quantized": y_quantized,
            "y_continuous": y_continuous,
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
