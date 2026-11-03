"""Unit tests for HDF5 dataset loading, SWMR mode, and float32 casting."""

import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.datasets.hdf5_dataset import HDF5LatentDataset


class TestHDF5Dataset(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        self.h5_path = Path(self.temp_file.name)

        # Create mock HDF5 archive
        self.num_samples = 20
        with h5py.File(str(self.h5_path), "w") as f:
            f.create_dataset(
                "latents_quantized",
                data=np.random.randn(self.num_samples, 320, 16, 16).astype(np.float16),
                dtype=np.float16,
            )
            f.create_dataset(
                "latents_continuous",
                data=np.random.randn(self.num_samples, 320, 16, 16).astype(np.float16),
                dtype=np.float16,
            )
            f.create_dataset("labels", data=np.arange(self.num_samples) % 10, dtype=np.int64)
            # 15 train, 5 test
            is_train = np.array([True] * 15 + [False] * 5, dtype=bool)
            f.create_dataset("is_train", data=is_train, dtype=bool)
            f.create_dataset("bitrates_bpp", data=np.ones((self.num_samples, 6), dtype=np.float32))
            f.attrs["lambda"] = 0.032

    def tearDown(self):
        if self.h5_path.exists():
            self.h5_path.unlink()

    def test_split_filtering_and_dtype_casting(self):
        """HDF5LatentDataset correctly partitions splits and casts float16 -> float32."""
        train_ds = HDF5LatentDataset(self.h5_path, split="train")
        test_ds = HDF5LatentDataset(self.h5_path, split="test")

        self.assertEqual(len(train_ds), 15)
        self.assertEqual(len(test_ds), 5)

        sample = train_ds[0]
        self.assertEqual(sample["y_quantized"].dtype, torch.float32)
        self.assertEqual(sample["y_continuous"].dtype, torch.float32)
        self.assertEqual(sample["bitrates_bpp"].dtype, torch.float32)
        self.assertEqual(sample["y_quantized"].shape, (320, 16, 16))
        self.assertIsInstance(sample["label"], int)

    def test_multi_worker_dataloader(self):
        """Multi-worker DataLoader reads without concurrency conflict under SWMR mode."""
        dataset = HDF5LatentDataset(self.h5_path, split="train")
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

        batches = list(loader)
        self.assertTrue(len(batches) > 0)
        self.assertEqual(batches[0]["y_quantized"].shape[1:], (320, 16, 16))

    def test_in_memory_loading_matches_disk_streaming(self):
        """In-memory preloading yields identical shapes, labels, and values as disk streaming."""
        train_disk = HDF5LatentDataset(self.h5_path, split="train", in_memory=False)
        test_disk = HDF5LatentDataset(self.h5_path, split="test", in_memory=False)

        train_mem = HDF5LatentDataset(self.h5_path, split="train", in_memory=True)
        test_mem = HDF5LatentDataset(self.h5_path, split="test", in_memory=True)

        self.assertEqual(len(train_mem), len(train_disk))
        self.assertEqual(len(test_mem), len(test_disk))

        # Check all train samples match exactly
        for i in range(len(train_mem)):
            d_sample = train_disk[i]
            m_sample = train_mem[i]
            self.assertEqual(d_sample["label"], m_sample["label"])
            self.assertEqual(d_sample["index"], m_sample["index"])
            torch.testing.assert_close(d_sample["y_quantized"], m_sample["y_quantized"])
            torch.testing.assert_close(d_sample["bitrates_bpp"], m_sample["bitrates_bpp"])

        # Check all test samples match exactly
        for i in range(len(test_mem)):
            d_sample = test_disk[i]
            m_sample = test_mem[i]
            self.assertEqual(d_sample["label"], m_sample["label"])
            self.assertEqual(d_sample["index"], m_sample["index"])
            torch.testing.assert_close(d_sample["y_quantized"], m_sample["y_quantized"])

        # Test cache clearing
        HDF5LatentDataset.clear_cache()
        self.assertEqual(len(HDF5LatentDataset._archive_cache), 0)

    def test_split_file_selection_and_validation(self):
        """HDF5LatentDataset with split_file filters train/val/test and verifies labels SHA256."""
        import hashlib
        # Read labels from H5
        with h5py.File(str(self.h5_path), "r") as f:
            labels = f["labels"][:]
        labels_sha = hashlib.sha256(labels.tobytes()).hexdigest()

        # Create valid npz split file (10 train, 5 val, 5 test)
        roles = np.array([0] * 10 + [1] * 5 + [2] * 5, dtype=np.int8)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_npz:
            npz_path = Path(tmp_npz.name)
            np.savez(
                str(npz_path),
                role=roles,
                source_id=np.arange(len(roles), dtype=np.int32),
                is_augmented=np.zeros(len(roles), dtype=bool),
                val_frac=0.25,
                seed=0,
                n_total=len(roles),
                labels_sha256=labels_sha,
            )

        try:
            ds_train = HDF5LatentDataset(self.h5_path, split="train", split_file=npz_path)
            ds_val = HDF5LatentDataset(self.h5_path, split="val", split_file=npz_path)
            ds_test = HDF5LatentDataset(self.h5_path, split="test", split_file=npz_path)
            ds_all = HDF5LatentDataset(self.h5_path, split="all", split_file=npz_path)

            self.assertEqual(len(ds_train), 10)
            self.assertEqual(len(ds_val), 5)
            self.assertEqual(len(ds_test), 5)
            self.assertEqual(len(ds_all), 20)

            # Mismatched labels_sha256 must raise ValueError
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as bad_tmp_npz:
                bad_npz_path = Path(bad_tmp_npz.name)
                np.savez(
                    str(bad_npz_path),
                    role=roles,
                    source_id=np.arange(len(roles), dtype=np.int32),
                    is_augmented=np.zeros(len(roles), dtype=bool),
                    val_frac=0.25,
                    seed=0,
                    n_total=len(roles),
                    labels_sha256="bad_sha256_hash",
                )
            try:
                with self.assertRaises(ValueError):
                    HDF5LatentDataset(self.h5_path, split="val", split_file=bad_npz_path)
            finally:
                if bad_npz_path.exists():
                    bad_npz_path.unlink()
        finally:
            if npz_path.exists():
                npz_path.unlink()


if __name__ == "__main__":
    unittest.main()
