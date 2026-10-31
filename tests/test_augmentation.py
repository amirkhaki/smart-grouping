"""Unit tests for stratified partitioning and class-balanced offline augmentation."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from benchmarks.datasets.caltech256 import (
    CaltechSample,
    build_caltech256_sample_manifest,
)


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create mock category directories (at least 256 directories for manifest validity)
        self.num_classes = 257
        for c in range(self.num_classes):
            cat_dir = self.root / f"{c:03d}.category_{c}"
            cat_dir.mkdir()
            # Category 0 has 10 images, category 1 has 20, others have 5
            count = 10 if c == 0 else (20 if c == 1 else 5)
            for i in range(count):
                img_path = cat_dir / f"img_{i:04d}.jpg"
                img = Image.new("RGB", (64, 64), color=(c % 255, i % 255, 100))
                img.save(img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_partition_integrity_and_zero_leakage(self):
        """Train and test splits have zero overlap and test split is 100% unaugmented."""
        target_k = 25
        samples, _ = build_caltech256_sample_manifest(
            self.root,
            target_k=target_k,
            split_ratio=0.8,
        )

        train_samples = [s for s in samples if s.is_train]
        test_samples = [s for s in samples if not s.is_train]

        # Check total counts
        self.assertTrue(len(train_samples) > 0)
        self.assertTrue(len(test_samples) > 0)

        # 1. Zero leakage: test file paths must not appear as unaugmented train source paths
        train_base_paths = {s.filepath for s in train_samples if not s.is_augmented}
        test_paths = {s.filepath for s in test_samples}
        overlap = train_base_paths.intersection(test_paths)
        self.assertEqual(len(overlap), 0, f"Found data leakage between train and test: {overlap}")

        # 2. Test samples must strictly be unaugmented
        for s in test_samples:
            self.assertFalse(s.is_augmented, "Test sample was marked as augmented!")

        # 3. Class 0 and 1 have < target_k base train images, so train count must equal exactly target_k
        cls0_train = [s for s in train_samples if s.label == 0]
        self.assertEqual(len(cls0_train), target_k)

        cls1_train = [s for s in train_samples if s.label == 1]
        self.assertEqual(len(cls1_train), target_k)


if __name__ == "__main__":
    unittest.main()
