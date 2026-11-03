"""Unit tests for deterministic leak-free validation splitting."""

import unittest
import numpy as np

from benchmarks.make_val_split import (
    ROLE_TRAIN,
    ROLE_VAL,
    ROLE_TEST,
    ROLE_EXCLUDED,
    assign_roles,
)


class TestMakeValSplit(unittest.TestCase):
    def setUp(self):
        """Create a synthetic dataset with 5 classes and cycling augmentations."""
        np.random.seed(42)
        self.num_classes = 5
        self.labels = []
        self.is_train = []
        self.is_augmented = []
        self.source_id = []

        curr_source_id = 0
        for c in range(self.num_classes):
            # Clean train sources: 10 sources per class (c=0..3), 5 sources for c=4
            n_clean = 10 if c < 4 else 5
            clean_sources = list(range(curr_source_id, curr_source_id + n_clean))
            curr_source_id += n_clean

            # Clean train samples
            for s in clean_sources:
                self.labels.append(c)
                self.is_train.append(True)
                self.is_augmented.append(False)
                self.source_id.append(s)

            # Augmented train samples (cycling over clean sources up to target_k=15)
            n_aug = 15 - n_clean
            for i in range(n_aug):
                s = clean_sources[i % len(clean_sources)]
                self.labels.append(c)
                self.is_train.append(True)
                self.is_augmented.append(True)
                self.source_id.append(s)

            # Clean test samples: 4 test samples per class
            for _ in range(4):
                s = curr_source_id
                curr_source_id += 1
                self.labels.append(c)
                self.is_train.append(False)
                self.is_augmented.append(False)
                self.source_id.append(s)

        self.labels = np.array(self.labels, dtype=np.int64)
        self.is_train = np.array(self.is_train, dtype=bool)
        self.is_augmented = np.array(self.is_augmented, dtype=bool)
        self.source_id = np.array(self.source_id, dtype=np.int32)
        self.N = len(self.labels)

    def test_assign_roles_leak_free_and_partitioning(self):
        """assign_roles generates valid, leak-free roles satisfying all constraints."""
        roles = assign_roles(
            labels=self.labels,
            is_train=self.is_train,
            is_augmented=self.is_augmented,
            source_id=self.source_id,
            val_frac=0.2,
            seed=0,
        )

        self.assertEqual(len(roles), self.N)

        # 1. Role counts add up to N
        n_train = np.sum(roles == ROLE_TRAIN)
        n_val = np.sum(roles == ROLE_VAL)
        n_test = np.sum(roles == ROLE_TEST)
        n_excluded = np.sum(roles == ROLE_EXCLUDED)
        self.assertEqual(n_train + n_val + n_test + n_excluded, self.N)

        # 2. Test samples are strictly role 2
        self.assertEqual(n_test, np.sum(~self.is_train))
        self.assertTrue(np.all(roles[~self.is_train] == ROLE_TEST))

        # 3. Validation contains ONLY clean samples
        val_indices = np.where(roles == ROLE_VAL)[0]
        self.assertTrue(np.all(~self.is_augmented[val_indices]))
        self.assertTrue(np.all(self.is_train[val_indices]))

        # 4. Train and Val source_ids are strictly disjoint
        train_sources = set(self.source_id[roles == ROLE_TRAIN])
        val_sources = set(self.source_id[roles == ROLE_VAL])
        self.assertTrue(train_sources.isdisjoint(val_sources))

        # 5. Held-out sources' augmented copies are strictly role 3 (excluded)
        excluded_sources = set(self.source_id[roles == ROLE_EXCLUDED])
        self.assertTrue(train_sources.isdisjoint(excluded_sources))
        self.assertTrue(excluded_sources.issubset(val_sources))

        # 6. Every class keeps >=1 train source and has >=1 val sample
        for c in range(self.num_classes):
            c_train = np.sum((self.labels == c) & (roles == ROLE_TRAIN))
            c_val = np.sum((self.labels == c) & (roles == ROLE_VAL))
            self.assertGreaterEqual(c_train, 1)
            self.assertGreaterEqual(c_val, 1)

    def test_reproducibility_and_seed_sensitivity(self):
        """Identical seeds yield identical splits; differing seeds yield different splits."""
        roles_s0_a = assign_roles(self.labels, self.is_train, self.is_augmented, self.source_id, seed=0)
        roles_s0_b = assign_roles(self.labels, self.is_train, self.is_augmented, self.source_id, seed=0)
        roles_s1 = assign_roles(self.labels, self.is_train, self.is_augmented, self.source_id, seed=1)

        np.testing.assert_array_equal(roles_s0_a, roles_s0_b)
        self.assertFalse(np.array_equal(roles_s0_a, roles_s1))

    def test_class_fewer_than_two_clean_sources_raises(self):
        """Classes with fewer than 2 clean train sources raise ValueError."""
        # Class 0 only has 1 clean train source
        labels = np.array([0, 0, 0], dtype=np.int64)
        is_train = np.array([True, True, False], dtype=bool)
        is_augmented = np.array([False, True, False], dtype=bool)
        source_id = np.array([10, 10, 20], dtype=np.int32)

        with self.assertRaises(ValueError):
            assign_roles(labels, is_train, is_augmented, source_id, val_frac=0.1)


if __name__ == "__main__":
    unittest.main()
