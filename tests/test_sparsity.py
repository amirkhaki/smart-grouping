"""Unit tests for channel activation density and least-sparse subset selection."""

import unittest
import numpy as np

from benchmarks.sparsity import (
    format_density_summary,
    get_channel_slice,
    get_channel_slice_info,
    rank_channels_by_density,
)


class TestSparsityUtils(unittest.TestCase):
    def test_get_channel_slice_mapping(self):
        """Channels must map to correct ELIC slices based on index boundaries."""
        # S0: 0..15
        self.assertEqual(get_channel_slice(0), "S0")
        self.assertEqual(get_channel_slice(15), "S0")
        self.assertEqual(get_channel_slice_info(15), ("S0", 15))

        # S1: 16..31
        self.assertEqual(get_channel_slice(16), "S1")
        self.assertEqual(get_channel_slice(17), "S1")
        self.assertEqual(get_channel_slice_info(17), ("S1", 1))
        self.assertEqual(get_channel_slice(31), "S1")

        # S2: 32..63
        self.assertEqual(get_channel_slice(32), "S2")
        self.assertEqual(get_channel_slice_info(35), ("S2", 3))
        self.assertEqual(get_channel_slice(63), "S2")

        # S3: 64..127
        self.assertEqual(get_channel_slice(64), "S3")
        self.assertEqual(get_channel_slice(127), "S3")

        # S4: 128..319
        self.assertEqual(get_channel_slice(128), "S4")
        self.assertEqual(get_channel_slice(319), "S4")
        self.assertEqual(get_channel_slice_info(130), ("S4", 2))

    def test_rank_channels_by_density(self):
        """Ranking must sort channels in descending order of activation density."""
        mock_densities = np.zeros(320)
        mock_densities[17] = 0.85  # in S1
        mock_densities[3] = 0.90   # in S0
        mock_densities[45] = 0.50  # in S2

        ranked = rank_channels_by_density(mock_densities)
        self.assertEqual(len(ranked), 320)
        self.assertEqual(ranked[0]["channel"], 3)
        self.assertEqual(ranked[0]["slice"], "S0")
        self.assertEqual(ranked[0]["density"], 0.90)

        self.assertEqual(ranked[1]["channel"], 17)
        self.assertEqual(ranked[1]["slice"], "S1")
        self.assertEqual(ranked[1]["rel_idx"], 1)

        self.assertEqual(ranked[2]["channel"], 45)
        self.assertEqual(ranked[2]["slice"], "S2")

    def test_format_density_summary(self):
        """Summary text must contain channel indices and slice distribution."""
        mock_densities = np.zeros(320)
        mock_densities[0] = 0.9
        mock_densities[17] = 0.8
        mock_densities[35] = 0.7

        ranked = rank_channels_by_density(mock_densities)
        summary = format_density_summary(ranked, top_k_list=[8, 16])

        self.assertIn("Top 8 Least-Sparse Channels", summary)
        self.assertIn("ch_17", summary)
        self.assertIn("S1", summary)


if __name__ == "__main__":
    unittest.main()
