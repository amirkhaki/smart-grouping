"""Unit tests for channel activation density and least-sparse subset selection."""

import unittest
import numpy as np

from benchmarks.sparsity import (
    format_density_summary,
    format_rate_summary,
    get_channel_slice,
    get_channel_slice_info,
    rank_channels_by_density,
    rank_channels_by_rate,
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

    def test_rank_channels_by_rate(self):
        """Rate ranking must sort channels in descending order of bpp."""
        mock_rates = np.zeros(320)
        mock_rates[10] = 0.0052  # in S0
        mock_rates[75] = 0.0120  # in S3
        mock_rates[200] = 0.0015 # in S4

        ranked = rank_channels_by_rate(mock_rates)
        self.assertEqual(len(ranked), 320)

        # Highest rate first
        self.assertEqual(ranked[0]["channel"], 75)
        self.assertEqual(ranked[0]["slice"], "S3")
        self.assertAlmostEqual(ranked[0]["rate_bpp"], 0.0120)

        # Second highest
        self.assertEqual(ranked[1]["channel"], 10)
        self.assertEqual(ranked[1]["slice"], "S0")
        self.assertAlmostEqual(ranked[1]["rate_bpp"], 0.0052)

        # Third highest
        self.assertEqual(ranked[2]["channel"], 200)
        self.assertEqual(ranked[2]["slice"], "S4")
        self.assertAlmostEqual(ranked[2]["rate_bpp"], 0.0015)

    def test_format_rate_summary(self):
        """Rate summary report must contain total decodable rate and slice breakdown."""
        mock_rates = np.zeros(320)
        mock_rates[10] = 0.0050
        mock_rates[75] = 0.0100

        ranked = rank_channels_by_rate(mock_rates)
        summary = format_rate_summary(ranked, top_k_list=[8, 16])

        self.assertIn("ELIC PER-CHANNEL TRANSMISSION RATE RANKING", summary)
        self.assertIn("Top 8 Highest-Rate Channels", summary)
        self.assertIn("Total Decodable Rate:", summary)
        self.assertIn("ch_75", summary)
        self.assertIn("S3", summary)


if __name__ == "__main__":
    unittest.main()
