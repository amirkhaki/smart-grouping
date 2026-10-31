"""Unit tests for latent slice channel definitions and boundaries."""

import unittest
from benchmarks.config import (
    CUMULATIVE_SLICES,
    ISOLATED_SLICES,
    LATENT_TOTAL_CHANNELS,
    SLICE_SPECS,
    SLICE_VIEW_NAMES,
)


class TestSlicing(unittest.TestCase):
    def test_unique_slice_count(self):
        """Deduplicating S0 [0:16] leaves exactly 9 unique slice views."""
        self.assertEqual(len(SLICE_SPECS), 9)
        self.assertEqual(len(SLICE_VIEW_NAMES), 9)

    def test_isolated_slice_boundaries(self):
        """Verify boundaries and channel widths of all 5 isolated slices."""
        expected = {
            "S0": (0, 16, 16),
            "S1": (16, 32, 16),
            "S2": (32, 64, 32),
            "S3": (64, 128, 64),
            "S4": (128, 320, 192),
        }
        for name, (start, end, width) in expected.items():
            spec = SLICE_SPECS[name]
            self.assertEqual(spec.start_ch, start)
            self.assertEqual(spec.end_ch, end)
            self.assertEqual(spec.num_channels, width)
            self.assertEqual(spec.category, "isolated")

    def test_cumulative_slice_boundaries(self):
        """Verify cumulative slice spans starting from channel 0."""
        expected = {
            "C1": (0, 32, 32),
            "C2": (0, 64, 64),
            "C3": (0, 128, 128),
            "C4": (0, 320, 320),
        }
        for name, (start, end, width) in expected.items():
            spec = SLICE_SPECS[name]
            self.assertEqual(spec.start_ch, start)
            self.assertEqual(spec.end_ch, end)
            self.assertEqual(spec.num_channels, width)
            self.assertEqual(spec.category, "cumulative")

    def test_full_capacity_c4(self):
        """Confirm C4 covers the full 320-channel latent capacity."""
        self.assertEqual(SLICE_SPECS["C4"].num_channels, LATENT_TOTAL_CHANNELS)
        self.assertEqual(SLICE_SPECS["C4"].end_ch, 320)


if __name__ == "__main__":
    unittest.main()
