"""Unit tests for ExecutionProfiler timing and export."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from benchmarks.profiler import ExecutionProfiler


class TestProfiler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_epoch_and_summary(self):
        profiler = ExecutionProfiler(device="cpu")
        rec = profiler.record_epoch(
            epoch=1,
            t_data=5.0,
            t_transfer=0.5,
            t_forward=1.0,
            t_backward=1.5,
            t_val=0.5,
            t_overhead=0.2,
        )
        self.assertEqual(rec["epoch"], 1)
        self.assertAlmostEqual(rec["total_epoch_sec"], 8.7, places=1)
        self.assertTrue(rec["data_loading_pct"] > 50.0)

        # Summary strings
        epoch_str = profiler.format_epoch_summary(rec)
        self.assertIn("Data Loading", epoch_str)

        overall_str = profiler.format_overall_summary()
        self.assertIn("PRIMARY BOTTLENECK", overall_str)
        self.assertIn("Data Loading", overall_str)

        # Export CSV and plot
        csv_path = self.out_dir / "timing_profile.csv"
        plot_path = self.out_dir / "timing_profile.png"
        df = profiler.export(csv_path, plot_path)

        self.assertTrue(csv_path.is_file())
        self.assertTrue(plot_path.is_file())
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
