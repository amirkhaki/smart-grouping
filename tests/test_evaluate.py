"""Unit tests for evaluation metrics, training history, and overfitting analysis."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from benchmarks.evaluate import (
    export_master_results,
    export_overfitting_analysis,
    export_training_history,
    generate_generalization_gap_plot,
    generate_learning_curves_plot,
    generate_loss_curves_plot,
)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_and_plot_history(self):
        # Mock history records for 2 heads across 3 epochs
        history_records = [
            {
                "epoch": 1,
                "head_key": "S0_latent_head_native_quantized",
                "view": "S0",
                "category": "isolated",
                "arch": "latent_head_native",
                "quant": "quantized",
                "in_channels": 16,
                "stem_channels": 16,
                "num_params": 18961,
                "train_loss": 5.12,
                "train_top1": 5.2,
                "train_top5": 14.1,
                "val_loss": 4.90,
                "val_top1": 6.8,
                "val_top5": 16.3,
                "is_frozen": False,
            },
            {
                "epoch": 2,
                "head_key": "S0_latent_head_native_quantized",
                "view": "S0",
                "category": "isolated",
                "arch": "latent_head_native",
                "quant": "quantized",
                "in_channels": 16,
                "stem_channels": 16,
                "num_params": 18961,
                "train_loss": 3.82,
                "train_top1": 15.4,
                "train_top5": 30.5,
                "val_loss": 3.75,
                "val_top1": 16.2,
                "val_top5": 32.1,
                "is_frozen": False,
            },
            {
                "epoch": 3,
                "head_key": "S0_latent_head_native_quantized",
                "view": "S0",
                "category": "isolated",
                "arch": "latent_head_native",
                "quant": "quantized",
                "in_channels": 16,
                "stem_channels": 16,
                "num_params": 18961,
                "train_loss": 3.10,
                "train_top1": 21.0,
                "train_top5": 39.8,
                "val_loss": 3.40,
                "val_top1": 19.2,
                "val_top5": 37.8,
                "is_frozen": False,
            },
            {
                "epoch": 1,
                "head_key": "LS16_latent_head_native_quantized",
                "view": "LS16",
                "category": "least_sparse",
                "arch": "latent_head_native",
                "quant": "quantized",
                "in_channels": 16,
                "stem_channels": 16,
                "num_params": 18961,
                "train_loss": 5.05,
                "train_top1": 6.1,
                "train_top5": 15.8,
                "val_loss": 4.82,
                "val_top1": 7.5,
                "val_top5": 17.9,
                "is_frozen": False,
            },
            {
                "epoch": 2,
                "head_key": "LS16_latent_head_native_quantized",
                "view": "LS16",
                "category": "least_sparse",
                "arch": "latent_head_native",
                "quant": "quantized",
                "in_channels": 16,
                "stem_channels": 16,
                "num_params": 18961,
                "train_loss": 3.70,
                "train_top1": 17.2,
                "train_top5": 32.4,
                "val_loss": 3.60,
                "val_top1": 17.8,
                "val_top5": 34.0,
                "is_frozen": False,
            },
            {
                "epoch": 3,
                "head_key": "LS16_latent_head_native_quantized",
                "view": "LS16",
                "category": "least_sparse",
                "arch": "latent_head_native",
                "quant": "quantized",
                "in_channels": 16,
                "stem_channels": 16,
                "num_params": 18961,
                "train_loss": 2.95,
                "train_top1": 22.5,
                "train_top5": 41.2,
                "val_loss": 3.30,
                "val_top1": 20.5,
                "val_top5": 38.8,
                "is_frozen": False,
            },
        ]

        history_csv = self.out_dir / "training_history.csv"
        history_df = export_training_history(history_records, history_csv)
        self.assertTrue(history_csv.is_file())
        self.assertEqual(len(history_df), 6)

        # Overfitting analysis
        overfitting_csv = self.out_dir / "overfitting_analysis.csv"
        overfitting_df = export_overfitting_analysis(pd.DataFrame(), history_df, overfitting_csv)
        self.assertTrue(overfitting_csv.is_file())
        self.assertEqual(len(overfitting_df), 2)
        self.assertIn("generalization_gap", overfitting_df.columns)
        self.assertIn("diagnosis", overfitting_df.columns)

        # Verify plots generate without error
        curves_png = self.out_dir / "train_val_curves.png"
        generate_learning_curves_plot(history_df, curves_png)
        self.assertTrue(curves_png.is_file())

        loss_png = self.out_dir / "train_val_loss_curves.png"
        generate_loss_curves_plot(history_df, loss_png)
        self.assertTrue(loss_png.is_file())

        gap_png = self.out_dir / "generalization_gap.png"
        generate_generalization_gap_plot(overfitting_df, gap_png)
        self.assertTrue(gap_png.is_file())


if __name__ == "__main__":
    unittest.main()
