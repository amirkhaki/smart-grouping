"""Unit tests for MultiHeadClassifierContainer execution and gradient propagation."""

import unittest
import torch

from benchmarks.config import NUM_CLASSES
from benchmarks.models.multi_head_container import MultiHeadClassifierContainer


class TestMultiHeadContainer(unittest.TestCase):
    def test_container_head_count(self):
        """Container must initialize exactly 36 heads (9 views x 2 archs x 2 quants)."""
        container = MultiHeadClassifierContainer(num_classes=NUM_CLASSES)
        self.assertEqual(len(container.heads), 36)
        self.assertEqual(len(container.head_keys), 36)

    def test_forward_pass_and_logits_shape(self):
        """All 36 heads must return logits of shape (B, 257)."""
        container = MultiHeadClassifierContainer(num_classes=NUM_CLASSES)
        B = 2
        y_q = torch.randn(B, 320, 16, 16)
        y_c = torch.randn(B, 320, 16, 16)

        logits_dict = container(y_q, y_c)
        self.assertEqual(len(logits_dict), 36)
        for key, logits in logits_dict.items():
            self.assertEqual(logits.shape, (B, NUM_CLASSES), f"Head {key} produced wrong shape")

    def test_loss_computation_and_backward_propagation(self):
        """Total loss aggregates all heads and propagates gradients into all heads."""
        container = MultiHeadClassifierContainer(num_classes=NUM_CLASSES)
        B = 2
        y_q = torch.randn(B, 320, 16, 16)
        y_c = torch.randn(B, 320, 16, 16)
        targets = torch.tensor([0, 1], dtype=torch.long)

        logits_dict = container(y_q, y_c)
        total_loss, per_head_losses = container.compute_loss(logits_dict, targets)

        self.assertEqual(len(per_head_losses), 36)
        self.assertTrue(total_loss.requires_grad)

        total_loss.backward()

        # Verify every head's parameters received gradients
        for head_key, head in container.heads.items():
            for name, param in head.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(
                        param.grad,
                        f"Parameter {name} in head {head_key} did not receive gradients!",
                    )

    def test_loss_computation_with_label_smoothing(self):
        """Total loss computation supports label smoothing regularization."""
        container = MultiHeadClassifierContainer(
            num_classes=NUM_CLASSES,
            views=["C4"],
            architectures=["latent_head"],
            quant_modes=["quantized"],
        )
        B = 2
        y_q = torch.randn(B, 320, 16, 16)
        y_c = torch.randn(B, 320, 16, 16)
        targets = torch.tensor([0, 1], dtype=torch.long)

        logits_dict = container(y_q, y_c)
        loss_standard, _ = container.compute_loss(logits_dict, targets, label_smoothing=0.0)
        loss_smoothed, _ = container.compute_loss(logits_dict, targets, label_smoothing=0.1)

        self.assertTrue(loss_smoothed.requires_grad)
        self.assertNotEqual(loss_standard.item(), loss_smoothed.item())

    def test_stem_channels_ablation(self):
        """Container instantiates specified stem channel heads for ablation studies."""
        container = MultiHeadClassifierContainer(
            num_classes=NUM_CLASSES,
            views=["C4"],
            architectures=["latent_head"],
            stem_channels=[8, 16, 32, 128],
        )
        expected_keys = [
            "C4_latent_head_w8_quantized",
            "C4_latent_head_w8_continuous",
            "C4_latent_head_w16_quantized",
            "C4_latent_head_w16_continuous",
            "C4_latent_head_w32_quantized",
            "C4_latent_head_w32_continuous",
            "C4_latent_head_w128_quantized",
            "C4_latent_head_w128_continuous",
        ]
        self.assertEqual(len(container.heads), 8)
        self.assertEqual(sorted(container.head_keys), sorted(expected_keys))

        # Verify parameter scaling
        params_w8 = container.head_metadata["C4_latent_head_w8_quantized"]["num_params"]
        params_w16 = container.head_metadata["C4_latent_head_w16_quantized"]["num_params"]
        params_w32 = container.head_metadata["C4_latent_head_w32_quantized"]["num_params"]
        params_w128 = container.head_metadata["C4_latent_head_w128_quantized"]["num_params"]

        self.assertTrue(params_w8 < params_w16 < params_w32 < params_w128)

    def test_custom_channel_subsets(self):
        """Container instantiates heads for custom least-sparse channel subsets."""
        custom_subsets = {
            "LS8": [0, 1, 2, 3, 16, 17, 32, 33],
            "LS16": list(range(16)),
        }
        container = MultiHeadClassifierContainer(
            num_classes=NUM_CLASSES,
            architectures=["latent_head"],
            quant_modes=["quantized"],
            custom_channel_subsets=custom_subsets,
        )
        self.assertEqual(len(container.heads), 2)
        self.assertIn("LS8_latent_head_quantized", container.heads)
        self.assertIn("LS16_latent_head_quantized", container.heads)
        self.assertEqual(container.head_metadata["LS8_latent_head_quantized"]["in_channels"], 8)
        self.assertEqual(container.head_metadata["LS16_latent_head_quantized"]["in_channels"], 16)

    def test_native_width(self):
        """Container instantiates heads with stem_channels == in_channels when native_width=True."""
        custom_subsets = {
            "LS8": list(range(8)),
            "LS16": list(range(16)),
        }
        container = MultiHeadClassifierContainer(
            num_classes=NUM_CLASSES,
            views=["S0"],
            architectures=["latent_head"],
            quant_modes=["quantized"],
            custom_channel_subsets=custom_subsets,
            native_width=True,
        )
        self.assertEqual(len(container.heads), 3)
        self.assertIn("LS8_latent_head_native_quantized", container.heads)
        self.assertIn("LS16_latent_head_native_quantized", container.heads)
        self.assertIn("S0_latent_head_native_quantized", container.heads)

        meta_ls8 = container.head_metadata["LS8_latent_head_native_quantized"]
        self.assertEqual(meta_ls8["stem_channels"], 8)
        self.assertTrue(meta_ls8["native_width"])
        self.assertEqual(meta_ls8["arch"], "latent_head_native")

        meta_ls16 = container.head_metadata["LS16_latent_head_native_quantized"]
        self.assertEqual(meta_ls16["stem_channels"], 16)
        self.assertTrue(meta_ls16["native_width"])

        meta_s0 = container.head_metadata["S0_latent_head_native_quantized"]
        self.assertEqual(meta_s0["stem_channels"], 16)
        self.assertTrue(meta_s0["native_width"])

        # Parameters should strictly scale with native width
        self.assertTrue(meta_ls8["num_params"] < meta_ls16["num_params"])


if __name__ == "__main__":
    unittest.main()
