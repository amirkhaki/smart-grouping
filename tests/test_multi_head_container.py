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


if __name__ == "__main__":
    unittest.main()
