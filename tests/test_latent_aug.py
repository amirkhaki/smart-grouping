"""Unit tests for latent space augmentations and drop-path."""

import unittest
import torch

from benchmarks.latent_aug import apply_latent_aug, random_shift
from benchmarks.models.latent_head import LatentHead, PreActBlock


class TestLatentAug(unittest.TestCase):
    def test_random_shift_shape_and_no_in_place_mutation(self):
        """random_shift preserves shape, returns identical on max_shift=0, does not mutate in place."""
        x = torch.randn(8, 64, 16, 16)
        x_clone = x.clone()

        # max_shift=0 returns identical tensor
        out_zero = random_shift(x, max_shift=0)
        torch.testing.assert_close(out_zero, x)

        # random shift preserves shape and does not mutate in place
        out_shifted = random_shift(x, max_shift=2)
        self.assertEqual(out_shifted.shape, (8, 64, 16, 16))
        torch.testing.assert_close(x, x_clone)

        # max_shift >= H raises ValueError
        with self.assertRaises(ValueError):
            random_shift(x, max_shift=16)

    def test_apply_latent_aug_cutmix_and_defaults(self):
        """apply_latent_aug with p=1.0 applies CutMix; with all off returns inputs unchanged."""
        torch.manual_seed(0)
        B, C, H, W = 4, 32, 16, 16
        y_q = torch.randn(B, C, H, W)
        y_c = torch.randn(B, C, H, W)
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        y_q_orig = y_q.clone()
        y_c_orig = y_c.clone()

        # All off returns inputs untouched
        q_out, c_out, labels_b, lam = apply_latent_aug(
            y_q, y_c, labels, shift=0, cutmix_p=0.0
        )
        self.assertIsNone(labels_b)
        self.assertEqual(lam, 1.0)
        torch.testing.assert_close(q_out, y_q_orig)
        torch.testing.assert_close(c_out, y_c_orig)

        # Empty y_c passes through untouched
        empty_c = torch.empty(0)
        q_out, c_out, _, _ = apply_latent_aug(y_q, empty_c, labels, shift=1, cutmix_p=0.0)
        self.assertEqual(c_out.numel(), 0)

        # CutMix with p=1.0 and fixed seed
        q_aug, c_aug, labels_b, lam = apply_latent_aug(
            y_q, y_c, labels, shift=0, cutmix_p=1.0, cutmix_alpha=1.0
        )
        self.assertEqual(q_aug.shape, y_q.shape)
        self.assertEqual(c_aug.shape, y_c.shape)
        self.assertIsNotNone(labels_b)
        self.assertGreater(lam, 0.0)
        self.assertLessEqual(lam, 1.0)
        # Inputs were not mutated in place
        torch.testing.assert_close(y_q, y_q_orig)
        torch.testing.assert_close(y_c, y_c_orig)
        # Outputs differ from original inputs
        self.assertFalse(torch.equal(q_aug, y_q_orig))

    def test_preact_block_drop_path(self):
        """PreActBlock in train mode with drop_path~1.0 returns x (shortcut); eval mode does not."""
        torch.manual_seed(0)
        block = PreActBlock(16, 16, stride=1, drop_path=0.999999)
        x = torch.randn(4, 16, 8, 8)

        # Train mode: residual branch is dropped to zero, returns identity shortcut x
        block.train()
        out_train = block(x)
        torch.testing.assert_close(out_train, x)

        # Eval mode: residual branch is active, output differs from x
        block.eval()
        out_eval = block(x)
        self.assertFalse(torch.allclose(out_eval, x))

        # Out-of-range drop_path raises ValueError
        with self.assertRaises(ValueError):
            PreActBlock(16, 16, drop_path=1.0)
        with self.assertRaises(ValueError):
            PreActBlock(16, 16, drop_path=-0.1)

    def test_latent_head_drop_path(self):
        """LatentHead with drop_path outputs correct shape and has identical param count as drop_path=0."""
        head_dp = LatentHead(32, 64, 10, drop_path=0.2)
        head_base = LatentHead(32, 64, 10, drop_path=0.0)

        x = torch.randn(4, 32, 16, 16)
        out = head_dp(x)
        self.assertEqual(out.shape, (4, 10))

        params_dp = sum(p.numel() for p in head_dp.parameters())
        params_base = sum(p.numel() for p in head_base.parameters())
        self.assertEqual(params_dp, params_base)


if __name__ == "__main__":
    unittest.main()
