"""Unit tests for probe architectures and PreActBlock downsampling."""

import unittest
import torch

from benchmarks.config import NUM_CLASSES
from benchmarks.models.latent_head import LatentHead, PreActBlock
from benchmarks.models.linear_probe import LinearProbe


class TestProbeModels(unittest.TestCase):
    def test_preact_block_identity_stride1(self):
        """Stride=1 PreActBlock preserves spatial dimensions."""
        block = PreActBlock(in_ch=128, out_ch=128, stride=1)
        x = torch.randn(2, 128, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 16, 16))

    def test_preact_block_downsample_stride2(self):
        """Stride=2 PreActBlock correctly downsamples spatial dimensions from 16x16 to 8x8."""
        block = PreActBlock(in_ch=128, out_ch=128, stride=2)
        x = torch.randn(2, 128, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 8, 8))

    def test_linear_probe_variable_channels(self):
        """LinearProbe outputs logits of shape (B, 257) for all channel configurations."""
        channel_configs = [16, 32, 64, 128, 320]
        for c in channel_configs:
            probe = LinearProbe(in_channels=c, num_classes=NUM_CLASSES)
            x = torch.randn(4, c, 16, 16)
            logits = probe(x)
            self.assertEqual(logits.shape, (4, NUM_CLASSES))

    def test_latent_head_variable_channels(self):
        """LatentHead normalizes arbitrary input channels and outputs (B, 257)."""
        channel_configs = [16, 32, 64, 128, 320]
        for c in channel_configs:
            head = LatentHead(in_channels=c, stem_channels=128, num_classes=NUM_CLASSES)
            x = torch.randn(4, c, 16, 16)
            logits = head(x)
            self.assertEqual(logits.shape, (4, NUM_CLASSES))

    def test_preact_block_and_head_spatial_dropout(self):
        """PreActBlock and LatentHead support spatial dropout in train mode."""
        block = PreActBlock(in_ch=128, out_ch=128, stride=1, spatial_dropout=0.2)
        block.train()
        x = torch.randn(2, 128, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 16, 16))

        head = LatentHead(in_channels=320, stem_channels=128, num_classes=NUM_CLASSES, spatial_dropout=0.2)
        head.train()
        x_head = torch.randn(2, 320, 16, 16)
        logits = head(x_head)
        self.assertEqual(logits.shape, (2, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()
