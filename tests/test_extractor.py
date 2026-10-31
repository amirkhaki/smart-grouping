"""Unit tests for ELICFeatureExtractor wrapper."""

import unittest
import torch
import torch.nn as nn

from benchmarks.elic_extractor import ELICFeatureExtractor


class DummyCompressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)


class TestExtractor(unittest.TestCase):
    def test_freeze_and_eval_mode(self):
        """Extractor must set model to eval mode and freeze all parameters."""
        dummy = DummyCompressionModel()
        extractor = ELICFeatureExtractor(dummy)

        self.assertFalse(extractor.model.training)
        for p in extractor.model.parameters():
            self.assertFalse(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
