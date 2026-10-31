"""Minimal linear probe classifier head."""

import torch
import torch.nn as nn

from benchmarks.config import NUM_CLASSES


class LinearProbe(nn.Module):
    """Minimal linear probe baseline.

    Architecture:
      Input (B, C_in, H, W)
      -> AdaptiveAvgPool2d((1, 1))
      -> Flatten()
      -> BatchNorm1d(C_in, eps=1e-5)
      -> Linear(C_in, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(in_channels, eps=1e-5)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Latent tensor of shape (B, in_channels, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.pool(x)
        x = self.flatten(x)
        x = self.bn(x)
        logits = self.fc(x)
        return logits
