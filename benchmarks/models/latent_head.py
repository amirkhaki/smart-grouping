"""Capacity-normalized PreAct ResNet probe head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.config import DEFAULT_DROPOUT, LATENT_HEAD_STEM_CHANNELS, NUM_CLASSES


class PreActBlock(nn.Module):
    """Pre-activation Residual Block with optional shortcut projection, 2D spatial dropout, and drop-path."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        spatial_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if not (0.0 <= drop_path < 1.0):
            raise ValueError(f"drop_path must be in range [0.0, 1.0), got {drop_path}")
        self.drop_path = float(drop_path)
        self.b1 = nn.BatchNorm2d(in_ch, eps=1e-5)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.drop = nn.Dropout2d(p=spatial_dropout) if spatial_dropout > 0.0 else nn.Identity()
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.relu(self.b1(x))
        sc = self.shortcut(o if isinstance(self.shortcut, nn.Conv2d) else x)
        h = self.drop(F.relu(self.b2(self.c1(o))))
        r = self.c2(h)
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = (torch.rand(r.size(0), 1, 1, 1, device=r.device) < keep).to(r.dtype)
            r = r * mask / keep
        return r + sc


class LatentHead(nn.Module):
    """Capacity-normalized PreAct ResNet probe head.

    Architecture:
      Input (B, C_in, 16, 16)
      -> Stem: Conv2d(C_in, 128, kernel=1, bias=False) -> BatchNorm2d(128) -> ReLU()
      -> Body:
           PreActBlock(128, 128, stride=1, spatial_dropout, drop_path[0])   (16x16)
           PreActBlock(128, 128, stride=1, spatial_dropout, drop_path[1])   (16x16)
           PreActBlock(128, 128, stride=2, spatial_dropout, drop_path[2])   (8x8, strided downsample)
      -> Neck: BatchNorm2d(128) -> ReLU()
      -> Head:
           AdaptiveAvgPool2d((1, 1))
           Flatten()
           Dropout(p=0.3)
           Linear(128, num_classes)

    Drop-path rates ramp linearly across the body blocks:
    [0.0, drop_path * 0.5, drop_path]. Drop-path adds zero parameters.
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = LATENT_HEAD_STEM_CHANNELS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        spatial_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_classes = num_classes
        self.spatial_dropout = spatial_dropout
        self.drop_path = drop_path

        # 1. Stem (normalizes arbitrary slice channel counts to stem_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels, eps=1e-5),
            nn.ReLU(inplace=True),
        )

        # 2. Residual Body with optional 2D Spatial Dropout and Drop-Path
        dp_rates = [drop_path * i / 2.0 for i in range(3)]
        self.body = nn.Sequential(
            PreActBlock(stem_channels, stem_channels, stride=1, spatial_dropout=spatial_dropout, drop_path=dp_rates[0]),
            PreActBlock(stem_channels, stem_channels, stride=1, spatial_dropout=spatial_dropout, drop_path=dp_rates[1]),
            PreActBlock(stem_channels, stem_channels, stride=2, spatial_dropout=spatial_dropout, drop_path=dp_rates[2]),
        )

        # 3. Neck
        self.neck = nn.Sequential(
            nn.BatchNorm2d(stem_channels, eps=1e-5),
            nn.ReLU(inplace=True),
        )

        # 4. Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(stem_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Latent tensor of shape (B, in_channels, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.stem(x)
        x = self.body(x)
        x = self.neck(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
