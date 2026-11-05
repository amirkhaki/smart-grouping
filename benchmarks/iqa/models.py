"""Capacity-normalized IQA regression heads and multi-head container."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.config import SLICE_SPECS
from benchmarks.iqa.metrics import pearson_correlation_loss
from benchmarks.models.latent_head import PreActBlock


class LatentIQARegressor(nn.Module):
    """Capacity-normalized PreAct ResNet regressor mapping latent slices to perceptual MOS in [0, 1]."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 256,
        spatial_dropout: float = 0.05,
        drop_path: float = 0.1,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels

        # 1. 1x1 Conv stem normalizing channel capacity
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Residual PreAct trunk
        self.b1 = PreActBlock(stem_channels, stem_channels, stride=1, spatial_dropout=spatial_dropout, drop_path=drop_path)
        self.b2 = PreActBlock(stem_channels, stem_channels, stride=1, spatial_dropout=spatial_dropout, drop_path=drop_path)
        self.b3 = PreActBlock(stem_channels, stem_channels, stride=2, spatial_dropout=spatial_dropout, drop_path=drop_path)

        # 3. Pooling and non-linear regression head
        self.neck = nn.Sequential(
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(stem_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # MOS targets normalized to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.neck(h)
        h = torch.flatten(h, 1)
        score = self.fc(h).squeeze(-1)  # (B,)
        return score


class JointIQALoss(nn.Module):
    """Combined Pearson correlation loss and smooth L1 regression loss."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l1 = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_plcc = pearson_correlation_loss(preds, targets)
        loss_l1 = self.l1(preds, targets)
        return self.alpha * loss_plcc + self.beta * loss_l1


class MultiHeadIQARegressionContainer(nn.Module):
    """Container evaluating multiple IQA regression heads concurrently in a single DataLoader pass."""

    # Default slice mapping: name -> (start_ch, end_ch)
    IQA_SLICE_MAP: Dict[str, Tuple[int, int]] = {
        "S0": (0, 16),
        "S1": (16, 32),
        "S2": (32, 64),
        "S3": (64, 128),
        "S4": (128, 320),
        "C1": (0, 32),
        "C2": (0, 64),
        "C3": (0, 128),
        "C4": (0, 320),
    }

    def __init__(
        self,
        views: Optional[List[str]] = None,
        stem_channels: int = 256,
        spatial_dropout: float = 0.05,
        drop_path: float = 0.1,
        loss_alpha: float = 1.0,
        loss_beta: float = 0.5,
    ):
        super().__init__()
        self.views = views or ["S0", "S1", "S2", "S3", "S4", "C4"]
        self.heads = nn.ModuleDict()
        self.loss_fn = JointIQALoss(alpha=loss_alpha, beta=loss_beta)
        self.head_metadata: Dict[str, Dict[str, Any]] = {}

        for view in self.views:
            if view not in self.IQA_SLICE_MAP:
                raise ValueError(f"Unknown slice view '{view}'. Supported: {list(self.IQA_SLICE_MAP.keys())}")
            start_ch, end_ch = self.IQA_SLICE_MAP[view]
            num_channels = end_ch - start_ch

            head = LatentIQARegressor(
                in_channels=num_channels,
                stem_channels=stem_channels,
                spatial_dropout=spatial_dropout,
                drop_path=drop_path,
            )
            self.heads[view] = head
            self.head_metadata[view] = {
                "view": view,
                "in_channels": num_channels,
                "start_ch": start_ch,
                "end_ch": end_ch,
                "stem_channels": stem_channels,
                "num_params": sum(p.numel() for p in head.parameters()),
            }

    def forward(self, full_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute predictions across all heads from (B, 320, 16, 16) full latent tensors."""
        predictions = {}
        for view, head in self.heads.items():
            start_ch, end_ch = self.IQA_SLICE_MAP[view]
            x_slice = full_latents[:, start_ch:end_ch, :, :]
            predictions[view] = head(x_slice)
        return predictions

    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute summed multi-head loss and per-head scalar metrics."""
        per_head_losses = {}
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)

        for view, preds in predictions.items():
            head_loss = self.loss_fn(preds, targets)
            total_loss = total_loss + head_loss
            per_head_losses[view] = float(head_loss.item())

        return total_loss, per_head_losses
