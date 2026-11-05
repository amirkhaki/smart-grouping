"""Capacity-normalized IQA regression heads and multi-head container."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.config import SLICE_SPECS
from benchmarks.iqa.metrics import pearson_correlation_loss
from benchmarks.models.latent_head import PreActBlock


class SoftPairwiseLogisticLoss(nn.Module):
    """Soft pairwise logistic (RankNet-style) loss with label-noise thresholding and anchor MSE.

    Operates on raw unbounded logits s in (-inf, +inf) and normalized MOS labels y in [0, 1].
    Near-ties where |y_i - y_j| <= tau are filtered out to avoid fitting crowd-worker noise.
    A small anchor MSE on sigmoid(s) keeps predictions calibrated on [0, 1].
    """

    def __init__(self, tau: float = 0.03, lambda_anchor: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.lambda_anchor = float(lambda_anchor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Pairwise differences
        ds = logits[:, None] - logits[None, :]  # (B, B)
        dy = targets[:, None] - targets[None, :]  # (B, B)

        # Exclude pairs within noise threshold tau (also excludes diagonal)
        mask = dy.abs() > self.tau

        if mask.any():
            pairwise_loss = F.softplus(-torch.sign(dy) * ds)[mask].mean()
        else:
            pairwise_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Calibration anchor
        anchor_loss = F.mse_loss(torch.sigmoid(logits), targets)

        return pairwise_loss + self.lambda_anchor * anchor_loss


class JointIQALoss(nn.Module):
    """Combined Pearson correlation loss and smooth L1 regression loss."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l1 = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # If preds are unbounded logits, map through sigmoid for bounded PLCC and L1
        p = torch.sigmoid(preds) if (preds.min() < 0.0 or preds.max() > 1.0) else preds
        loss_plcc = pearson_correlation_loss(p, targets)
        loss_l1 = self.l1(p, targets)
        return self.alpha * loss_plcc + self.beta * loss_l1


class LatentIQARegressor(nn.Module):
    """Capacity-normalized PreAct ResNet regressor mapping latent slices to perceptual MOS logits.

    Features:
      - Channel dropout on inputs
      - 1x1 stem normalizing slice width to stem_channels (default: 128)
      - Regularized trunk with 2 PreAct residual blocks
      - Spatial statistics pooling: concatenating spatial mean, standard deviation, and maximum
      - Raw logit output (unbounded) for direct RankNet gradient optimization
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 128,
        num_blocks: int = 2,
        channel_dropout: float = 0.1,
        spatial_dropout: float = 0.05,
        drop_path: float = 0.05,
        dropout_p: float = 0.2,
        use_stats_pooling: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.use_stats_pooling = use_stats_pooling

        # 0. Input channel dropout
        self.input_drop = nn.Dropout2d(p=channel_dropout) if channel_dropout > 0.0 else nn.Identity()

        # 1. 1x1 Conv stem normalizing channel capacity
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Residual PreAct trunk
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                PreActBlock(
                    stem_channels,
                    stem_channels,
                    stride=1,
                    spatial_dropout=spatial_dropout,
                    drop_path=drop_path,
                )
            )
        self.body = nn.Sequential(*blocks)

        # 3. Trunk output normalization
        self.neck = nn.Sequential(
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # 4. Pooling and linear regression head
        fc_in_dim = (stem_channels * 3) if use_stats_pooling else stem_channels
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1),
            # Raw logit output without sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_drop(x)
        h = self.stem(h)
        h = self.body(h)
        h = self.neck(h)

        if self.use_stats_pooling:
            mean = h.mean(dim=(-2, -1))
            var = h.var(dim=(-2, -1), unbiased=False)
            std = torch.sqrt(torch.clamp(var, min=1e-8))
            max_val = h.amax(dim=(-2, -1))
            pooled = torch.cat([mean, std, max_val], dim=1)
        else:
            pooled = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)

        score = self.fc(pooled).squeeze(-1)  # (B,) raw logits
        return score


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
        custom_subsets: Optional[Dict[str, List[int]]] = None,
        stem_channels: int = 128,
        num_blocks: int = 2,
        channel_dropout: float = 0.1,
        spatial_dropout: float = 0.05,
        drop_path: float = 0.05,
        dropout_p: float = 0.2,
        use_stats_pooling: bool = True,
        loss_type: str = "pairwise",
        loss_tau: float = 0.03,
        loss_anchor: float = 0.1,
        loss_alpha: float = 1.0,
        loss_beta: float = 0.5,
    ):
        super().__init__()
        requested_views = list(views) if views else ["S0", "S1", "S2", "S3", "S4", "C4"]
        if custom_subsets:
            for k in custom_subsets:
                if k not in requested_views:
                    requested_views.append(k)

        self.views = requested_views
        self.heads = nn.ModuleDict()
        self.loss_type = loss_type

        if loss_type == "pairwise":
            self.loss_fn = SoftPairwiseLogisticLoss(tau=loss_tau, lambda_anchor=loss_anchor)
        else:
            self.loss_fn = JointIQALoss(alpha=loss_alpha, beta=loss_beta)

        self.head_metadata: Dict[str, Dict[str, Any]] = {}
        self.view_selectors: Dict[str, Tuple[str, Any]] = {}

        for view in self.views:
            if view in self.IQA_SLICE_MAP:
                start_ch, end_ch = self.IQA_SLICE_MAP[view]
                num_channels = end_ch - start_ch
                self.view_selectors[view] = ("slice", (start_ch, end_ch))
                meta_extra = {"start_ch": start_ch, "end_ch": end_ch, "indices": None}
            elif custom_subsets and view in custom_subsets:
                indices = custom_subsets[view]
                num_channels = len(indices)
                self.view_selectors[view] = ("indices", torch.tensor(indices, dtype=torch.long))
                meta_extra = {"start_ch": None, "end_ch": None, "indices": indices}
            else:
                raise ValueError(
                    f"Unknown view '{view}'. Must be in IQA_SLICE_MAP ({list(self.IQA_SLICE_MAP.keys())}) "
                    f"or custom_subsets ({list(custom_subsets.keys()) if custom_subsets else []})"
                )

            head = LatentIQARegressor(
                in_channels=num_channels,
                stem_channels=stem_channels,
                num_blocks=num_blocks,
                channel_dropout=channel_dropout,
                spatial_dropout=spatial_dropout,
                drop_path=drop_path,
                dropout_p=dropout_p,
                use_stats_pooling=use_stats_pooling,
            )
            self.heads[view] = head
            self.head_metadata[view] = {
                "view": view,
                "in_channels": num_channels,
                "stem_channels": stem_channels,
                "num_params": sum(p.numel() for p in head.parameters()),
                **meta_extra,
            }

    def forward(self, full_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute predictions across all heads from full latent tensors."""
        predictions = {}
        for view, head in self.heads.items():
            mode, selector = self.view_selectors[view]
            if mode == "slice":
                start_ch, end_ch = selector
                x_slice = full_latents[:, start_ch:end_ch, :, :]
            else:
                idx = selector.to(full_latents.device)
                x_slice = full_latents[:, idx, :, :]
            predictions[view] = head(x_slice)
        return predictions

    def compute_head_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute un-summed tensor loss per head for independent optimization."""
        return {view: self.loss_fn(preds, targets) for view, preds in predictions.items()}

    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute summed multi-head loss and per-head scalar metrics."""
        losses = self.compute_head_losses(predictions, targets)
        total_loss = sum(losses.values())
        return total_loss, {v: float(l.item()) for v, l in losses.items()}
