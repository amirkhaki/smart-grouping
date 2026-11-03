"""GPU-side latent augmentations: random shift and CutMix."""

import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def random_shift(x: torch.Tensor, max_shift: int = 2, groups: int = 4) -> torch.Tensor:
    """Apply random spatial shift with reflect padding to batch of latent tensors.

    Args:
        x: Input tensor of shape (B, C, H, W) on device.
        max_shift: Maximum integer shift offset in pixels on each axis.
        groups: Number of chunks along the batch dimension to group for shared shift.

    Returns:
        Shifted tensor of shape (B, C, H, W) written to a new tensor.
    """
    if max_shift <= 0 or x.numel() == 0:
        return x

    H, W = x.shape[-2], x.shape[-1]
    if max_shift >= min(H, W):
        raise ValueError(
            f"max_shift ({max_shift}) must be strictly less than min(H, W) ({min(H, W)})"
        )

    B = x.size(0)
    padded = F.pad(x, (max_shift, max_shift, max_shift, max_shift), mode="reflect")

    out = torch.empty_like(x)
    chunks = torch.arange(B, device=x.device).chunk(groups)
    for chunk in chunks:
        dy = int(torch.randint(0, 2 * max_shift + 1, (1,), device=x.device).item())
        dx = int(torch.randint(0, 2 * max_shift + 1, (1,), device=x.device).item())
        out[chunk] = padded[chunk, :, dy : dy + H, dx : dx + W]

    return out


def apply_latent_aug(
    y_q: torch.Tensor,
    y_c: torch.Tensor,
    labels: torch.Tensor,
    shift: int = 0,
    cutmix_p: float = 0.0,
    cutmix_alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]:
    """Apply random shift and CutMix augmentations to latent representations.

    Args:
        y_q: Quantized latent tensor (B, C_q, H, W) or empty tensor.
        y_c: Continuous latent tensor (B, C_c, H, W) or empty tensor.
        labels: Class labels tensor (B,) on device.
        shift: Maximum pixel offset for random shift (0 to disable).
        cutmix_p: Probability of applying CutMix per batch.
        cutmix_alpha: Dirichlet/Beta distribution parameter for CutMix lambda.

    Returns:
        Tuple of (y_q, y_c, labels_b, lam) where labels_b is None and lam=1.0
        if CutMix is not applied.
    """
    # 1. Random spatial shift applied to each non-empty tensor independently
    if shift > 0:
        if y_q is not None and y_q.numel() > 0:
            y_q = random_shift(y_q, max_shift=shift)
        if y_c is not None and y_c.numel() > 0:
            y_c = random_shift(y_c, max_shift=shift)

    # 2. CutMix augmentation
    B = labels.size(0)
    if cutmix_p > 0.0 and B > 1 and random.random() < cutmix_p:
        ref = y_q if (y_q is not None and y_q.numel() > 0) else y_c
        if ref is not None and ref.numel() > 0:
            H, W = ref.shape[-2], ref.shape[-1]
            lam0 = float(np.random.beta(cutmix_alpha, cutmix_alpha))
            cut_h = int(H * math.sqrt(1.0 - lam0))
            cut_w = int(W * math.sqrt(1.0 - lam0))

            cy = int(torch.randint(0, H, (1,), device=labels.device).item())
            cx = int(torch.randint(0, W, (1,), device=labels.device).item())

            y1 = int(np.clip(cy - cut_h // 2, 0, H))
            y2 = int(np.clip(cy + (cut_h - cut_h // 2), 0, H))
            x1 = int(np.clip(cx - cut_w // 2, 0, W))
            x2 = int(np.clip(cx + (cut_w - cut_w // 2), 0, W))

            perm = torch.randperm(B, device=labels.device)
            labels_b = labels[perm]

            if y_q is not None and y_q.numel() > 0:
                y_q_aug = y_q.clone()
                y_q_aug[:, :, y1:y2, x1:x2] = y_q[perm, :, y1:y2, x1:x2]
                y_q = y_q_aug

            if y_c is not None and y_c.numel() > 0:
                y_c_aug = y_c.clone()
                y_c_aug[:, :, y1:y2, x1:x2] = y_c[perm, :, y1:y2, x1:x2]
                y_c = y_c_aug

            box_area = (y2 - y1) * (x2 - x1)
            lam = 1.0 - float(box_area) / float(H * W)
            return y_q, y_c, labels_b, lam

    return y_q, y_c, None, 1.0
