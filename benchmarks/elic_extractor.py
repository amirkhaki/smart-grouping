"""Non-invasive latent representation extractor for frozen ELIC models."""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

try:
    from compressai.zoo import load_state_dict as compressai_load_state_dict
except ImportError:
    compressai_load_state_dict = None

from Network import TestModel


class ELICFeatureExtractor(nn.Module):
    """Encapsulates a frozen TestModel to extract continuous and quantized latents

    along with exact per-slice bitrates, without modifying Network.py.
    """

    def __init__(self, model: TestModel):
        super().__init__()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "ELICFeatureExtractor":
        """Instantiate TestModel, load checkpoint weights, and wrap in extractor."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        raw_state = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = raw_state.get("state_dict", raw_state)

        if compressai_load_state_dict is not None:
            state_dict = compressai_load_state_dict(state_dict)

        model = TestModel()
        model.load_state_dict(state_dict)
        model = model.to(device).eval()

        return cls(model)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Extract continuous (y) and quantized (y_hat) latents and exact bitrates.

        Args:
            x: Input RGB image batch of shape (B, 3, H, W) where H and W are
               multiples of 64 (standard Caltech-256 crop is 256x256).

        Returns:
            Dictionary with:
              - 'y': Continuous latent tensor (B, 320, H/16, W/16)
              - 'y_hat': Context-quantized latent tensor (B, 320, H/16, W/16)
              - 'bpp_z': Hyperprior bitrate tensor (B,)
              - 'bpp_slices': Per-slice bitrate tensor (B, 5)
              - 'bitrates_bpp': Combined bitrate tensor (B, 6): [z, s0, s1, s2, s3, s4]
        """
        B, _, H, W = x.size()
        num_pixels = H * W

        # 1. Continuous latent representation from analysis transform
        y = self.model.g_a(x)
        _, C, H_y, W_y = y.size()

        # 2. Hyperprior forward pass and deterministic integer rounding
        z = self.model.h_a(y)
        _, z_likelihoods = self.model.entropy_bottleneck(z)
        z_offset = self.model.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset

        # Bitrate of hyperprior: -sum(log2(p)) / num_pixels
        bpp_z = -torch.log2(z_likelihoods).sum(dim=(1, 2, 3)) / num_pixels  # (B,)

        # 3. Spatial parameter prior synthesis
        latent_means, latent_scales = self.model.h_s(z_hat).chunk(2, 1)

        # 4. Checkerboard pattern masks
        anchor = torch.zeros_like(y)
        non_anchor = torch.zeros_like(y)
        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.model.groups[1:], 1)
        anchor_split = torch.split(anchor, self.model.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.model.groups[1:], 1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H_y, W_y, device=x.device),
            [2 * i for i in self.model.groups[1:]],
            1,
        )

        y_hat_slices = []
        bpp_slices_list = []

        # 5. Sequential context-model decoding and quantization across the 5 slices
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support = torch.concat([latent_means, latent_scales], dim=1)
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.model.cc_transforms[slice_index - 1](support_slices)
                mean_ch, scale_ch = support_slices_ch.chunk(2, 1)
                support = torch.concat([mean_ch, scale_ch, latent_means, latent_scales], dim=1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.model.cc_transforms[slice_index - 1](support_slices)
                mean_ch, scale_ch = support_slices_ch.chunk(2, 1)
                support = torch.concat([mean_ch, scale_ch, latent_means, latent_scales], dim=1)

            # Checkerboard step 1: Anchor decoding
            means_anchor, scales_anchor = self.model.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            # Deterministic evaluation quantization: round(y - mu) + mu
            y_a_quant = torch.round(anchor_split[slice_index] - means_anchor) + means_anchor
            y_a_quant[:, :, 0::2, 1::2] = 0
            y_a_quant[:, :, 1::2, 0::2] = 0

            # Checkerboard step 2: Non-anchor decoding conditioned on reconstructed anchors
            masked_ctx = self.model.context_prediction[slice_index](y_a_quant)
            means_non_anchor, scales_non_anchor = self.model.ParamAggregation[slice_index](
                torch.concat([masked_ctx, support], dim=1)
            ).chunk(2, 1)

            y_na_quant = torch.round(non_anchor_split[slice_index] - means_non_anchor) + means_non_anchor
            y_na_quant[:, :, 0::2, 0::2] = 0
            y_na_quant[:, :, 1::2, 1::2] = 0

            # Combined reconstructed slice
            y_hat_slice = y_a_quant + y_na_quant
            y_hat_slices.append(y_hat_slice)

            # Assemble full means and scales for likelihood estimation
            means_full = torch.zeros_like(y_slice)
            scales_full = torch.zeros_like(y_slice)
            means_full[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_full[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            means_full[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_full[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

            scales_full[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_full[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            scales_full[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_full[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]

            _, slice_likelihood = self.model.gaussian_conditional(y_slice, scales_full, means=means_full)
            bpp_slice = -torch.log2(slice_likelihood).sum(dim=(1, 2, 3)) / num_pixels  # (B,)
            bpp_slices_list.append(bpp_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        bpp_slices = torch.stack(bpp_slices_list, dim=1)  # (B, 5)
        bitrates_bpp = torch.cat([bpp_z.unsqueeze(1), bpp_slices], dim=1)  # (B, 6)

        return {
            "y": y,
            "y_hat": y_hat,
            "bpp_z": bpp_z,
            "bpp_slices": bpp_slices,
            "bitrates_bpp": bitrates_bpp,
        }
