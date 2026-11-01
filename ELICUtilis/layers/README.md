# ELICUtilis Layers

## Overview
Neural network building blocks and custom layers for learned image compression models, including Generalized Divisive Normalization (GDN/IGDN), masked convolutions for autoregressive entropy modeling, residual blocks, and self-attention modules.

## Interface
- `[gdn.py:GDN]`: Generalized Divisive Normalization layer (`nn.Module`) with learnable non-negative parameters `beta` and `gamma`. Supports inverse mode (`inverse=True`).
- `[gdn.py:GDN1]`: Simplified computationally efficient GDN layer subclassing `GDN` using absolute values instead of quadratic terms.
- `[layers.py:conv1x1]`: 1x1 2D convolution helper (`nn.Conv2d`).
- `[layers.py:conv3x3]`: 3x3 2D convolution helper with padding 1 (`nn.Conv2d`).
- `[layers.py:subpel_conv3x3]`: 3x3 sub-pixel convolution helper using `nn.PixelShuffle` for upsampling.
- `[layers.py:MaskedConv2d]`: Autoregressive masked 2D convolution with `'A'` or `'B'` masking patterns to prevent future pixel leakage.
- `[layers.py:CheckboardMaskedConv2d]`: Checkerboard-masked 2D convolution for two-pass spatial context modeling.
- `[layers.py:ResidualBlock]`: Standard residual block with two 3x3 convolutions and LeakyReLU activations.
- `[layers.py:ResidualBlockWithStride]`: Downsampling residual block with strided convolution and GDN.
- `[layers.py:ResidualBlockUpsample]`: Upsampling residual block with sub-pixel convolution and inverse GDN.
- `[layers.py:AttentionBlock]`: Self-attention gating block using cascaded residual units and sigmoid modulation.

## Invariants
- `[gdn.py:GDN]`: Input tensor `x` must be a 4-dimensional tensor `(B, C, H, W)` where channel dimension `C` matches `in_channels`.
- `[layers.py:MaskedConv2d.__init__]`: `mask_type` must be strictly `"A"` or `"B"`; raises `ValueError` if an invalid type is supplied.

## Side Effects
- `[layers.py:MaskedConv2d.forward]`: In-place mutation of `self.weight.data` via multiplication with `self.mask`.
- `[layers.py:CheckboardMaskedConv2d.forward]`: In-place mutation of `self.weight.data` via multiplication with `self.mask`.

## Verification
- Run: `pytest`
