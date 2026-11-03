# Benchmarks Models

## Overview
Probe classifier architectures and multi-head container supporting concurrent evaluation of linear probes and PreAct ResNet heads across ELIC latent slices, quantization states, and stem widths.

## Interface
- `[linear_probe.py:LinearProbe]`: Minimal linear probe classifier comprising `AdaptiveAvgPool2d((1, 1))`, `Flatten`, `BatchNorm1d(in_channels)`, and `Linear(in_channels, num_classes)`.
- `[latent_head.py:PreActBlock]`: Pre-activation residual block module with `BatchNorm2d`, `ReLU`, 3x3 `Conv2d` layers, optional 2D spatial dropout, drop-path (stochastic depth), and optional 1x1 `Conv2d` shortcut projection.
- `[latent_head.py:LatentHead]`: Capacity-normalized PreAct ResNet probe head consisting of stem 1x1 conv, 3-block residual body with linearly ramped drop-path, neck, adaptive pooling, dropout, and linear classifier.
- `[multi_head_container.py:MultiHeadClassifierContainer]`: Multi-head container module holding probe heads across slice views, architectures, and quantization states with configurable spatial dropout, drop-path, and dynamic per-head early stopping.
- `[multi_head_container.py:MultiHeadClassifierContainer.freeze_head]`: Freezes parameters (`requires_grad=False`) and removes `head_key` from `active_head_keys` when validation performance plateaus.
- `[multi_head_container.py:MultiHeadClassifierContainer.is_active]`: Returns boolean indicating if a head is currently active in training.
- `[multi_head_container.py:MultiHeadClassifierContainer.all_frozen]`: Returns True when all registered heads have converged and been frozen.
- `[multi_head_container.py:MultiHeadClassifierContainer.compute_loss]`: Computes summed cross-entropy loss across active heads and returns total loss tensor and per-head loss dictionary.
- `[multi_head_container.py:MultiHeadClassifierContainer.compute_accuracy]`: Static method computing top-k accuracy percentages given logits and target tensors.
- `[__init__.py:LinearProbe]`: Re-exported linear probe module from `linear_probe.py`.
- `[__init__.py:PreActBlock]`: Re-exported PreAct residual block from `latent_head.py`.
- `[__init__.py:LatentHead]`: Re-exported PreAct ResNet head from `latent_head.py`.
- `[__init__.py:MultiHeadClassifierContainer]`: Re-exported multi-head container from `multi_head_container.py`.

## Invariants
- `[linear_probe.py:LinearProbe.forward]`: Input tensor `x` must be 4D shape `(B, in_channels, H, W)`.
- `[latent_head.py:LatentHead.forward]`: Input tensor `x` must match `in_channels` with spatial dimensions compatible with body downsampling.
- `[multi_head_container.py:MultiHeadClassifierContainer.__init__]`: Requested views must be valid keys in `SLICE_SPECS`; raises `ValueError` if view name is unknown.
- `[multi_head_container.py:MultiHeadClassifierContainer.forward]`: `y_quantized` and `y_continuous` input tensors must have shape `(B, 320, H, W)`.
- `[multi_head_container.py:MultiHeadClassifierContainer.freeze_head]`: Frozen heads have `requires_grad` set to `False` on all parameters and are excluded from training loss calculations.

## Side Effects
- `[multi_head_container.py:MultiHeadClassifierContainer.freeze_head]`: Modifies `requires_grad` attribute of head parameters and alters `active_head_keys` list state.

## Verification
- Run: `pytest`
