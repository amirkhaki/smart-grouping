# ELICUtilis

## Overview
Core utility subsystem supporting the ELIC learned image compression implementation. Groups specialized neural network layers and transforms, dataset loading wrappers, and model checkpoint management utilities.

## Subdirectories & Boundaries
- datasets: [datasets](./datasets/README.md) - PyTorch dataset loader implementations for directory-based train and validation image splits.
- layers: [layers](./layers/README.md) - Custom neural network layers including GDN/IGDN normalization, autoregressive and checkerboard masked convolutions, residual blocks, and self-attention mechanisms.
- utilis: [utilis](./utilis/README.md) - Checkpoint loading helpers and disk cleanup utility functions.
