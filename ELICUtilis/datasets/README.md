# ELICUtilis Datasets

## Overview
Dataset loader utilities for image compression training and evaluation. Provides a PyTorch `Dataset` wrapper for directory-based image datasets partitioned into train and validation split directories.

## Interface
- `[utils.py:ImageFolder]`: PyTorch `Dataset` subclass for loading directory-based image datasets.
- `[utils.py:ImageFolder.__init__]`: Initializes dataset loader for `root` path, optional callable `transform`, and split directory name (`split="train"` by default).
- `[utils.py:ImageFolder.__getitem__]`: Loads and converts image sample at integer `index` to RGB, applying optional `transform`.
- `[utils.py:ImageFolder.__len__]`: Returns total number of sample image files in the split directory.

## Invariants
- `[utils.py:ImageFolder.__init__]`: Subdirectory `Path(root) / split` must exist and be a directory; raises `RuntimeError` if missing.
- `[utils.py:ImageFolder.__getitem__]`: Files in `self.samples` must be readable image files convertible to RGB mode by PIL.

## Side Effects
- `[utils.py:ImageFolder.__init__]`: Reads filesystem directory contents via `Path.iterdir()` to populate `self.samples`.
- `[utils.py:ImageFolder.__getitem__]`: Reads image files from disk via `PIL.Image.open`.

## Verification
- Run: `<test-command>`
