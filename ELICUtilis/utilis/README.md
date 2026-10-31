# ELICUtilis Utilities

## Overview
Filesystem and model checkpoint utility helpers supporting training checkpoint lifecycle management and weights loading.

## Interface
- `[utilis.py:DelfileList]`: Recursively searches directory `path` and deletes all files whose names begin with `filestarts` (defaulting to `'checkpoint_last'`).
- `[utilis.py:load_checkpoint]`: Loads a PyTorch model checkpoint from `filepath` to CPU, automatically resolving `"network"` or `"state_dict"` key wrappers.

## Invariants
- `[utilis.py:load_checkpoint]`: Target `filepath` must be a readable path to a serialized PyTorch checkpoint file.

## Side Effects
- `[utilis.py:DelfileList]`: Filesystem mutation; permanently deletes matching files from disk using `os.remove`.
- `[utilis.py:load_checkpoint]`: Reads binary file contents from disk via `torch.load`.

## Verification
- Run: `<test-command>`
