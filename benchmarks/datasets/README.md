# Benchmarks Datasets

## Overview
Dataset mounting, stratified deterministic partitioning, offline class balancing, and thread-safe HDF5 latent storage for ELIC representation benchmarking on Caltech-256.

## Interface
- `[caltech256.py:find_caltech256_root]`: Recursively searches search_dir (default `/kaggle/input`) for `256_ObjectCategories` or `001.ak47` category folder and returns `Path`.
- `[caltech256.py:CaltechSample]`: Data descriptor for a prepared benchmarking image sample with attributes `filepath`, `label`, `is_train`, `is_augmented`, and `sample_id`.
- `[caltech256.py:build_caltech256_sample_manifest]`: Scans `data_root`, performs deterministic SHA-256 hash partitioning (`split_ratio=0.80`), and balances training classes via offline augmentation up to `target_k` (default 150).
- `[caltech256.py:Caltech256PreparedDataset]`: PyTorch `Dataset` subclass loading Caltech-256 images with clean center-crop transforms or offline train augmentation (random resized crop, flip, rotation).
- `[caltech256.py:IMG_EXTENSIONS]`: Supported image file extension whitelist tuple.
- `[hdf5_dataset.py:HDF5LatentDataset]`: PyTorch `Dataset` subclass reading cached float16 ELIC latents from HDF5 in SWMR mode and casting dynamically to float32 PyTorch tensors.
- `[hdf5_dataset.py:HDF5LatentDataset.close]`: Closes the worker-local HDF5 file handle.
- `[__init__.py:find_caltech256_root]`: Re-exported helper from `caltech256.py`.
- `[__init__.py:Caltech256PreparedDataset]`: Re-exported PyTorch `Dataset` class from `caltech256.py`.
- `[__init__.py:HDF5LatentDataset]`: Re-exported PyTorch `Dataset` class from `hdf5_dataset.py`.

## Invariants
- `[caltech256.py:find_caltech256_root]`: Target `search_dir` must exist on filesystem and contain `256_ObjectCategories` or `001.ak47`; raises `FileNotFoundError` if missing.
- `[caltech256.py:build_caltech256_sample_manifest]`: `data_root` must contain at least 256 category subdirectories; raises `ValueError` otherwise.
- `[caltech256.py:build_caltech256_sample_manifest]`: Test samples are strictly clean and unaugmented; train split classes with fewer than `target_k` images are augmented deterministically.
- `[hdf5_dataset.py:HDF5LatentDataset.__init__]`: `h5_path` must be an existing file; raises `FileNotFoundError` if not found.
- `[hdf5_dataset.py:HDF5LatentDataset.__init__]`: `split` parameter must be one of `'train'`, `'test'`, or `'all'`; raises `ValueError` for invalid choices.
- `[hdf5_dataset.py:HDF5LatentDataset.__getitem__]`: Reads float16 latents from datasets `latents_quantized` and `latents_continuous` and casts to float32 PyTorch tensors.

## Side Effects
- `[caltech256.py:find_caltech256_root]`: Traverses local filesystem directory hierarchy via `Path.glob`.
- `[caltech256.py:build_caltech256_sample_manifest]`: Iterates filesystem directories and prints sample counts to stdout.
- `[caltech256.py:Caltech256PreparedDataset.__getitem__]`: Opens and reads image files from disk using `PIL.Image.open`.
- `[hdf5_dataset.py:HDF5LatentDataset.__init__]`: Opens HDF5 archive in read mode to read partition masks and metadata attributes.
- `[hdf5_dataset.py:HDF5LatentDataset._get_handle]`: Lazily opens and maintains a process-local HDF5 file handle in SWMR read mode.
- `[hdf5_dataset.py:HDF5LatentDataset.close]`: Closes active HDF5 file handle descriptor.

## Verification
- Run: `pytest`
