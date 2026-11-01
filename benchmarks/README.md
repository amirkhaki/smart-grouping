# Benchmarks

## Overview
Benchmark execution suite for evaluating representations learned by ELIC image compression models on Caltech-256 downstream classification, including latent caching to HDF5, multi-head probe training, channel sparsity profiling, and Rate-Accuracy Pareto visualization.

## Interface
- [datasets](./datasets/README.md): Child component package managing Caltech-256 data loading, sample manifest generation, and HDF5 dataset access.
- [models](./models/README.md): Child component package defining probe classifier architectures and multi-head container.
- `[config.py:SliceSpec]`: Frozen dataclass specifying channel slice start/end bounds, channel counts, and category (isolated or cumulative).
- `[config.py:SLICE_SPECS]`: Dictionary mapping 9 canonical slice views (`S0`-`S4`, `C1`-`C4`) to `SliceSpec` configurations.
- `[config.py:CHECKPOINT_REGISTRY]`: Pretrained ELIC model registry mapping rate lambda values to Google Drive file IDs and paths.
- `[download_checkpoints.py:download_checkpoint]`: Downloads pretrained ELIC model checkpoints from Google Drive via `gdown` with exponential backoff.
- `[download_checkpoints.py:main]`: CLI entrypoint for downloading single or all registered ELIC checkpoints.
- `[elic_extractor.py:ELICFeatureExtractor]`: Non-invasive wrapper module for frozen `TestModel` extracting continuous and context-quantized latents alongside exact per-slice bitrates.
- `[elic_extractor.py:ELICFeatureExtractor.from_checkpoint]`: Factory loading `TestModel` weights from checkpoint file and initializing frozen extractor.
- `[cache_latents.py:CustomDataParallel]`: DataParallel wrapper providing attribute pass-through to inner extractor module.
- `[cache_latents.py:resolve_checkpoint]`: Resolves checkpoint path and lambda value from registry or filesystem path.
- `[cache_latents.py:main]`: Tier-1 CLI pipeline caching continuous/quantized latents, labels, split masks, and bitrates into chunked HDF5.
- `[sparsity.py:get_channel_slice]`: Maps channel index (`0..319`) to originating ELIC slice view name (`S0`-`S4`).
- `[sparsity.py:get_channel_slice_info]`: Returns slice name and relative channel index within that slice.
- `[sparsity.py:compute_channel_densities]`: Computes empirical non-zero activation density across 320 latent channels on quantized latents.
- `[sparsity.py:rank_channels_by_density]`: Sorts channels from highest activation density (least sparse) to lowest.
- `[sparsity.py:format_density_summary]`: Generates formatted terminal summary table of channel density rankings.
- `[train_probe.py:get_autocast]`: Factory providing version-agnostic AMP autocast context manager for PyTorch 1.x and 2.x.
- `[train_probe.py:get_grad_scaler]`: Factory providing version-agnostic AMP GradScaler.
- `[train_probe.py:find_h5_cache]`: Searches filesystem and Kaggle input directories for `caltech256_elic_latents.h5`.
- `[train_probe.py:calculate_slice_bitrates]`: Computes empirical mean decodable bitrates for isolated and cumulative slice views from test set.
- `[train_probe.py:evaluate_active_heads]`: Runs validation evaluation measuring Top-1 accuracy for active heads to guide early stopping.
- `[train_probe.py:main]`: Tier-2 CLI pipeline conducting joint training of probe heads with per-head early stopping, metrics logging, and plot generation.
- `[evaluate.py:export_master_results]`: Exports benchmark configuration records to CSV and JSON files.
- `[evaluate.py:generate_pareto_plot]`: Generates Rate-Accuracy Pareto curve visualization comparing probe architectures across cumulative slice views.
- `[evaluate.py:generate_stem_ablation_plot]`: Generates stem channel width ablation curve plotting stem width vs Top-1 accuracy.
- `[evaluate.py:generate_least_sparse_plot]`: Plots Top-1 accuracy across dynamically selected least-sparse channel subsets.

## Invariants
- `[elic_extractor.py:ELICFeatureExtractor.forward]`: Input images must have 3 RGB channels with spatial dimensions divisible by 64 (standard Caltech-256 crop is 256x256).
- `[elic_extractor.py:ELICFeatureExtractor]`: Encapsulated `TestModel` parameters are frozen (`requires_grad=False`) and set to evaluation mode.
- `[cache_latents.py:main]`: HDF5 archive is formatted with chunk shape `(32, 320, 16, 16)` and `float16` precision for latent datasets.
- `[train_probe.py:calculate_slice_bitrates]`: Test bitrates array must have shape `(N, 6)` corresponding to `[R(z), R(y0), R(y1), R(y2), R(y3), R(y4)]`.
- `[sparsity.py:compute_channel_densities]`: Target HDF5 archive must contain `latents_quantized` dataset of shape `(N, 320, 16, 16)`; raises `KeyError` if missing.
- `[evaluate.py:generate_pareto_plot]`: Filtered dataframe must contain cumulative category slice records.

## Side Effects
- `[download_checkpoints.py:download_checkpoint]`: Creates output directory and downloads checkpoint files from Google Drive via network.
- `[cache_latents.py:main]`: Writes large multi-gigabyte HDF5 cache archive (`caltech256_elic_latents.h5`) to disk.
- `[train_probe.py:main]`: Trains neural network probes, writes `channel_sparsity_ranking.csv`/`json`, `master_results.csv`, `metrics.json`, and saves PNG plots.
- `[evaluate.py:export_master_results]`: Writes CSV and JSON evaluation metric artifacts to filesystem.
- `[evaluate.py:generate_pareto_plot]`: Saves Pareto curve image to filesystem.
- `[evaluate.py:generate_stem_ablation_plot]`: Saves stem width ablation plot image to filesystem.
- `[evaluate.py:generate_least_sparse_plot]`: Saves least-sparse channel accuracy plot image to filesystem.

## Verification
- Run: `pytest`
