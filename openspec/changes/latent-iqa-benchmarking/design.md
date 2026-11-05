# Design

## Mode
exploratory

## Baseline and the single changed variable
- **Baseline**: Low-frequency base slice `S0` [0:16] representation extracted by frozen ELIC model at maximum fidelity ($\lambda = 0.45$), regressed to human perceptual Mean Opinion Score (MOS) via a capacity-normalized probe (`stem_channels=256`, 1-dim scalar output).
- **Single changed variable**: Latent slice view used as input to the IQA regression probe:
  - Isolated slices: `S0` [0:16], `S1` [16:32], `S2` [32:64], `S3` [64:128], `S4` [128:320].
  - Cumulative reference: `C4` [0:320] (full latent representation).
  - Invariant: Same frozen ELIC backbone (`elic_0450.pth.tar`), same training loss (Pearson correlation loss $\mathcal{L}_{\text{PLCC}} = 1 - r$ combined with $L_1$ loss), same optimizer (AdamW, lr=1e-3, weight_decay=1e-4), same data split (70/10/20), and same batch size.

## Modular Code Organization (`benchmarks/iqa/`)
To prevent cluttering the root benchmark files, all IQA-related code is isolated inside a dedicated subpackage `benchmarks/iqa/`:
- `benchmarks/iqa/__init__.py`: Package initialization and exports.
- `benchmarks/iqa/dataset.py`: `iqadataset` integration and 70/10/20 train/val/test splitting.
- `benchmarks/iqa/cache_latents.py`: Feature extraction and HDF5 latent caching pipeline.
- `benchmarks/iqa/models.py`: Capacity-normalized PreAct ResNet IQA regression heads and multi-head container.
- `benchmarks/iqa/metrics.py`: SRCC and 5-parameter logistic curve-fitted PLCC metrics.
- `benchmarks/iqa/train_probe.py`: Concurrent multi-head trainer and evaluator.

## Evaluation (protected)
- Metric: Spearman Rank Order Correlation Coefficient (SRCC) between predicted quality score and human subjective MOS on held-out test split (primary metric, higher is better). Secondary metric: Pearson Linear Correlation Coefficient (PLCC, higher is better) using 5-parameter logistic regression mapping.
- Split / data version: Standard 70/10/20 train/val/test split on KonIQ-10k loaded via the `iqadataset` package (`koniq10k_elic0450_split_70_10_20.npz`):
  - Train (70%): ~7,051 images for optimizing probe weights.
  - Val (10%): ~1,007 images strictly for early stopping and hyperparameter monitoring.
  - Test (20%): ~2,015 images held out exclusively for protected final evaluation.
  - Scores normalized to $[0, 1]$.
- Eval script: `benchmarks.iqa.train_probe` (evaluates multi-head IQA container concurrently across all views, checkpoint selection guided strictly by the 10% validation split, final metric evaluation reported on the 20% test split).
- Oracle or best-case bound: Inter-subject human consensus ceiling on KonIQ-10k ($\text{SRCC} \approx 0.92 - 0.94$); Full latent representation `C4` serves as the empirical capacity ceiling of the ELIC feature space.

## Tiers & CI Architecture
Following the existing classification composite action (`.github/actions/kaggle-run` and `ci/kaggle_runner.py`), the IQA pipeline is split into a 2-Tier Kaggle GPU workflow automated via GitHub Actions:
- **Tier-1 Kernel (`smart-grouping-iqa-tier1-cache`)**:
  - Installs `iqadataset`, loads KonIQ-10k, passes images through frozen `ELIC 0.45`, and extracts deterministic continuous ($y$) and quantized ($\hat{y}$) latents alongside normalized MOS scores and 70/10/20 split indices into an indexed HDF5 archive (`koniq10k_elic0450_latents.h5`) on Kaggle GPU via `python -m benchmarks.iqa.cache_latents`.
- **Tier-2 Kernel (`smart-grouping-iqa-tier2-eval`)**:
  - Attaches the Tier-1 cache kernel output, loads the HDF5 archive in-memory, and trains all regression heads (`S0`..`S4`, `C4`) concurrently via `python -m benchmarks.iqa.train_probe`.
  - Outputs `iqa_master_results.csv`, `iqa_metrics.json`, and SRCC/PLCC correlation plots back to GitHub Actions as workflow artifacts.

### Experiment Tiers
- Tier 1 (runs): Smoke test verifying dataset loading via `iqadataset`, HDF5 latent caching pipeline (`benchmarks.iqa.cache_latents`), regression loss computation, and numerical stability of SRCC/PLCC calculation on synthetic mock latents without NaNs.
- Tier 2 (small signal, no conclusions): 5-epoch training on 1,000 cached KonIQ-10k latents to verify loss convergence, non-collapsed variance, and gradient backpropagation across all isolated slices (`S0` through `S4`).
- Tier 3 (full eval, only tier used in analysis): Full Tier-2 execution training concurrent regression heads across `S0`, `S1`, `S2`, `S3`, `S4`, and `C4` on the full KonIQ-10k 70/10/20 split for 30 epochs with early stopping on validation SRCC, logging final test SRCC to `research/runs.jsonl`.

## Seeds and runtime
- Seeds: Probe initialization seed 42; data split seed 42.
- Rough runtime per run: Tier 1 smoke test: ~1 minute; Tier 2 signal test: ~3 minutes; Tier 3 full multi-head evaluation: ~30-40 minutes on Kaggle 2x Tesla T4 GPUs.
- Stopping rule: If high-frequency slice `S4` fails to outperform base slice `S0` ($\text{SRCC}_{S4} \le \text{SRCC}_{S0}$) in Tier 3 evaluation, reject the Slice Inversion Hypothesis. If all individual slices achieve $\text{SRCC} < 0.50$, halt the isolated slice investigation and evaluate multi-scale intermediate backbone features ($g_a$).

## Confounds and risks
- Crop vs. Full-Image Distortion: KonIQ-10k images are $1024 \times 768$. Resizing introduces interpolation blur. We extract center crops ($512 \times 512$ or $256 \times 256$) to preserve raw camera pixels, ensuring ELIC downsampling ($16\times$) produces unpadded feature grids.
- Mean Prediction Collapse: In regression tasks with noisy subjective scores, mean squared error ($L_2$) can lead probes to predict dataset mean, yielding low variance and poor ranking. Combining Pearson correlation loss ($\mathcal{L}_{\text{PLCC}} = 1 - r$) with smooth $L_1$ loss enforces rank monotonicity and preserves dynamic range.
- Backbone Freezing: The ELIC $\lambda=0.45$ compression model weights remain strictly frozen. No regression gradients backpropagate into the compression backbone.
