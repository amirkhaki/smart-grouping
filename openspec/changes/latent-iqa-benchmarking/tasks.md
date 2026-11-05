# Tasks

## 1. Code and Configuration

- [x] 1.1 Create `benchmarks/iqa/` subpackage with `__init__.py` and `benchmarks/iqa/dataset.py` integrating `iqadataset` to load KonIQ-10k with normalized MOS scores $[0, 1]$ and deterministic 70/10/20 train/val/test splitting
- [x] 1.2 Implement `benchmarks/iqa/metrics.py` computing Spearman Rank Correlation (SRCC) and 5-parameter logistic curve-fitted Pearson Correlation (PLCC) following the proven implementation in `idfiqa`
- [x] 1.3 Implement `benchmarks/iqa/cache_latents.py` to extract and cache continuous ($y$) and quantized ($\hat{y}$) latents alongside MOS and 70/10/20 split masks into an indexed HDF5 archive using frozen ELIC $\lambda=0.45$
- [x] 1.4 Implement `benchmarks/iqa/models.py` defining `LatentIQARegressor` and `MultiHeadIQARegressionContainer` with Pearson correlation loss ($\mathcal{L}_{\text{PLCC}} = 1 - r$) and $L_1$ loss
- [x] 1.5 Implement `benchmarks/iqa/train_probe.py` supporting parallel multi-head training across isolated slices (`S0`..`S4`) and full latent `C4`, with early stopping on the 10% validation split and final reporting on the 20% test split
- [x] 1.6 Create GitHub Actions workflows `.github/workflows/kaggle-iqa-tier1.yml` and `.github/workflows/kaggle-iqa-tier2.yml` following the existing `kaggle-run` composite action architecture
- [x] 1.7 Create run configuration files: `configs/exp_r007_iqa_smoke.json`, `configs/exp_r008_iqa_signal.json`, `configs/exp_r009_iqa_benchmarks.json`

## 2. Runs

- [x] 2.1 R007 Smoke test verifying IQA latent caching via `iqadataset`, regression loss, and SRCC/PLCC metric calculation on synthetic latents | config: configs/exp_r007_iqa_smoke.json | tier: 1 | done when: row in research/runs.jsonl
- [ ] 2.2 R008 Small-signal verification of IQA regression convergence across 1,000 cached KonIQ-10k samples on 70/10/20 split | config: configs/exp_r008_iqa_signal.json | tier: 2 | done when: row in research/runs.jsonl
- [ ] 2.3 R009 Full multi-head IQA probe evaluation across S0, S1, S2, S3, S4, and C4 on KonIQ-10k for 30 epochs on Kaggle 2x Tesla T4 GPUs | config: configs/exp_r009_iqa_benchmarks.json | tier: 3 | done when: row in research/runs.jsonl

## 3. Close-out

- [ ] 3.1 Write analysis.md from logged runs
- [ ] 3.2 Update research/findings.md
- [ ] 3.3 Write decision.md
