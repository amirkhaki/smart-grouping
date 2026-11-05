# Tasks

## 1. Code and Configuration

- [x] 1.1 Update `benchmarks/iqa/dataset.py` to support `--no-crop` / native full-frame loading ($1024 \times 768 \rightarrow \text{ToTensor}$ without spatial cropping)
- [x] 1.2 Update `benchmarks/iqa/cache_latents.py` to support full-frame spatial dimensions ($48 \times 64$), batch size 16, and HDF5 `compression="lzf"` chunked storage
- [x] 1.3 Implement `SoftPairwiseLogisticLoss` with label-noise thresholding ($\tau = 0.03$), anchor MSE ($\lambda = 0.1$), spatial statistics pooling (mean, std, max), and raw logit output in `benchmarks/iqa/models.py`
- [x] 1.4 Update `benchmarks/iqa/train_probe.py` to support independent per-head AdamW optimizers, per-head gradient norm clipping (1.0), 5-seed training loop, test prediction saving, and paired bootstrap statistical testing
- [x] 1.5 Create run configuration files: `configs/exp_r013_fullframe_smoke.json`, `configs/exp_r014_fullframe_cache.json`, `configs/exp_r015_fullframe_pairwise_benchmarks.json`
- [x] 1.6 Create/update GitHub Actions workflows `.github/workflows/kaggle-iqa-tier1.yml` and `.github/workflows/kaggle-iqa-tier2.yml` for full-frame caching and 5-seed paired bootstrap evaluation

## 2. Runs

- [ ] 2.1 R013 Smoke test verifying full-frame extraction ($48 \times 64$), LZF compression, independent optimizers, and soft pairwise loss on mock latents | config: configs/exp_r013_fullframe_smoke.json | tier: 1 | done when: row in research/runs.jsonl
- [ ] 2.2 R014 Full-frame latent extraction and LZF HDF5 caching of all 10,073 KonIQ-10k images on Kaggle 2x Tesla T4 GPUs | config: configs/exp_r014_fullframe_cache.json | tier: 2 | done when: row in research/runs.jsonl
- [ ] 2.3 R015 5-seed benchmark evaluation across S0, C1, C2, C3, C4 with independent optimizers, soft pairwise ranking loss, and paired bootstrap testing | config: configs/exp_r015_fullframe_pairwise_benchmarks.json | tier: 3 | done when: row in research/runs.jsonl

## 3. Close-out

- [ ] 3.1 Write analysis.md from logged runs
- [ ] 3.2 Update research/findings.md
- [ ] 3.3 Write decision.md
