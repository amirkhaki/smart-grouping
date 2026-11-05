# Tasks

## 1. Code and Configuration

- [x] 1.1 Extend `benchmarks/iqa/models.py:MultiHeadIQARegressionContainer` to support arbitrary channel subsets (e.g. `custom_subsets={"SD8": [...], ...}`) alongside contiguous slice views (`S0`, `C1`, `C2`, `C3`, `C4`)
- [x] 1.2 Update `benchmarks/iqa/train_probe.py` to support `--symbol-density` channel ranking from cached continuous latents and pass custom subsets to the container
- [x] 1.3 Create run configuration files: `configs/exp_r010_iqa_cumulative_smoke.json`, `configs/exp_r011_iqa_symbol_signal.json`, `configs/exp_r012_iqa_cumulative_density_benchmarks.json`
- [x] 1.4 Update `.github/workflows/kaggle-iqa-tier2.yml` default command to evaluate cumulative (`S0`, `C1`, `C2`, `C3`, `C4`) and symbol-density (`SD8`, `SD16`, `SD32`, `SD64`, `SD128`) views

## 2. Runs

- [ ] 2.1 R010 Smoke test verifying multi-head probe training with arbitrary channel subsets and cumulative slices on synthetic latents | config: configs/exp_r010_iqa_cumulative_smoke.json | tier: 1 | done when: row in research/runs.jsonl
- [ ] 2.2 R011 Empirical computation of transmitted symbol density $s_c = P(\text{round}(y-\mu) \ne 0)$ across 2,000 KonIQ-10k latents and ranking all 320 channels | config: configs/exp_r011_iqa_symbol_signal.json | tier: 2 | done when: row in research/runs.jsonl
- [ ] 2.3 R012 Full benchmark evaluation of cumulative views (S0, C1, C2, C3, C4) and symbol-density views (SD8, SD16, SD32, SD64, SD128) across 30 epochs on Kaggle 2x Tesla T4 GPUs | config: configs/exp_r012_iqa_cumulative_density_benchmarks.json | tier: 3 | done when: row in research/runs.jsonl

## 3. Close-out

- [ ] 3.1 Write analysis.md from logged runs
- [ ] 3.2 Update research/findings.md
- [ ] 3.3 Write decision.md
