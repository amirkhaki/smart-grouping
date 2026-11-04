# Tasks

## 1. Code and Configuration

- [x] 1.1 Implement `compute_channel_symbol_density_from_latents` and `rank_channels_by_symbol_density` in `benchmarks/sparsity.py` to compute discrete symbol densities $s_c = \mathbb{E}[\frac{1}{HW}\sum \mathbb{I}(\text{round}(y - \mu) \ne 0)]$ from cached continuous latents and frozen ELIC context layers
- [x] 1.2 Add unit tests in `tests/test_sparsity.py` verifying symbol density calculation, bounds in $[0, 1]$, and ranking
- [x] 1.3 Update `benchmarks/train_probe.py` to support `--symbol-density` channel subsets (`SD8`, `SD16`, `SD32`, `SD64`, `SD128`), saving `channel_symbol_density_ranking.csv`
- [x] 1.4 Add `generate_symbol_density_plot` in `benchmarks/evaluate.py`
- [x] 1.5 Create run configuration files: `configs/exp_r004_symbol_smoke.json`, `configs/exp_r005_symbol_signal.json`, `configs/exp_r006_symbol_density_benchmarks.json`

## 2. Runs

- [x] 2.1 R004 Smoke test verifying symbol density extraction pipeline on synthetic latents | config: configs/exp_r004_symbol_smoke.json | tier: 1 | done when: row in research/runs.jsonl
- [x] 2.2 R005 Small-signal verification of non-tied symbol density ranking across 1,000 cached training samples | config: configs/exp_r005_symbol_signal.json | tier: 2 | done when: row in research/runs.jsonl
- [x] 2.3 R006 Full Tier-2 probe evaluation across SD8, SD16, SD32, SD64, SD128 (30 epochs, leak-free validation split) | config: configs/exp_r006_symbol_density_benchmarks.json | tier: 3 | done when: row in research/runs.jsonl

## 3. Close-out

- [x] 3.1 Write analysis.md from logged runs
- [x] 3.2 Update research/findings.md
- [x] 3.3 Write decision.md
