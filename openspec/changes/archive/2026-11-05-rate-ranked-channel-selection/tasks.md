# Tasks

## 1. Code and Configuration

- [x] 1.1 Implement `compute_channel_rates_from_latents` in `benchmarks/sparsity.py` to compute per-channel bitrates $r_c = \mathbb{E}[-\log_2 p(y_c)]$ from cached continuous latents and frozen ELIC entropy layers
- [x] 1.2 Add unit and shape tests in `tests/test_sparsity.py` verifying rate computation, finite bounds, and sorting
- [x] 1.3 Update `benchmarks/train_probe.py` to support `--top-rate` channel subsets alongside `--least-sparse`, saving `channel_rate_ranking.csv`
- [x] 1.4 Create run configuration files: `configs/exp_r001_smoke.json`, `configs/exp_r002_signal.json`, `configs/exp_r003_top_rate_benchmarks.json`

## 2. Runs

- [x] 2.1 R001 Smoke test on synthetic latents verifying rate extraction pipeline | config: configs/exp_r001_smoke.json | tier: 1 | done when: row in research/runs.jsonl
- [x] 2.2 R002 Signal verification of channel rate monotonicity on 1,000 cached training samples | config: configs/exp_r002_signal.json | tier: 2 | done when: row in research/runs.jsonl
- [x] 2.3 R003 Full probe evaluation across TR8, TR16, TR32, TR64, TR128 (30 epochs, leak-free validation split) | config: configs/exp_r003_top_rate_benchmarks.json | tier: 3 | done when: row in research/runs.jsonl

## 3. Close-out

- [x] 3.1 Write analysis.md from logged runs
- [x] 3.2 Update research/findings.md
- [x] 3.3 Write decision.md
