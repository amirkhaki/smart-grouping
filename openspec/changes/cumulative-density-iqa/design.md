# Design: Cumulative and Symbol-Density Latent Subsets for No-Reference Image Quality Assessment

## Mode
Exploratory

## Baseline and the single changed variable
- **Baseline**: Isolated base slice `S0` (`[0:16]`, 16 ch, Test SRCC = 0.7314) and full latent `C4` (`[0:320]`, 320 ch, Test SRCC = 0.7289) from Run `R009`.
- **Single changed variable**: Latent channel subset selection scheme, evaluating two paradigms concurrently under identical capacity-matched regression probe architectures (`w=256`, joint Pearson+$L_1$ loss, 30 epochs):
  1. **Cumulative contiguous slices**: `S0` (`[0:16]`, 16 ch), `C1` (`[0:32]`, 32 ch), `C2` (`[0:64]`, 64 ch), `C3` (`[0:128]`, 128 ch), `C4` (`[0:320]`, 320 ch).
  2. **Data-driven symbol-density channels**: `SD8` (8 ch), `SD16` (16 ch), `SD32` (32 ch), `SD64` (64 ch), `SD128` (128 ch), ranked by empirical transmitted discrete symbol density $s_c = P(\text{round}(y_c - \mu_c) \ne 0)$ computed over KonIQ-10k training latents via frozen ELIC $\lambda=0.45$.

## Evaluation (protected)
- **Metric**: Spearman Rank Correlation Coefficient (SRCC) and 5-parameter non-linear logistic curve-fitted Pearson Correlation (PLCC).
- **Split / data version**: Pinned deterministic 70/10/20 train/val/test split ($N = 10,073$, train = 7,051, val = 1,007, test = 2,015) in `koniq10k_elic0450_latents.h5` cached in Kaggle kernel `smart-grouping-iqa-tier1-cache`.
- **Eval script**: `benchmarks/iqa/train_probe.py` using `benchmarks/iqa/metrics.py:compute_iqa_metrics`.
- **Oracle / best-case bound**: Fully supervised end-to-end NR-IQA models (e.g. HyperIQA, MUSIQ) achieve $\text{SRCC} \approx 0.90$–$0.92$ on KonIQ-10k. Lightweight linear/MLP probes over frozen self-supervised latents typically achieve $\text{SRCC} \approx 0.72$–$0.78$.

## Tiers
- **Tier 1 (Smoke test, Run R010)**: Fast verification of multi-head container supporting arbitrary channel subsets alongside contiguous slices on mock synthetic latents.
- **Tier 2 (Signal test, Run R011)**: Empirical computation and ranking of all 320 latent channels by transmitted symbol density $s_c$ across 2,000 KonIQ-10k training latents, validating non-zero variance and index extraction.
- **Tier 3 (Full benchmark, Run R012)**: Parallel multi-head probe training across all 10 views (`S0`, `C1`, `C2`, `C3`, `C4`, `SD8`, `SD16`, `SD32`, `SD64`, `SD128`) for 30 epochs on Kaggle 2x Tesla T4 GPUs with early stopping on validation SRCC.

## Seeds and runtime
- **Seed**: 42 (pinned across all heads and data loaders).
- **Rough runtime**: ~20 minutes on Kaggle 2x Tesla T4 GPUs attaching `smart-grouping-iqa-tier1-cache`.
- **Stopping rule**: Terminate after Tier-3 evaluation. If `C1`..`C4` degrade relative to `S0`, and `SD16`/`SD32` fail to beat `S0`/`C1`, the hypothesis is refuted.

## Confounds and risks
- **Channel Correlation**: Top symbol-density channels might overlap heavily with early slices `S0` and `S1`. We will record channel set intersection (Jaccard similarity) between `SD16` vs `S0`, and `SD32` vs `C1`.
- **Multi-Head Interference**: All 10 heads share the input data loader pass, but each head maintains completely separate parameters (`LatentIQARegressor`) with independent gradients and early stopping counters to prevent cross-head parameter interference.
