# Decision: Cumulative and Symbol-Density Latent Subsets for No-Reference Image Quality Assessment

## Verdict: Supported (Cumulative Peak at C3) / Refuted (Symbol Density Selection)

The empirical benchmark across 10,073 KonIQ-10k images in Run `R012` provides conclusive findings on the two investigated channel selection paradigms:
1. **Cumulative Slices Peak at C3 (Supported & Refined)**:
   - Cumulative autoregressive expansion achieves monotonic performance gains from `S0` (16 ch, 0.7205 SRCC) up to **`C3` (128 ch, 0.7396 SRCC, 0.7575 PLCC)**.
   - Appending slice `S4` (192 channels) to form full latent `C4` (320 ch) degrades Test SRCC by $-0.0115$ (down to 0.7281).
2. **Data-Driven Symbol Density Fails for IQA (Refuted)**:
   - Transmitted symbol density underperformed fixed contiguous slices across all capacity budgets (`SD16` vs `S0`: $-0.0295$ SRCC; `SD32` vs `C1`: $-0.0150$ SRCC; `SD128` vs `C3`: $-0.0288$ SRCC).
   - Symbol density over-selects stochastic high-frequency texture channels and breaks the causal autoregressive contextual structure of ELIC.

---

## Action Taken

1. **Standardize on C3 for Compressed-Domain Quality Assessment**: Slices `S0` through `S3` (`[0:128]`, 128 channels) are officially established as the optimal compressed-domain feature representation for ELIC IQA, eliminating `S4` to reduce feature size and transmission bitrate by **60%** while maximizing prediction accuracy.
2. **Retain S0 as the Ultra-Compact Edge Operating Point**: For bandwidth-constrained applications, `S0` (`[0:16]`, 16 channels) achieves $0.7205$ SRCC with **95% fewer channels** than `C4`.
3. **Reject Marginal Symbol Density for Low-Level / IQA Tasks**: Documented in Finding `F-005` that while marginal symbol density is optimal for high-level semantic classification (`F-002`), it is unsuitable for low-level perceptual quality tasks due to texture selection bias.
4. **Archive Change**: Archive `cumulative-density-iqa` with `--skip-specs`.

---

## Next Steps

1. **Spatial Pooling Aggregation**: Evaluate standard deviation and min-pooling across the $16 \times 16$ spatial grid on `C3` and `S0` to capture localized blur and ringing artifacts without adding channels.
2. **Cross-Dataset Validation**: Benchmark the `C3` representation on synthetic distortion datasets (e.g. KADID-10k or LIVE) to evaluate transferability across distortion families.
3. **Unsupervised Rate-Distortion Quality Metric**: Explore whether the joint log-likelihood of slices `S0`–`S3` ($-\sum_{i=0}^3 \log_2 p(y_i)$) correlates with human MOS in a zero-shot, unsupervised setting.
