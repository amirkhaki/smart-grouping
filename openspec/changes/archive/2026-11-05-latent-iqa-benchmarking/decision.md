# Decision: Latent Slice Inversion for No-Reference Image Quality Assessment

## Verdict: Refuted (Hypothesis Rejected)

The empirical evaluation conducted in runs `R007`, `R008`, and `R009` across 10,073 KonIQ-10k images on Kaggle 2x Tesla T4 GPUs decisively refutes the **Slice Inversion Hypothesis**:
- **Falsification threshold**: $\text{SRCC}_{S4} - \text{SRCC}_{S0} \ge +0.10$
- **Empirical result**: $\text{SRCC}_{S4} - \text{SRCC}_{S0} = 0.6566 - 0.7314 = \mathbf{-0.0748}$

Base slice `S0` (16 channels) strictly outperforms high-frequency slice `S4` (192 channels) and even edges out the complete 320-channel latent representation `C4` ($\text{SRCC}_{S0} = 0.7314$ vs. $\text{SRCC}_{C4} = 0.7289$).

---

## Action Taken

1. **Reject Slice Inversion Architecture**: We will **not** design downstream IQA probes, transmission protocols, or feature selectors that discard base slices in favor of high-frequency residual slices (`S3`, `S4`).
2. **Establish Universal Base Dominance**: Findings `F-001`, `F-002`, and `F-003` together establish that in ELIC, early slices (`S0`–`S2`) constitute the universal informational backbone:
   - For **semantic classification**: 32 channels (`TR32` / `SD32`) achieve 49.40% Top-1 (beating full latent `C4` at 48.53%).
   - For **perceptual quality assessment**: 16 channels (`S0`) achieve 0.7314 SRCC (beating full latent `C4` at 0.7289).
3. **Archive Change**: Archive `latent-iqa-benchmarking` as a completed negative result with high scientific confidence, documenting Finding `F-003` in `research/findings.md`.

---

## Next Steps

1. **Ultra-Compact Latent IQA Head**: Because 16-channel `S0` surpasses the full 320-channel representation, future work can construct lightweight edge IQA assessors operating strictly on the first 16 decoded channels, saving 95% of communication bandwidth and memory footprint.
2. **Unsupervised Quality Probing**: Investigate whether negative log-likelihood ($-\log_2 p(y_0)$) of the autoregressive context model on base slice `S0` predicts quality degradation without any supervised probe training.
3. **Multi-Scale Spatial Aggregation**: Test standard deviation and minimum pooling across the $16 \times 16$ spatial latent grid of `S0` to capture localized artifacts (e.g. out-of-focus background vs. sharp foreground).
