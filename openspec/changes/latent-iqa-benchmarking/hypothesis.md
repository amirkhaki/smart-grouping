# Hypothesis: Latent Slice Inversion for No-Reference Image Quality Assessment

## Claim
In a frozen maximum-fidelity ELIC compression model ($\lambda = 0.45$), high-frequency latent slices (`S3` [64:128] and `S4` [128:320]) yield significantly higher Spearman Rank Correlation Coefficient (SRCC $\ge +0.10$) with human perceptual quality scores (MOS) than low-frequency semantic base slices (`S0` [0:16] and `S1` [16:32]), inverting the channel hierarchy observed in semantic classification.

## Motivation and prior evidence
- In findings `F-001` and `F-002`, downstream semantic classification on Caltech-256 established that object semantics are concentrated almost entirely in the first 32 channels (`S0` through `S2`), where `TR32` and `SD32` matched or outperformed the full 320-channel representation `C4` (48.81%–49.40% vs. 48.53% Top-1). Slices `S3` and `S4` (256 combined channels) provided zero marginal semantic gain.
- In image quality assessment (IQA), human perceptual degradation (blur, sensor noise, ringing, blocking, and texture smearing) primarily perturbs high spatial frequencies and residual textures rather than high-level category identity.
- ELIC (`he2022elic`) uses uneven space-channel contextual grouping where early slices (`S0`, `S1`) encode low-frequency structure and hyperprior anchor contexts, while later slices (`S3`, `S4`) are allocated strictly to high-frequency textural residuals entering the synthesis transform $g_s$.
- At maximum fidelity ($\lambda = 0.45$), quantization steps are narrow and dead-zone collapse is avoided, allowing `S3` and `S4` to act as high-precision distortion sensors.

## Falsification criterion
A capacity-matched regression probe trained on isolated high-frequency slice `S4` fails to exceed the Spearman rank correlation of base slice `S0` by at least 0.10 ($\text{SRCC}_{S4} - \text{SRCC}_{S0} < 0.10$), or base slices outperform high-frequency slices ($\text{SRCC}_{S0} \ge \text{SRCC}_{S4}$) on an evaluation benchmark (e.g., KonIQ-10k or Kadid-10k).

## Expected effect
`S4` (192 channels) and `S3` (64 channels) will achieve $\text{SRCC} \ge 0.75$, whereas `S0` (16 channels) will achieve $\text{SRCC} \le 0.60$, demonstrating an advantage of $\Delta \text{SRCC} \ge +0.15$ for high-frequency slices.

## Next step by outcome
- **Supported**: Design an asymmetric IQA probe head that prioritizes high-frequency slices (`S3`, `S4`) and test whether zero-shot entropy likelihoods ($-\log_2 p$) provide unsupervised quality indicators.
- **Refuted**: If `S0` and `S1` equal or outperform `S4`, it indicates that global semantic structural degradation dominates human quality ratings over high-frequency residuals, requiring a shift toward semantic-fidelity IQA.
- **Inconclusive**: If neither slice achieves $\text{SRCC} \ge 0.50$ or cross-fold variance exceeds 0.08, ablate probe capacity (e.g. multi-scale pooling vs. spatial regression) and patch sampling strategy.

## Related and already tried
- `F-001`: Rate-ranked channel selection proved `S4` was redundant for semantic classification at $\lambda=0.032$.
- `F-002`: Discrete symbol density confirmed that spatial symbol activation frequency mirrors rate allocation.
- $\lambda=0.004$ low-rate exploration: Confirmed that low rate causes severe dead-zone collapse in `S4` (0.0065 bpp, 42% channels dead), validating that $\lambda=0.45$ is necessary to preserve distortion signal.
