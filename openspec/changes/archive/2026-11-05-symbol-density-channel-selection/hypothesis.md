# Hypothesis: True Transmitted Symbol Density Channel Selection

## Claim
Selecting latent channels by true transmitted integer symbol density ($P(q_c \ne 0)$ where $q = \text{round}(y - \mu)$ under ELIC's context prediction) produces non-tied channel subsets that achieve competitive classification accuracy with rate-ranked selection ($-\log_2 p(y_c)$), demonstrating that spatial activation frequency of discrete symbols is a primary driver of semantic utility.

## Motivation and prior evidence
- **Builds on Finding `F-001`**: Rate-ranked channel selection ($-\log_2 p$) proved that dynamic channel subsetting dramatically outperforms handcrafted contiguous slices (`TR32` achieves 49.40% Top-1 at 0.0475 bpp vs. full `C4`'s 48.53% at 0.1835 bpp).
- **Overcoming the Density Saturation Artifact**: In earlier explorations, non-zero activation density measured on reconstructed latents $\hat{y} = \text{round}(y - \mu) + \mu$ saturated at 100% across >128 channels because the continuous predicted mean $\mu$ is floating-point and almost never zero.
- **Isolating the Transmitted Discrete Symbol $q$**: In ELIC (He et al., 2022, CVPR), the discrete symbol actually transmitted over the channel is $q = \text{round}(y - \mu) \in \mathbb{Z}$. When $|y - \mu| < 0.5$, $q = 0$, representing an entropy dead zone where zero information is sent.
- **Scientific Value**: Comparing $P(q \ne 0)$ against $-\log_2 p(y)$ disentangles two competing hypotheses for compression-for-machines:
  1. *Frequency Hypothesis*: Channels that fire non-zero symbols across many spatial coordinates encode category semantics.
  2. *Bit-Precision Hypothesis*: Channels with high likelihood cost (large $|q|$ or wide scale $\sigma$) encode category semantics, even if spatially sparse.

## Falsification criterion
Subsets selected by true symbol density ($SD_K$ for $K \in \{8, 16, 32, 64, 128\}$) fail to achieve a continuous non-tied ranking, or underperform rate-ranked subsets ($TR_K$) by more than 1.0% Top-1 accuracy on the Caltech-256 test split under identical probe training conditions.

## Expected effect
- Discrete symbol density $P(q \ne 0)$ will produce a strictly continuous ranking across all 320 channels with zero saturation ties.
- Symbol-density subsets ($SD_K$) will achieve within $\pm 0.5\%$ Top-1 accuracy of rate-ranked subsets ($TR_K$) at $K \in \{16, 32\}$, and substantially outperform naive reconstructed density ($LS_K$) by +2.0% to +3.5%.

## Next step by outcome
- **Supported**: If $SD_K \approx TR_K$, conclude that spatial activation frequency of discrete symbols captures the essential semantic content of ELIC latents without requiring scale parameter $\sigma$ integration; pursue hybrid rate-frequency channel grouping.
- **Refuted**: If $SD_K$ significantly underperforms $TR_K$ ($>1.0\%$ gap), conclude that bit magnitude and scale precision (which $-\log_2 p$ measures) are indispensable for separating semantic channels from low-magnitude background noise.
- **Inconclusive**: Expand sample size for symbol density estimation from 2,000 to 5,000 samples, or evaluate on lower rate regimes ($\lambda=0.004$).

## Related and already tried
- **`F-001` (Rate Ranking $-\log_2 p$)**: Supported. Achieved 49.40% Top-1 on TR32.
- **Heuristic Non-Zero Activation Density (`LS`)**: Refuted for ELIC centered latents due to $\hat{y} = q + \mu$ mean restoration causing 100% density saturation ties.
