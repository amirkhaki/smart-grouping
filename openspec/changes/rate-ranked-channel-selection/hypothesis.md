# Hypothesis: Rate-Ranked Latent Channel Selection

## Claim
Selecting latent channels by empirical per-channel transmission bitrate ($-\log_2 p(y_c)$ under ELIC's Gaussian conditional) yields subsets that achieve equal or higher Caltech-256 Top-1 classification accuracy at lower total decodable bitrates than both handcrafted slice unions ($C_1..C_4$) and heuristic non-zero activation density rankings (`LS`).

## Motivation and prior evidence
In ELIC (He et al., 2022, CVPR), the entropy coding mechanism quantizes latents centered on contextually predicted means: $\hat{y} = \text{round}(y - \mu) + \mu$.
Prior empirical results on Caltech-256 using non-zero activation density ($\mathbb{I}(\hat{y} \ne 0)$) revealed a critical measurement artifact: over 128 channels are tied at 100.00% density because $\mu$ is continuous and non-zero even when the transmitted integer symbol is 0.
Despite this tie, subset selection demonstrated strong potential: `LS128` (49.29% Top-1 at 0.1025 bpp) outperformed full 320-channel `C4` (48.53% Top-1 at 0.1835 bpp) with a lower generalization gap.
Per-channel bitrate $r_c = \mathbb{E}[-\frac{1}{HW}\sum_{h,w} \log_2 p(y_{c,h,w})]$ directly quantifies the entropy allocated to channel $c$ by the compression model without thresholding artifacts or ties.

## Falsification criterion
Probes trained on rate-ranked channel subsets ($TR_K$ for $K \in \{8, 16, 32, 64, 128\}$) fail to achieve a strictly superior Rate-Accuracy Pareto curve compared to handcrafted cumulative slices ($S_0, C_1, C_2, C_3$) and density-ranked subsets ($LS8..LS128$) evaluated on the leak-free validation and test splits.

## Expected effect
Rate-ranked subsets will resolve the 100% density tie across channels, isolate semantic vs. textural bit allocation, and improve Top-1 accuracy by +0.5% to +1.5% at low to medium bitrates ($K \le 64$) relative to density-selected channels.

## Next step by outcome
- Supported: Adopt rate-based channel ranking as the canonical dynamic channel prioritization method in the benchmark suite; evaluate on additional rate regimes ($\lambda=0.004$).
- Refuted: If rate-ranked channels underperform density or handcrafted slices, investigate whether high-bitrate channels primarily encode high-frequency perceptual texture residuals rather than category-discriminative semantic features.
- Inconclusive: Expand sample subset size for channel bitrate estimation or evaluate across multiple probe seeds.

## Related and already tried
- Activation density ranking (`benchmarks/sparsity.py`, commit `cd62b72`): Proved subset selection is viable, but suffers from saturated 100% density ties in $\hat{y}$ due to $\mu$ addition.
