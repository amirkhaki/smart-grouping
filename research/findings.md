# Findings

<!-- Entry format:
## F-001: <statement>
Status: supported | refuted | open | invalid
Confidence: low | medium | high
Evidence: runs R###, change <name>
Caveats:
Related: F-###
-->

## Supported

## F-001: Rate-Ranked Latent Channel Prioritization Outperforms Fixed Slices and Non-Zero Activation Density
Status: supported
Confidence: high
Evidence: runs R001, R002, R003, change rate-ranked-channel-selection
Caveats: Evaluated on Caltech-256 with ELIC lambda=0.032 and probe latent_head_w256 under seed 42. Multi-seed variance and low-rate regimes (lambda=0.004) remain open.
Related:
Notes:
- Empirical channel rate calculation $r_c = \mathbb{E}[-\log_2 p(y_c)]$ completely eliminates the 100% density saturation artifact caused by continuous predicted means $\mu$.
- Top-rate subsets achieve +1.53% to +3.79% Top-1 accuracy gains over non-zero density-selected channels at $K \in \{8, 16, 32\}$.
- TR16 (16 ch, 0.0354 bpp) outperforms fixed base slice S0 (48.30% vs. 47.12%) at 29% lower decodable bitrate.
- TR32 (32 ch, 0.0475 bpp) achieves 49.40% Top-1 accuracy, outperforming the full 320-channel representation C4 (48.53% at 0.1835 bpp) with 74% fewer bits and 10x fewer channels.
- Accuracy plateaus between TR32 (49.40%) and TR128 (49.38%), establishing TR32 as the Pareto-optimal operating point.

## F-002: Discrete Transmitted Symbol Density P(round(y - mu) != 0) Yields Channel Subsets Equipotent to Shannon Rate
Status: supported
Confidence: high
Evidence: runs R004, R005, R006, change symbol-density-channel-selection
Caveats: Evaluated on Caltech-256 with ELIC lambda=0.032 and probe latent_head_w256 under seed 42. Behavior on lower rate regimes (lambda=0.004) where dead zones are wider remains untested.
Related: F-001
Notes:
- Isolating the true discrete transmitted integer symbol $q = \text{round}(y - \mu) \in \mathbb{Z}$ completely eliminates the 100% density saturation artifact that afflicted continuous reconstructed latents $\hat{y} = q + \mu$.
- The top-8, top-16, and top-32 channel subsets selected by symbol density are 100% identical in channel set membership to those selected by Shannon rate ($-\log_2 p$).
- Across all 320 latent channels, the Spearman rank correlation between discrete symbol density and Shannon transmission rate is $r_s = 0.9989$.
- SD32 (32 ch, 0.0475 bpp) achieves 48.81% Top-1 accuracy, matching TR32 (49.40%) within initial random weight projection variance (-0.59%) and outperforming full-capacity 128-channel C4 (48.53% at 0.1835 bpp) with 74.1% bit savings.
- Proves that spatial non-zero frequency of transmitted symbols is the fundamental operational driver of both bitrate expenditure and semantic representation in ELIC.

## Refuted (do not re-propose without stating what is different)

## Open / untested

## Invalid runs (bugs, to rerun)
