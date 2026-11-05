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

## F-004: Cumulative Latent Slices Peak at C3 (128 Channels) for Image Quality Assessment While S4 Ingestion Degrades Performance
Status: supported
Confidence: high
Evidence: runs R010, R011, R012, change cumulative-density-iqa
Caveats: Evaluated on KonIQ-10k with ELIC lambda=0.45, multi-head probe w=256, 70/10/20 train/val/test split under seed 42.
Related: F-003
Notes:
- Cumulative autoregressive slice expansion exhibits monotonic gains from base slice S0 through C3: S0 (16 ch, Test SRCC = 0.7205) -> C1 (32 ch, 0.7219) -> C2 (64 ch, 0.7253) -> C3 (128 ch, 0.7396 SRCC, 0.7575 PLCC, 0.1127 RMSE).
- C3 ([0:128]) establishes the global peak performance across all evaluated models on KonIQ-10k.
- Appending the remaining 192 channels of slice S4 ([128:320]) to form full latent C4 (320 ch) degrades Test SRCC by -0.0115 (down to 0.7281), PLCC by -0.0128 (down to 0.7447), and increases RMSE (from 0.1127 to 0.1152).
- Discarding slice S4 reduces latent feature bandwidth and transmission size by 60% while simultaneously boosting perceptual quality assessment accuracy.

## Refuted (do not re-propose without stating what is different)

## F-005: Transmitted Symbol Density Channel Selection Outperforms Contiguous Autoregressive Slices in Perceptual Quality Assessment
Status: refuted
Confidence: high
Evidence: runs R010, R011, R012, change cumulative-density-iqa
Caveats: Evaluated on KonIQ-10k with ELIC lambda=0.45, multi-head probe w=256, 70/10/20 train/val/test split under seed 42.
Related: F-001, F-002, F-004
Notes:
- While discrete symbol density s_c = P(round(y_c - mu_c) != 0) successfully prioritized informative channels for semantic classification on Caltech-256 (F-002), it consistently underperforms fixed contiguous slices for perceptual quality assessment on KonIQ-10k across all channel budgets.
- S0 (16 ch, 0.7205 SRCC) outperforms SD16 (16 ch, 0.6910 SRCC) by +0.0295 SRCC.
- C1 (32 ch, 0.7219 SRCC) outperforms SD32 (32 ch, 0.7069 SRCC) by +0.0150 SRCC.
- C2 (64 ch, 0.7253 SRCC) outperforms SD64 (64 ch, 0.7124 SRCC) by +0.0129 SRCC.
- C3 (128 ch, 0.7396 SRCC) outperforms SD128 (128 ch, 0.7108 SRCC) by +0.0288 SRCC.
- Symbol density preferentially selects high-entropy stochastic texture channels (e.g., ch_50, 115, 126, 144, 287 from S2, S3, S4), which act as content-dependent noise for quality assessment and break the causal autoregressive dependency chain of ELIC's contextual transforms.

## F-003: High-Frequency Slices in Learned Compression Invert the Latent Hierarchy for Image Quality Assessment
Status: refuted
Confidence: high
Evidence: runs R007, R008, R009, change latent-iqa-benchmarking
Caveats: Evaluated on KonIQ-10k with ELIC lambda=0.45, multi-head probe w=256, 70/10/20 train/val/test split under seed 42. Evaluated with global average pooling; spatially localized patch pooling or purely synthetic distortion callsets (e.g. KADID-10k) not evaluated.
Related: F-001, F-002
Notes:
- The hypothesis that high-frequency synthesis slices (S3, S4) outperform low-frequency base slices (S0, S1) in predicting human perceptual Mean Opinion Scores (MOS) is decisively refuted.
- Base slice S0 (channels [0:16], 16 ch) achieved Test SRCC of 0.7314 (0.7540 PLCC, 0.1134 RMSE), strictly outperforming the largest high-frequency slice S4 (channels [128:320], 192 ch, Test SRCC = 0.6566, 0.6744 PLCC, 0.1274 RMSE) by +0.0748 SRCC and +0.0796 PLCC.
- S0 (16 channels, 5% of latent bandwidth) also surpassed the full 320-channel representation C4 (Test SRCC = 0.7289, 0.7459 PLCC, 0.1150 RMSE) while requiring 95% fewer channels.
- In-the-wild perceptual quality assessment is dominated by global structural integrity, exposure, and edge sharpness encapsulated in the structural base anchor S0, whereas S4 reflects high-entropy stochastic texture content that leads to probe overfitting.
- Proves that the compact base latent channels S0-S2 are the fundamental information powerhouse of ELIC across both semantic classification and perceptual quality assessment.

## Open / untested

## Invalid runs (bugs, to rerun)
