# Decision

## Decision
Adopt dynamic per-channel transmission bitrate ranking ($r_c = \mathbb{E}[-\log_2 p(y_c)]$) calculated from cached continuous latents and frozen ELIC Gaussian conditional layers as the canonical channel prioritization metric for smart grouping in the benchmarking suite, permanently replacing heuristic non-zero activation density (`LS`).

Establish **TR32** (32 channels, 0.0475 bpp, 49.40% Top-1 accuracy) as the primary compact baseline representation for future grouping and compression-for-machine experiments, outperforming the full 320-channel representation (48.53% at 0.1835 bpp).

## Alternatives considered
1. **Continuation with Non-Zero Activation Density (`LS`)**: Rejected. Checking non-zero activation density on $\hat{y}$ resulted in severe saturation artifacts with >128 channels tied at 100.00% density due to continuous predicted means ($\mu$). Furthermore, TR subsets beat LS subsets by up to +3.79% Top-1 accuracy across all low-rate budgets.
2. **Fixed Handcrafted Slice Unions ($S_0, C_1..C_4$)**: Retained as architectural baselines, but superseded for rate-distortion-accuracy optimization. Fixed contiguous slice boundaries enforce rigid architectural assumptions that fail to bundle high-information channels distributed across different frequency scales (e.g., TR8 gathers channels across S0, S2, and S3).
3. **Thresholded Continuous Latent Density ($\mathbb{I}(|y| > \epsilon)$)**: Considered as an intermediate heuristic, but discarded because choosing $\epsilon$ is arbitrary, uncalibrated across channels with differing variances $\sigma_c$, and does not directly correspond to actual bits transmitted over the channel.

## Revisit if
- Evaluating a learned compression architecture that does not utilize a Gaussian/logistic entropy conditional with context prediction (e.g., non-parametric factorized hyperpriors where activations are uncentered and density might behave differently).
- A channel pruning or grouping mechanism demonstrates that joint channel correlation/redundancy metrics significantly improve upon marginal per-channel rate ranking.

## New hypotheses spawned
1. **Multi-Rate Generalization**: Does the rate-ranked channel ordering remain invariant across compression quality levels ($\lambda \in \{0.004, 0.016, 0.15, 0.45\}$), or do lower-rate models undergo selective channel collapse that requires rate-adaptive slice grouping?
2. **Dynamic Channel Grouping**: Can grouping the top 32–64 rate-ranked channels into learned adaptive subsets (smart grouping) match the decoding throughput of fixed contiguous slices while retaining the +1.0% to +3.8% accuracy advantage?
3. **Multi-Seed Variance**: Does the +0.87% Top-1 accuracy advantage of TR32 over full C4 remain statistically significant across multiple probe training seeds (e.g., seeds 43, 44)?

## Stop pursuing
- **Activation Density Ranking (`LS`) on ELIC Centered Latents**: Cease all further experiments using non-zero activation counts on contextually predicted latents $\hat{y} = \text{round}(y - \mu) + \mu$, as the continuous offset $\mu$ fundamentally invalidates non-zero density as an informative sparsity metric.
- **Probe Scalability beyond 64 Channels**: Cease allocating compute to large channel subsets ($K \ge 128$) when optimizing for Top-1 accuracy, as accuracy plateaus at 32 channels while probe capacity overhead exacerbates the generalization gap.

## Findings changed
- `F-001`: Added (Rate-Ranked Latent Channel Prioritization Outperforms Fixed Slices and Non-Zero Activation Density).
