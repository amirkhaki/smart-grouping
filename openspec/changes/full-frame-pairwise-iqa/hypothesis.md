# Hypothesis: Full-Frame Latent Encoding and Soft Pairwise Ranking for No-Reference Image Quality Assessment

## Claim
Training capacity-regularized probe heads with independent per-head optimizers and a soft pairwise logistic ranking loss with label-noise filtering ($\tau=0.03$) on native full-frame ELIC ($\lambda=0.45$) latents ($1024 \times 768 \rightarrow 48 \times 64$) achieves statistically significant gains over center-cropped Pearson-trained baselines across 5 random seeds ($\Delta \text{SRCC} \ge +0.030$, paired bootstrap $p < 0.01$).

## Motivation and prior evidence
- **Resolution Blind Spot**: In prior findings `F-003` and `F-004`, all probes were trained on $256 \times 256$ center crops. On KonIQ-10k's native $1024 \times 768$ images, center cropping discards **91.7%** of the image area. Human observers assign Mean Opinion Scores (MOS) based on the holistic image, including peripheral lens softness, vignetting, corner chromatic aberrations, framing, and global contrast balance—all of which are invisible to a center crop.
- **Native Divisibility**: $1024$ and $768$ are exact multiples of 64 ($16 \times 64$ and $12 \times 64$), matching ELIC's downsampling stride ($g_a \times 16$ and hyperprior $h_a \times 4$) and yielding a pristine $48 \times 64$ latent grid with zero interpolation blur.
- **Objective Mismatch**: The target evaluation metric is Spearman Rank Correlation (SRCC), yet the previous loss was 99% linear Pearson correlation (PLCC). A soft pairwise logistic loss (RankNet-style) operates directly on rank order, avoids sigmoid gradient saturation by acting on raw unbounded logits, amplifies the batch signal ($B=64 \rightarrow \sim 1,700$ informative pairs), and filters out near-ties ($|y_i - y_j| \le \tau$) as crowd-worker label noise.
- **Optimization Decoupling**: In `R009` and `R012`, training multiple heads under a shared `clip_grad_norm_` caused large heads to starve smaller heads (`S0`). Giving each head its own optimizer and independent clipping threshold ($1.0$) guarantees uncorrupted gradient updates.
- **Statistical Rigor**: Single-seed evaluations cannot differentiate true $+0.010$ gains from initialization jitter. Evaluating across 5 seeds with paired bootstrap confidence intervals on test predictions establishes genuine statistical significance.

## Falsification criterion
A 5-seed paired bootstrap evaluation on the held-out 20% test split shows that full-frame soft pairwise probe training fails to achieve a statistically significant improvement over the center-cropped baseline ($\Delta \text{SRCC} < +0.015$ or $p \ge 0.05$), or soft pairwise ranking underperforms batch Pearson loss at matched resolution.

## Expected effect
- Full-frame encoding combined with soft pairwise ranking will elevate `C3` (128 ch) from $0.7396$ to $\ge 0.770$–$0.785$ Test SRCC, and base slice `S0` (16 ch) from $0.7205$ to $\ge 0.745$–$0.755$ Test SRCC.
- 5-seed standard deviation across runs will remain tight ($\le 0.003$), and paired bootstrap will confirm $\Delta \text{SRCC} > 0$ at $p < 0.001$.

## Next step by outcome
- **Supported**: Adopt native full-frame encoding, C3 caching, and soft pairwise ranking as the universal gold standard for compressed-domain IQA.
- **Refuted (Full-frame fails)**: If full-frame encoding does not improve over center crops, it indicates ELIC's spatial pooling already acts as a scale-invariant texture descriptor, shifting focus to localized patch pooling.
- **Refuted (Pairwise fails)**: If soft pairwise ranking does not outperform Pearson correlation loss, retain Pearson loss and investigate metric-conditioned loss schedules.
- **Inconclusive**: If cross-seed variance exceeds $0.015$, ablate anchor MSE weight $\lambda_{\text{anchor}}$ or tune noise threshold $\tau \in [0.02, 0.05]$.

## Related and already tried
- `F-001`, `F-002`: Established discrete symbol density and rate ranking on Caltech-256.
- `F-003`: Refuted high-frequency slice inversion on center crops (`S0` beat `S4` and `C4`).
- `F-004`: Proved cumulative slices peak at `C3` (128 ch, $0.7396$ SRCC) while `S4` degrades performance.
- `F-005`: Proved marginal symbol density fails for IQA due to high-frequency stochastic texture bias.
