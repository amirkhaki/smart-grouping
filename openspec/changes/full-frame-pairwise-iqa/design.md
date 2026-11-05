# Design: Full-Frame Latent Encoding and Soft Pairwise Ranking for No-Reference Image Quality Assessment

## Mode
Confirmatory (5 random seeds on a held-out test split, evaluated with paired bootstrap hypothesis testing).

## Baseline and the single changed variable
- **Baseline**: Center-cropped ($256 \times 256$, $16 \times 16$ latent) single-optimizer Pearson-trained baseline (`S0` = $0.7205 \pm \text{std}$, `C3` = $0.7396 \pm \text{std}$) from Run `R012`.
- **Systematic Upgrades (Evaluated as a coherent paradigm shift)**:
  1. **Full-Frame Native Encoding**: Encode native $1024 \times 768$ KonIQ-10k images via frozen ELIC $\lambda=0.45$, yielding pristine $48 \times 64$ latent spatial maps with zero resampling distortion, cached with HDF5 LZF chunk compression.
  2. **Soft Pairwise Logistic Ranking Loss**: Replace batch-level Pearson loss with RankNet-style softplus loss with label-noise thresholding ($\tau = 0.03$) and an anchor MSE branch ($\lambda_{\text{anchor}} = 0.1$) on raw unbounded logits.
  3. **Decoupled Optimization**: Independent per-head AdamW optimizers and per-head gradient norm clipping ($1.0$) to prevent cross-head gradient starvation.
  4. **Regularized Head Architecture with Statistics Pooling**: Shrunk head (`stem=128`, 2 residual blocks, channel dropout $p=0.1$, weight decay $0.02$) with spatial statistics pooling (concatenating spatial mean, standard deviation, and maximum).

## Evaluation (protected)
- **Metric**: Spearman Rank Correlation Coefficient (SRCC) and 5-parameter non-linear logistic curve-fitted Pearson Correlation (PLCC).
- **Split / data version**: Pinned deterministic 70/10/20 train/val/test split ($N = 10,073$, train = 7,051, val = 1,007, test = 2,015) in `koniq10k_elic0450_fullframe_latents.h5` cached in Kaggle kernel `smart-grouping-iqa-tier1-fullframe`.
- **Eval script**: `benchmarks/iqa/train_probe.py` using `benchmarks/iqa/metrics.py:compute_iqa_metrics`.
- **Statistical Testing**: Evaluated across 5 random seeds (`seed` $\in \{42, 43, 44, 45, 46\}$), reporting Mean $\pm$ Std. Saved test prediction vectors are submitted to paired bootstrap resampling ($B = 2,000$ iterations) to compute 95% Confidence Intervals for $\Delta \text{SRCC}$ and empirical two-sided $p$-values.
- **Oracle / best-case bound**: Supervised end-to-end deep IQA backbones (e.g. HyperIQA) achieve $\text{SRCC} \approx 0.90$–$0.92$. Feature-extracted linear/MLP probes over frozen compression representations typically achieve $\text{SRCC} \approx 0.77$–$0.82$.

## Tiers
- **Tier 1 (Smoke test, Run R013)**: Unit test verifying full-frame extraction ($1024 \times 768 \rightarrow 48 \times 64$), HDF5 LZF compression, independent per-head optimizers, and soft pairwise loss on synthetic/mock latents.
- **Tier 2 (Signal test, Run R014)**: Full Tier-1 extraction pipeline on Kaggle 2x Tesla T4 GPUs caching all 10,073 KonIQ-10k images into `koniq10k_elic0450_fullframe_latents.h5` (~7 GB LZF compressed).
- **Tier 3 (Full benchmark, Run R015)**: 5-seed training across views (`S0`, `C1`, `C2`, `C3`, `C4`) for 30 epochs on Kaggle 2x Tesla T4 GPUs with independent optimizers, soft pairwise ranking loss, and paired bootstrap testing.

## Seeds and runtime
- **Seeds**: 42, 43, 44, 45, 46 (5 seeds).
- **Rough runtime per run**:
  - Run R014 (Tier-1 full-frame caching): ~12–15 minutes on 2x Tesla T4 GPUs (batch size 16).
  - Run R015 (Tier-3 5-seed probe training): ~35 minutes on 2x Tesla T4 GPUs attaching the full-frame cache.
- **Stopping rule**: Terminate after Run R015. If the 95% bootstrap confidence interval for $\Delta \text{SRCC}$ spans zero ($p \ge 0.05$), the hypothesis is falsified.

## Confounds and risks
- **GPU VRAM during Full-Frame Extraction**: A $1024 \times 768$ image requires $12\times$ more memory in ELIC's intermediate feature maps than a $256 \times 256$ crop. Mitigated by lowering batch size from 64 to 16 with multi-GPU `CustomDataParallel` distribution.
- **Kaggle Disk Limit (20 GB)**: Full 320 channels at $48 \times 64$ is 19.8 GB in raw float16. Mitigated by enabling HDF5 `compression="lzf"` with chunk shape `(16, 320, 48, 64)`, reducing disk footprint to ~6.5–7.5 GB.
- **Pairwise Tie Threshold ($\tau$)**: Setting $\tau$ too high discards real quality gradients; setting it too low fits crowd-worker noise. Mitigated by setting $\tau=0.03$, corresponding to the empirical standard error of KonIQ-10k subjective ratings on the normalized $[0, 1]$ scale.
