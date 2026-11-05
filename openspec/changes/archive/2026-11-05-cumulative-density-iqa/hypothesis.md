# Hypothesis: Cumulative and Symbol-Density Latent Subsets for No-Reference Image Quality Assessment

## Claim
In a frozen maximum-fidelity ELIC model ($\lambda=0.45$), cumulative slice expansion beyond base slice `S0` (`S0` [16ch] $\rightarrow$ `C1` [32ch] $\rightarrow$ `C2` [64ch] $\rightarrow$ `C3` [128ch] $\rightarrow$ `C4` [320ch]) monotonically degrades or plateaus test rank correlation (SRCC) with human MOS due to the accumulation of stochastic texture noise, whereas data-driven transmitted symbol-density channel selection (`SD8`, `SD16`, `SD32`, `SD64`, `SD128`) outperforms fixed slice groupings at capacity-matched channel budgets ($\text{SRCC}_{\text{SD16}} > \text{SRCC}_{S0}$ and $\text{SRCC}_{\text{SD32}} > \text{SRCC}_{C1}$).

## Motivation and prior evidence
- **Finding F-003**: In `latent-iqa-benchmarking`, isolated base slice `S0` (16 channels) achieved Test SRCC of 0.7314, strictly outperforming isolated high-frequency slice `S4` (0.6566 SRCC) and exceeding the full 320-channel latent `C4` (0.7289 SRCC). This indicated that high-frequency channels act as nuisance variance when estimating human perceptual quality.
- **Findings F-001 & F-002**: On Caltech-256 classification, discrete transmitted symbol density $s_c = P(\text{round}(y_c - \mu_c) \ne 0)$ perfectly isolated the most informative latent channels ($r_s = 0.9989$ rank correlation with Shannon rate), allowing `SD32` (32 ch) to beat both fixed slice `C1` (48.81% vs. 48.38%) and full latent `C4` (48.53%).
- **Open Question**: Does cumulative addition of slices (`C1` = S0+S1, `C2` = S0+S1+S2, `C3` = S0..S3, `C4` = S0..S4) dilute the clean structural quality signal in `S0`? And does data-driven symbol density channel selection identify an even cleaner, higher-correlation subset of channels within `S0`–`S2` that surpasses fixed contiguous boundaries?

## Falsification criterion
1. **Cumulative Hypothesis is Falsified if**: Cumulative channel growth monotonically increases quality correlation ($\text{SRCC}_{C4} > \text{SRCC}_{C3} > \text{SRCC}_{C2} > \text{SRCC}_{C1} > \text{SRCC}_{S0}$), or if `C1` / `C2` outperforms `S0` by at least $\Delta \text{SRCC} \ge +0.02$.
2. **Density Selection Hypothesis is Falsified if**: Capacity-matched symbol density subsets fail to achieve higher correlation than fixed contiguous slices ($\text{SRCC}_{\text{SD16}} \le \text{SRCC}_{S0}$ and $\text{SRCC}_{\text{SD32}} \le \text{SRCC}_{C1}$).

## Expected effect
- `S0` (16 ch) will achieve $\text{SRCC} \approx 0.73$, while `C1` (32 ch) and `C2` (64 ch) will drop or plateau ($\text{SRCC} \le 0.725$), and `C3` / `C4` will exhibit further dilution ($\text{SRCC} \approx 0.71$–$0.72$).
- `SD16` (16 ch) and `SD32` (32 ch) will achieve $\text{SRCC} \ge 0.740$ ($\Delta \text{SRCC} \ge +0.01$–$0.02$ over `S0` and `C1`), establishing data-driven symbol density as the optimal channel selection metric for learned latent IQA.

## Next step by outcome
- **Supported**: Adopt `SD16` or `SD32` as the definitive ultra-compact latent IQA representation, enabling 90%–95% transmission bit savings while maximizing perceptual quality assessment accuracy.
- **Refuted (Cumulative wins)**: If `C1` or `C2` substantially improves over `S0`, it indicates that multi-frequency synthesis benefits IQA only when base and mid-frequency slices are presented jointly, shifting focus to multi-scale hierarchical fusion.
- **Refuted (Density fails)**: If fixed slices match or beat symbol density, it indicates that spatial autoregressive structural ordering in ELIC (`S0` anchor) is more important for quality perception than raw marginal transmission frequency.
- **Inconclusive**: If all subsets perform within a narrow window ($\pm 0.005$ SRCC), investigate whether regression probe capacity (`w=256`) or patch sampling is bottlenecking quality discrimination.

## Related and already tried
- `F-001`: Rate-ranked channel selection on Caltech-256 (`TR32` beat `C4`).
- `F-002`: Transmitted symbol density equipotent to rate on Caltech-256 (`SD32` matched `TR32`).
- `F-003`: Latent slice inversion refuted; isolated `S0` beat `S4` and `C4` on KonIQ-10k.
