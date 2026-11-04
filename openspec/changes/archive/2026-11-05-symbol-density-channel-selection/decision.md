# Decision

## Decision
Adopt discrete transmitted integer symbol density $s_c = P(\text{round}(y - \mu) \ne 0)$ alongside Shannon transmission rate $r_c = -\log_2 p(y_c)$ as verified equivalent heuristics for channel prioritization. Top-32 channel selection (`SD32` / `TR32`) is now established as the primary benchmark baseline for efficient machine task decoding in ELIC, offering 74.1% bit savings over full representation `C4` while maintaining superior accuracy (48.81% vs. 48.53%).

## Alternatives considered
- **Continuous Latent Activation Density on Reconstructed Latents ($\hat{y} \ne 0$)**: Rejected permanently. Continuous mean $\mu$ restoration causes 100% density saturation across >128 channels.
- **Complex Scale-Aware Density Metrics**: Rejected as redundant. The near-perfect rank correlation ($r_s = 0.9989$) with Shannon rate shows that discrete non-zero symbol frequency alone captures the required channel ranking without requiring explicit scale $\sigma$ modeling.

## Revisit if
- Evaluating at ultra-low compression bitrates ($\lambda=0.004$ or lower) where dead zones expand significantly and channels become extremely sparse across spatial coordinates.
- Evaluating a non-Gaussian or asymmetric entropy model where likelihood cost may diverge from dead-zone frequency.

## New hypotheses spawned
- **Low-Rate Generalization**: Does the $SD32$ channel ranking derived at $\lambda=0.032$ transfer effectively to latents compressed at $\lambda=0.004$ without re-evaluating channel statistics?
- **Joint Rate-Symbol Pruning**: Can we prune channels whose symbol density is strictly below a dead threshold ($s_c < 10^{-4}$) during the autoregressive encoding loop to accelerate ELIC inference without any accuracy penalty?

## Stop pursuing
- Measuring activation density or channel sparsity directly on continuous latents $y$ or reconstructed $\hat{y}$ without subtracting conditional mean $\mu$.
- Searching for discrepancies between Shannon rate ranking and discrete symbol density ranking at $\lambda=0.032$ (overlap is 100% on Top 32).

## Findings changed
- `F-002`: Added finding confirming that discrete transmitted symbol density $P(\text{round}(y - \mu) \ne 0)$ produces channel subsets equipotent to Shannon rate ($r_s = 0.9989$) and resolves the saturation artifact.
