"""Evaluation metrics for IQA: Spearman Rank Correlation (SRCC), Pearson Correlation (PLCC), and RMSE."""

from typing import Dict, Tuple, Union
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn


def _logistic_func(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    """ITU-T standard 5-parameter monotonic logistic mapping function."""
    lp = np.clip(b2 * (x - b3), -100.0, 100.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(lp))) + b4 * x + b5


def compute_iqa_metrics(
    predictions: Union[np.ndarray, list],
    targets: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Compute SRCC, PLCC (via 5-parameter logistic curve fit), and RMSE.

    Args:
        predictions: Model predictions (N,).
        targets: Ground-truth MOS values (N,).

    Returns:
        Dict with keys 'srcc', 'plcc', 'rmse', and 'count'.
    """
    p = np.asarray(predictions, dtype=float).ravel()
    m = np.asarray(targets, dtype=float).ravel()

    valid = ~(np.isnan(p) | np.isnan(m) | np.isinf(p) | np.isinf(m))
    if valid.sum() < 3:
        return {"srcc": 0.0, "plcc": 0.0, "rmse": 999.0, "count": int(valid.sum())}

    pv = p[valid]
    mv = m[valid]

    # 1. Spearman Rank Order Correlation Coefficient (SRCC)
    srcc_val, _ = spearmanr(pv, mv)
    srcc = float(srcc_val) if not np.isnan(srcc_val) else 0.0

    # 2. Pearson Linear Correlation Coefficient (PLCC) via 5-parameter logistic mapping
    try:
        p0 = [float(np.max(mv)), 10.0, float(np.mean(pv)), 0.1, 0.1]
        popt, _ = curve_fit(_logistic_func, pv, mv, p0=p0, maxfev=10000)
        pv_mapped = _logistic_func(pv, *popt)
        plcc_val, _ = pearsonr(pv_mapped, mv)
        plcc = float(plcc_val) if not np.isnan(plcc_val) else 0.0
    except Exception:
        # Fallback to direct linear Pearson correlation if logistic fit fails to converge
        plcc_val, _ = pearsonr(pv, mv)
        plcc = float(plcc_val) if not np.isnan(plcc_val) else 0.0
        pv_mapped = pv

    # 3. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean((pv_mapped - mv) ** 2)))

    return {
        "srcc": round(srcc, 4),
        "plcc": round(plcc, 4),
        "rmse": round(rmse, 4),
        "count": int(valid.sum()),
    }


def pearson_correlation_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable Pearson correlation loss: L = 1 - r."""
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    p_mean = torch.mean(preds_flat)
    t_mean = torch.mean(targets_flat)

    p_sub = preds_flat - p_mean
    t_sub = targets_flat - t_mean

    cov = torch.sum(p_sub * t_sub)
    var_p = torch.sum(p_sub ** 2)
    var_t = torch.sum(t_sub ** 2)

    r = cov / (torch.sqrt(var_p * var_t) + eps)
    return 1.0 - r
