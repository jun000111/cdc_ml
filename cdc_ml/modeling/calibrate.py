import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def to_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt_scaler(oof_proba, y):
    platt = LogisticRegression(C=1e6)
    platt.fit(to_logit(oof_proba).reshape(-1, 1), y)
    return platt


def calibrate_proba(proba, platt):
    return platt.predict_proba(to_logit(proba).reshape(-1, 1))[:, 1]


def wilson_interval(k, n, z=1.96):
    """Calibration reliability diagrams segmented by polling-volume whales.

    * Segment on EXPOSURE (n_polls / is_whale), never on positive count — the
        slice must not be conditioned on the label.
    * Quantile (equal-count) bins, not equal-width, so no bin is mostly empty
        at the ~1.3% base rate. Edges are deduped, which also handles the
        discrete-prediction case (additive LUT emits one value per dow/hour cell).
    * Wilson score intervals for observed frequency — robust at low counts /
        extreme p, unlike the normal approximation.
    * Prediction histogram (log y) under the reliability axes so sparsity in the
        low-volume segment is visible rather than hidden.

    95% Wilson score interval for a binomial proportion.

    Returns (p_hat, lower, upper). p_hat is the raw observed frequency (the
    plotted point); the interval is the Wilson score interval, which stays
    sensible when k is tiny or n is small.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lower = np.clip(center - half, 0.0, 1.0)
    upper = np.clip(center + half, 0.0, 1.0)
    return p, lower, upper


def _quantile_bin_index(p_pred, n_bins):
    """Assign each prediction to a quantile bin. Dedups edges, so the effective
    number of bins may be < n_bins when predictions are discrete (LUT) or
    heavily point-massed. Returns (edges, idx, n_effective)."""
    edges = np.quantile(p_pred, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:  # all predictions identical
        return edges, np.zeros(len(p_pred), dtype=int), 1
    idx = np.clip(np.digitize(p_pred, edges[1:-1], right=False), 0, len(edges) - 2)
    return edges, idx, len(edges) - 1


def reliability_table(p_pred, y_true, n_bins=10):
    """Per-bin mean predicted prob, observed freq, Wilson CI, and count."""
    p_pred = np.asarray(p_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    _, idx, n_eff = _quantile_bin_index(p_pred, n_bins)

    x, k_arr, n_arr = [], [], []
    for b in range(n_eff):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        x.append(p_pred[mask].mean())
        k_arr.append(y_true[mask].sum())
        n_arr.append(n)

    x = np.array(x)
    k_arr = np.array(k_arr)
    n_arr = np.array(n_arr)
    obs, lo, hi = wilson_interval(k_arr, n_arr)
    return {"x": x, "obs": obs, "lo": lo, "hi": hi, "n": n_arr, "k": k_arr}
