"""Calibration reliability diagrams segmented by polling-volume whales.

Drop into your viz/plotting module. Main entry point is
`plot_calibration_by_volume`, which takes pooled OOF predictions and plots
whale vs non-whale reliability curves with Wilson CIs plus prediction
histograms underneath.

Design choices baked in:
  * Segment on EXPOSURE (n_polls / is_whale), never on positive count — the
    slice must not be conditioned on the label.
  * Quantile (equal-count) bins, not equal-width, so no bin is mostly empty
    at the ~1.3% base rate. Edges are deduped, which also handles the
    discrete-prediction case (additive LUT emits one value per dow/hour cell).
  * Wilson score intervals for observed frequency — robust at low counts /
    extreme p, unlike the normal approximation.
  * Prediction histogram (log y) under the reliability axes so sparsity in the
    low-volume segment is visible rather than hidden.

Read points BELOW the diagonal as overprediction (observed < predicted) — the
non-whale failure mode you diagnosed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion.

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


def plot_calibration_by_volume(
    p_pred,
    y_true,
    is_whale,
    *,
    n_bins=10,
    title="Calibration by polling volume (pooled OOF)",
    ax_range=None,
):
    """Reliability diagram for whale vs non-whale segments.

    Parameters
    ----------
    p_pred : array of predicted P(has_booking) per polling row (pooled OOF).
    y_true : array of 0/1 has_booking labels, same length.
    is_whale : bool array, True for high-n_polls users. Use the SAME whale
        definition as your baseline sweep so the graph matches your weighting.
    n_bins : requested quantile bins (effective count may be lower).
    """
    p_pred = np.asarray(p_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    is_whale = np.asarray(is_whale, dtype=bool)

    segments = {
        "whale (high n_polls)": (is_whale, "#2563eb"),
        "non-whale": (~is_whale, "#dc2626"),
    }

    if ax_range is None:
        hi_x = float(np.quantile(p_pred, 0.999))
        ax_range = (0.0, max(hi_x, float(y_true.mean()) * 3.0))

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)

    # perfect-calibration reference
    ax.plot(ax_range, ax_range, ls="--", lw=1.0, color="#6b7280", label="perfect", zorder=1)

    for name, (mask, color) in segments.items():
        if mask.sum() == 0:
            continue
        t = reliability_table(p_pred[mask], y_true[mask], n_bins=n_bins)
        yerr = np.vstack(
            [np.clip(t["obs"] - t["lo"], 0, None), np.clip(t["hi"] - t["obs"], 0, None)]
        )
        ax.errorbar(
            t["x"],
            t["obs"],
            yerr=yerr,
            marker="o",
            ms=4.5,
            lw=1.6,
            capsize=2.5,
            color=color,
            zorder=3,
            label=f"{name}  (base={y_true[mask].mean():.2%}, " f"N={mask.sum():,})",
        )
        axh.hist(p_pred[mask], bins=40, range=ax_range, alpha=0.5, color=color, label=name)

    ax.set_xlim(*ax_range)
    ax.set_ylim(0, ax_range[1])
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.text(
        0.97,
        0.04,
        "below diagonal = overprediction",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color="#6b7280",
        style="italic",
    )
    plt.setp(ax.get_xticklabels(), visible=False)

    axh.set_yscale("log")
    axh.set_ylabel("rows")
    axh.set_xlabel("Predicted P(has_booking)")
    axh.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axh.legend(loc="upper right", fontsize=8, frameon=False)

    fig.align_ylabels([ax, axh])
    return fig


# if __name__ == "__main__":
#     # --- synthetic demo mirroring the diagnosed failure mode ---------------
#     # whales calibrated; non-whales overpredicted (observed < predicted),
#     # base rate ~1.3%, heavily skewed polling volume.
#     rng = np.random.default_rng(7)
#     n_users = 3000

#     n_polls = np.maximum(5, rng.lognormal(mean=3.2, sigma=1.1, size=n_users).astype(int))
#     whale_cut = np.quantile(n_polls, 0.90)
#     user_is_whale = n_polls >= whale_cut

#     rows_pred, rows_y, rows_whale = [], [], []
#     for u in range(n_users):
#         m = n_polls[u]
#         mult = rng.lognormal(mean=0.0, sigma=0.5)
#         p_model = np.clip(0.013 * mult * (1.25 if user_is_whale[u] else 1.0), 5e-4, 0.15)
#         p_model_row = np.clip(p_model * rng.lognormal(0.0, 0.25, size=m), 5e-4, 0.2)
#         # whales calibrated; non-whales truly book ~55% of predicted
#         p_true_row = p_model_row * (1.0 if user_is_whale[u] else 0.55)
#         rows_pred.append(p_model_row)
#         rows_y.append(rng.random(m) < p_true_row)
#         rows_whale.append(np.full(m, user_is_whale[u]))

#     p_pred = np.concatenate(rows_pred)
#     y_true = np.concatenate(rows_y).astype(float)
#     is_whale = np.concatenate(rows_whale)

#     print(
#         f"rows={len(y_true):,}  base_rate={y_true.mean():.3%}  "
#         f"whale_rows={is_whale.sum():,}  non_whale_rows={(~is_whale).sum():,}"
#     )

#     fig = plot_calibration_by_volume(
#         p_pred,
#         y_true,
#         is_whale,
#         n_bins=10,
#         title="Calibration by polling volume " "(synthetic demo)",
#     )
#     fig.savefig(
#         "/mnt/user-data/outputs/calibration_by_volume_demo.png", dpi=150, bbox_inches="tight"
#     )
#     print("saved preview png")
