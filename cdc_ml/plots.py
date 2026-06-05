import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from cdc_ml.modeling.calibrate import *
from cdc_ml.modeling.evaluation import gains_bootstrap


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


def bootstrapped_gain_curve(df: pd.DataFrame, pred):

    # if your calibrated OOF preds are an array: df_train = df_train.assign(p_oof_cal=p_oof_cal)
    df_train = df.assign(p_oof_cal=pred)
    res = gains_bootstrap(df_train, pred_col="p_oof_cal")
    g, lo, med, hi = res["grid"] * 100, res["lo"] * 100, res["med"] * 100, res["hi"] * 100
    b_lo, b_med, b_hi = res["budget_ci"] * 100

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot([0, 100], [0, 100], "--", color="0.6", lw=1, label="random")
    ax.fill_between(g, lo, hi, color="#2f6db0", alpha=0.18, label="95% CI (user bootstrap)")
    ax.plot(g, med, color="#2f6db0", lw=2.2, label="median gains curve")
    ax.errorbar(
        b_med,
        90,
        xerr=[[b_med - b_lo], [b_hi - b_med]],
        fmt="o",
        color="#c0392b",
        capsize=4,
        zorder=6,
        label=f"90% recall @ {b_med:.0f}% polls  [{b_lo:.0f}-{b_hi:.0f}%]",
    )
    ax.axhline(90, color="#c0392b", ls=":", lw=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("% of polling volume kept (highest predicted P first)")
    ax.set_ylabel("% of bookings captured")
    ax.set_title("Per-poll budget vs bookings captured (user-bootstrapped)", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig, res
