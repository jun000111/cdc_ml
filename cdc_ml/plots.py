import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from cdc_ml.modeling.calibrate import *
from cdc_ml.modeling.evaluation import gains_bootstrap
import seaborn as sns
import pandas as pd


def plot_additive_heatmap(
    X, y, alpha=10.0, annot=True, dow_labels=None, cmap="Blues", ax=None, title=None
):
    """Heatmap of the additive_lut_logit prediction surface over (dow, hour).

    Mirrors additive_lut_logit: smoothed marginal logit deviations summed in
    logit space (outer-sum -> no interaction by construction), then squashed
    to probability.
    """
    df = X.assign(_y=np.asarray(y))
    base = df["_y"].mean()
    z0 = np.log(base / (1 - base))

    def smoothed_logit(col):
        g = df.groupby(col)["_y"]
        rate = (g.sum() + alpha * base) / (g.count() + alpha)  # shrink toward base
        return np.log(rate / (1 - rate))

    zd = smoothed_logit("polling_dow")
    zh = smoothed_logit("polling_hour")

    dows = sorted(df["polling_dow"].unique())
    hours = sorted(df["polling_hour"].unique())

    # deviations from base logit; unseen category -> 0 (same as the .fillna(0) in your fn)
    dev_d = (pd.Series(dows, index=dows).map(zd) - z0).fillna(0.0).to_numpy()
    dev_h = (pd.Series(hours, index=hours).map(zh) - z0).fillna(0.0).to_numpy()

    Z_logit = z0 + dev_d[:, None] + dev_h[None, :]  # additive (outer-sum) structure
    Z = 1.0 / (1.0 + np.exp(-Z_logit))  # predicted P(has_booking)
    P = Z * 100.0  # as percent

    if dow_labels is None:
        dow_labels = [
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] if 0 <= d < 7 else str(d)
            for d in dows
        ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(P, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours)
    ax.set_yticks(range(len(dows)))
    ax.set_yticklabels(dow_labels)
    ax.set_xlabel("polling_hour")
    ax.set_ylabel("polling_dow")
    ax.set_title(
        title or f"Additive baseline  ·  P(has_booking) %   (base rate {base*100:.2f}%)",
        fontsize=12,
        pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("P(has_booking) %", rotation=270, labelpad=14)

    if annot:
        vmin, vmax = P.min(), P.max()
        thresh = vmin + 0.55 * (vmax - vmin)
        for i in range(len(dows)):
            for j in range(len(hours)):
                ax.text(
                    j,
                    i,
                    f"{P[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                )

    fig.tight_layout()
    return fig, ax, Z


def concentration_share(df: pd.DataFrame):

    fig, ax = plt.subplots(figsize=(20, 8))

    counts = df["username"].value_counts().reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()

    counts = counts.sort_values("share", ascending=False).reset_index(drop=True)
    counts["cum_share"] = counts["share"].cumsum()
    counts["cum_frac"] = (counts.index + 1) / len(counts)

    ax.plot(counts["cum_frac"], counts["cum_share"], label="Poll count concentration")

    total_sum = df["has_booking"].sum()

    bookings = (
        df.groupby("username")["has_booking"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="sum")
    )

    bookings["share"] = bookings["sum"] / total_sum
    bookings["cum_share"] = bookings["share"].cumsum()
    bookings["cum_frac"] = (bookings.index + 1) / len(bookings)

    ax.plot(bookings["cum_frac"], bookings["cum_share"], label="Booking concentration")

    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect equality")

    ax.set_xlabel("cumulative fraction of users (ranked heaviest first)")
    ax.set_ylabel("cumulative share")
    ax.set_title("Concentration comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def bootstrap_distribution(boots, point, base, hi, lo, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    ax.hist(boots, bins=40, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.axvline(
        point,
        color="crimson",
        linewidth=1.8,
        label=f"Point estimate {point:.4f} ({point/base:.2f}×)",
    )
    ax.axvline(lo, color="dimgray", linewidth=1.2, linestyle="--")
    ax.axvline(
        hi, color="dimgray", linewidth=1.2, linestyle="--", label=f"95 % CI [{lo:.4f}, {hi:.4f}]"
    )
    ax.axvspan(lo, hi, alpha=0.08, color="dimgray")
    ax.axvline(
        base, color="darkorange", linewidth=1.2, linestyle=":", label=f"Base rate {base:.4f}"
    )
    ax.set_xlabel("PR-AUC (user-block bootstrap)")
    ax.set_ylabel("Bootstrap samples")
    ax.set_title("PR-AUC bootstrap distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def booking_rate_plot(df: pd.DataFrame, target):
    label_rate = df.groupby(target)["has_booking"].agg(share="mean", count="count").reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    base = df["has_booking"].mean()

    # Top: label rate with 95% CI

    sns.barplot(data=df, x=target, y="has_booking", ax=axes[0], errorbar=("ci", 95))
    axes[0].axhline(base, color="red", linestyle="--", label=f"base rate ({base:.3f})")
    axes[0].set_ylabel("P(has_booking)")

    # Bottom: exposure per category
    sns.barplot(data=label_rate, x=target, y="count", ax=axes[1], color="steelblue")
    axes[1].set_ylabel("n observations")

    plt.tight_layout()


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


def bootstrapped_gain_curve(res, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # if your calibrated OOF preds are an array: df_train = df_train.assign(p_oof_cal=p_oof_cal)

    g, lo, med, hi = res["grid"] * 100, res["lo"] * 100, res["med"] * 100, res["hi"] * 100
    b_lo, b_med, b_hi = res["budget_ci"] * 100

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
    return ax, res
