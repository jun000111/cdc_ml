import numpy as np


def baseline_const(X_tr, y_tr):
    return X_tr.assign(_y=y_tr)["_y"].mean()


def joint_lut_hier(X_tr, y_tr, X_va, alpha_marg=10.0, alpha_cell=10.0):
    df = X_tr.assign(_y=y_tr)
    base = df["_y"].mean()
    z0 = np.log(base / (1 - base))

    # 1) smoothed marginal logit deviations (shrink toward base)
    def marg_dev(col, a):
        g = df.groupby(col)["_y"]
        rate = (g.sum() + a * base) / (g.count() + a)
        return np.log(rate / (1 - rate)) - z0  # cat -> logit deviation

    d_dow, d_hour = marg_dev("polling_dow", alpha_marg), marg_dev("polling_hour", alpha_marg)

    # additive prediction (probability) for any (dow, hour) rows
    def add_prior(frame):
        z = (
            z0
            + frame["polling_dow"].map(d_dow).fillna(0.0)
            + frame["polling_hour"].map(d_hour).fillna(0.0)
        )
        return 1 / (1 + np.exp(-z))

    # 2) joint cell rate, shrunk toward the ADDITIVE prior (not toward base)
    cell = df.groupby(["polling_dow", "polling_hour"])["_y"].agg(n="size", m="mean")
    prior = add_prior(cell.reset_index()).to_numpy()
    cell["_rate"] = (cell["n"] * cell["m"] + alpha_cell * prior) / (cell["n"] + alpha_cell)

    # 3) map val rows; unseen cell -> additive prior (which backs off to base)
    merged = X_va[["polling_dow", "polling_hour"]].merge(
        cell["_rate"].reset_index(), on=["polling_dow", "polling_hour"], how="left"
    )
    p = np.where(merged["_rate"].isna(), add_prior(X_va).to_numpy(), merged["_rate"].to_numpy())
    return np.clip(p, 1e-6, 1 - 1e-6)


def additive_lut_logit(X_tr, y_tr, X_va, alpha=10.0):
    df = X_tr.assign(_y=y_tr)
    base = df["_y"].mean()
    z0 = np.log(base / (1 - base))  # logit(base)

    def smoothed_logit(col):
        g = df.groupby(col)["_y"]
        rate = (g.sum() + alpha * base) / (g.count() + alpha)  # shrink toward base
        return np.log(rate / (1 - rate))  # finite by construction

    zd = smoothed_logit("polling_dow")  # Series: dow -> logit
    zh = smoothed_logit("polling_hour")  # Series: hour -> logit

    d_dow = (X_va["polling_dow"].map(zd) - z0).fillna(0.0)  # unseen -> 0 deviation
    d_hour = (X_va["polling_hour"].map(zh) - z0).fillna(0.0)

    z = z0 + d_dow + d_hour
    p = 1 / (1 + np.exp(-z))
    return p.clip(1e-6, 1 - 1e-6).to_numpy()


def marg_dow_baseline(X_tr, y_tr, X_va, alpha=10.0):
    """
    Marginal baseline by polling_dow with Laplace smoothing.

    Args:
        X_tr, y_tr: training features and labels
        X_va: validation features
        alpha: smoothing strength (higher = more shrinkage toward base)
               Laplace = 1, but 5-20 often works better empirically

    Returns:
        np.array of predictions
    """
    base = y_tr.mean()

    # Compute counts and successes per dow
    counts = (
        X_tr.assign(_y=y_tr)
        .groupby("polling_dow")
        .agg(successes=("_y", "sum"), total=("_y", "count"))
        .reset_index()
    )

    # Shrink toward base: (successes + alpha*base) / (total + alpha)
    counts["_y"] = (counts["successes"] + alpha * base) / (counts["total"] + alpha)

    p_marg = (
        X_va[["polling_dow"]]
        .merge(counts[["polling_dow", "_y"]], on="polling_dow", how="left")["_y"]
        .fillna(base)
        .to_numpy()
    )
    return p_marg


def marg_hour_baseline(X_tr, y_tr, X_va, alpha=10.0):
    """
    Marginal baseline by polling_hour with Laplace smoothing.

    Args:
        X_tr, y_tr: training features and labels
        X_va: validation features
        alpha: smoothing strength (higher = more shrinkage toward base)

    Returns:
        np.array of predictions
    """
    base = y_tr.mean()

    # Compute counts and successes per hour
    counts = (
        X_tr.assign(_y=y_tr)
        .groupby("polling_hour")
        .agg(successes=("_y", "sum"), total=("_y", "count"))
        .reset_index()
    )

    # Shrink toward base
    counts["_y"] = (counts["successes"] + alpha * base) / (counts["total"] + alpha)

    p_marg = (
        X_va[["polling_hour"]]
        .merge(counts[["polling_hour", "_y"]], on="polling_hour", how="left")["_y"]
        .fillna(base)
        .to_numpy()
    )
    return p_marg
