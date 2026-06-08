import pandas as pd
import numpy as np
from scipy import stats
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from cdc_ml.config import FEATURE_ABLATIONS_RESULTS


def baseline_pr_auc():
    records = pd.read_json(FEATURE_ABLATIONS_RESULTS, lines=True)

    rf_xgb = records.loc[
        (
            (records["run_id"] == "baseline_baseline")
            & (records["feature_id"] == "baseline")
            & records["model"].isin(
                ["rf", "xgb", "const", "marg_dow", "marg_hour", "add", "joint"]
            )
        )
    ]
    latest_run_at = rf_xgb["run_at"].max()

    records = rf_xgb[rf_xgb["run_at"] == latest_run_at].copy()

    res = rf_xgb.groupby("model").agg(
        pr_auc=("pr_auc_pooled", "mean"),
        pr_auc_lift=("pr_auc_lift", "mean"),
        pr_auc_std=("pr_auc_folds_std", "mean"),
        whale_lift=("whale_pr_lift", "mean"),
        non_whale_lift=("non_whale_pr_lift", "mean"),
    )
    return res


def compare_feature_sets(model="xgb", target_run=None):
    records = pd.read_json(FEATURE_ABLATIONS_RESULTS, lines=True)

    df = records.loc[(records["run_id"] == target_run) & (records["model"] == model)].copy()

    if df.empty:
        return df

    latest_run_at = df["run_at"].max()
    df = df[df["run_at"] == latest_run_at].copy()

    baseline_row = df.loc[df["feature_id"] == "full", "pr_auc_pooled"]
    if baseline_row.empty:
        raise ValueError("Missing 'full' feature baseline")

    baseline = baseline_row.iloc[0]

    full_row = df.loc[df["feature_id"] == "full"].iloc[0]
    full_whale = full_row["whale_pr_lift"]
    full_non_whale = full_row["non_whale_pr_lift"]

    df["diff_vs_full"] = df["pr_auc_pooled"] - baseline
    df["pct_diff_vs_full"] = (df["diff_vs_full"] / baseline) * 100

    df["whale_lift_vs_full"] = df["whale_pr_lift"] - full_whale
    df["non_whale_lift_vs_full"] = df["non_whale_pr_lift"] - full_non_whale

    df["whale_lift_pct_vs_full"] = df["whale_lift_vs_full"] / (abs(full_whale) + 1e-9) * 100
    df["non_whale_lift_pct_vs_full"] = (
        df["non_whale_lift_vs_full"] / (abs(full_non_whale) + 1e-9) * 100
    )

    df = df.sort_values("diff_vs_full", ascending=False)

    # keep only display columns
    df = df[
        [
            "feature_id",
            "pr_auc_pooled",
            "diff_vs_full",
            "pr_auc_folds_std",
            "pct_diff_vs_full",
            "whale_pr_lift",
            "whale_lift_vs_full",
            "non_whale_pr_lift",
            "non_whale_lift_vs_full",
        ]
    ]

    return df


def cramers_v(df, x, y):
    ct = pd.crosstab(df[x], df[y])
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1  # df_min — drives the thresholds
    v = np.sqrt(chi2 / (n * k))

    small, medium, large = (c / np.sqrt(k) for c in (0.10, 0.30, 0.50))
    if v >= large:
        label = "large"
    elif v >= medium:
        label = "medium"
    elif v >= small:
        label = "small"
    else:
        label = "negligible / no association"

    print(f"χ²={chi2:.1f}  p={p:.2e}  V={v:.3f}  (df={k})  → {label}")
    return v, p


def paired_t(
    target_run: str,
    model_1_name: str,
    model_2_name: str,
):
    """returns d,t,p,ci,n_pos in dataframe form"""
    records = pd.read_json(FEATURE_ABLATIONS_RESULTS, lines=True)

    latest_run_at = records["run_at"].max()
    run_records = records.loc[(records["run_id"] == target_run) & (records["model"] == "xgb")]

    latest_run = run_records.loc[records["run_at"] == latest_run_at]

    model_1_metrics = latest_run[latest_run["features"] == model_1_name]["pr_auc_pooled"]
    model_2_metrics = latest_run[latest_run["features"] == model_2_name]["pr_auc_pooled"]
    diff = np.asarray(model_1_metrics) - np.asarray(
        model_2_metrics
    )  # paired, row-wise by iteration
    d_bar = diff.mean()  # = +0.0019, your point estimate
    t, p = stats.ttest_rel(model_1_metrics, model_2_metrics)  # two-sided
    ci = stats.t.interval(
        0.95, len(diff) - 1, loc=d_bar, scale=diff.std(ddof=1) / np.sqrt(len(diff))
    )
    n_pos = (diff > 0).sum()  # how many of 20 seeds favored dow

    return pd.DataFrame(
        {
            "d": d_bar,
            "t": t,
            "p": p,
            "ci": [f"{ci[0]} , {ci[1]}"],
            "n_pos": n_pos,
            "n_total": len(diff),
        }
    )


def pr_auc_ci_by_user(y, score, user_ids, n_boot=1000, seed=0, ax=None):
    y, score, user_ids = map(np.asarray, (y, score, user_ids))
    base, point = y.mean(), average_precision_score(y, score)
    users = np.unique(user_ids)
    rows_by_user = {u: np.where(user_ids == u)[0] for u in users}
    rng, boots = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rows_by_user[u] for u in rng.choice(users, len(users), replace=True)]
        )
        if y[idx].sum() == 0:
            continue
        boots.append(average_precision_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"users={len(users)} rows={len(y)} positives={int(y.sum())} base={base:.4f}")
    print(
        f"PR-AUC={point:.4f} ({point/base:.2f}x)  95% CI=[{lo:.4f}, {hi:.4f}] ([{lo/base:.2f}x, {hi/base:.2f}x])\n"
    )
    return boots, point, base, hi, lo


def adversarial_validation(X_train, X_test):
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]  # 0 = train, 1 = test
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
    print(f"adversarial AUC = {auc:.3f}   (0.5 = identical, ~0.65+ = medium, 0.8+ = strong)")
    clf.fit(X, y)
    return clf


def per_customer_at_budget(
    df, pred_col, budget=0.30, user_col="username", label_col="has_booking"
):
    p = pred_col
    keep = p >= np.quantile(p, 1 - budget)  # poll iff P >= tau (top `budget` of polls)
    book = df[label_col].to_numpy(float)
    t = (
        df.assign(_caught=book * keep, _booking=book, _polled=keep.astype(float))
        .groupby(user_col)
        .agg(
            bookings=("_booking", "sum"),
            caught=("_caught", "sum"),
            polls=("_booking", "size"),
            polled=("_polled", "sum"),
        )
    )
    t = t[t.bookings > 0]  # recall undefined without bookings
    t["recall"] = t.caught / t.bookings
    t["poll_kept"] = t.polled / t.polls
    t = t.sort_values("bookings", ascending=False)
    t["whale"] = t.bookings.cumsum().shift(fill_value=0) < 0.5 * t.bookings.sum()
    assert len(p) == len(df), f"pred length {len(p)} != df length {len(df)}"
    assert not np.isnan(p).any(), f"{np.isnan(p).sum()} NaN predictions"
    return t


def _gains_curve(p, y):
    """Cumulative gains (x=frac polls, y=frac bookings), stepping at each distinct
    prediction so a threshold includes whole tie-groups. Origin prepended.
    Assumes one row = one poll (matches polls = size)."""
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    cut = np.r_[True, p[1:] != p[:-1]]  # start of each distinct-P run
    ends = np.r_[np.flatnonzero(cut)[1:] - 1, p.size - 1]
    cum_polls = np.arange(1, p.size + 1)
    cum_bookings = np.cumsum(y)
    x = np.r_[0.0, cum_polls[ends] / cum_polls[-1]]
    yv = np.r_[0.0, cum_bookings[ends] / cum_bookings[-1]]
    return x, yv


def _budget_at_recall(x, y, r):
    j = np.searchsorted(y, r, side="left")  # first point with recall >= r
    if j == 0:
        return 0.0
    y0, y1, x0, x1 = y[j - 1], y[j], x[j - 1], x[j]
    return x0 if y1 == y0 else x0 + (r - y0) * (x1 - x0) / (y1 - y0)


def gains_bootstrap(
    df,
    pred_col,
    user_col="username",
    label_col="has_booking",
    target_recall=0.90,
    n_boot=1000,
    grid=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 201) if grid is None else np.asarray(grid, float)
    p = pred_col
    y = df[label_col].to_numpy(float)
    if y.sum() == 0:
        raise ValueError("no positive labels")

    idx = list(df.groupby(user_col, sort=False).indices.values())  # rows per user
    n_u = len(idx)

    x0, y0 = _gains_curve(p, y)  # point estimate
    point = {"curve": np.interp(grid, x0, y0), "budget": _budget_at_recall(x0, y0, target_recall)}

    curves, budgets = np.empty((n_boot, grid.size)), np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([idx[i] for i in rng.integers(0, n_u, n_u)])
        xb, yb = _gains_curve(p[rows], y[rows])
        curves[b] = np.interp(grid, xb, yb)
        budgets[b] = _budget_at_recall(xb, yb, target_recall)

    lo, med, hi = np.percentile(curves, [2.5, 50, 97.5], axis=0)
    return {
        "grid": grid,
        **point,
        "lo": lo,
        "med": med,
        "hi": hi,
        "budgets": budgets,
        "budget_ci": np.percentile(budgets, [2.5, 50, 97.5]),
        "x": x0,
        "y": y0,
    }
