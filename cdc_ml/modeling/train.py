from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance

from cdc_ml.config import MODELS_DIR, PROCESSED_DATA_DIR, STAGE_1_PROCESSED
from cdc_ml.features.build_features import drop_meta_high_card_cols

app = typer.Typer()


# def temporal_split(df: pd.DataFrame):
#     cutoff = df["cycle_start"].quantile(0.80)

#     train_data = df[df["cycle_start"] < cutoff]
#     test_data = df[df["cycle_start"] >= cutoff]

#     return train_data, test_data

#     # train_customers = set(train_data["username"])
#     # test_data = test_data[~test_data["username"].isin(train_customers)]


# def time_ordered_kfold(df, time_col, n_splits=5):
#     order = np.argsort(df[time_col].to_numpy(), kind="stable")  # positional
#     fold_size = len(order) // (n_splits + 1)
#     for i in range(n_splits):
#         yield (
#             order[: fold_size * (i + 1)],
#             order[fold_size * (i + 1) : fold_size * (i + 2)],
#         )


def baseline_const(X_tr, y_tr):
    return X_tr.assign(_y=y_tr)["_y"].mean()


def joint_lut_hier(X_tr, y_tr, X_va, alpha_marg=20.0, alpha_cell=30.0):
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


def additive_lut_logit(X_tr, y_tr, X_va, alpha=50.0):
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


def marg_dow_baseline(X_tr, y_tr, X_va):
    base = y_tr.mean()
    marg = X_tr.assign(_y=y_tr).groupby(["polling_dow"])["_y"].mean().reset_index()
    p_marg = (
        X_va[["polling_dow"]]
        .merge(marg, on=["polling_dow"], how="left")["_y"]
        .fillna(base)
        .to_numpy()
    )
    return p_marg


def marg_hour_baseline(X_tr, y_tr, X_va):
    base = y_tr.mean()
    marg = X_tr.assign(_y=y_tr).groupby(["polling_hour"])["_y"].mean().reset_index()
    p_marg = (
        X_va[["polling_hour"]]
        .merge(marg, on=["polling_hour"], how="left")["_y"]
        .fillna(base)
        .to_numpy()
    )
    return p_marg


def train(df: pd.DataFrame, extra: list, seed=42):
    y = df["has_booking"].to_numpy()
    X = drop_meta_high_card_cols(df).drop(columns=(["has_booking"] + extra))
    print(X.sample())
    n = len(y)
    oof_xgb = np.full(n, np.nan)
    oof_marg_dow = np.full(n, np.nan)
    oof_marg_hour = np.full(n, np.nan)
    oof_joint = np.full(n, np.nan)
    oof_const = np.full(n, np.nan)
    oof_add = np.full(n, np.nan)

    tr_brier_list, tr_pr_list = [], []
    val_brier_list, val_pr_list = [], []
    marg_dow_brier_list, marg_dow_pr_list = [], []
    marg_hour_brier_list, marg_hour_pr_list = [], []
    joint_brier_list, joint_pr_list = [], []
    const_brier_list, const_pr_list = [], []
    add_brier_list, add_pr_list = [], []
    models = []
    whales_pt_mask = df["username"].isin(["anmol", "jy", "mya"]).to_numpy()
    whales_pc_mask = df["username"].isin(["kim", "jy", "flower"]).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=120)

    for fold, (tr, va) in enumerate(sgkf.split(X, y=df["has_booking"], groups=df["username"])):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]
        # print("----------------------------")
        # print(df.iloc[va].groupby(["username"])["has_booking"].mean().sort_values())
        # --- XGBoost ---
        model = xgb.XGBClassifier(
            n_estimators=500,
            objective="binary:logistic",
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        models.append(model)

        r = permutation_importance(model, X_va, y_va, n_repeats=10, random_state=seed, n_jobs=-1)
        print(r.importances_mean)

        oof_xgb[va] = model.predict_proba(X_va)[:, 1]
        tr_pred = model.predict_proba(X_tr)[:, 1]
        # --- marg baseline (dow × hour lookup) ---
        p_lut = joint_lut_hier(X_tr, y_tr, X_va)
        oof_joint[va] = p_lut
        marg_dow = marg_dow_baseline(X_tr, y_tr, X_va)
        marg_hour = marg_hour_baseline(X_tr, y_tr, X_va)
        add = additive_lut_logit(X_tr, y_tr, X_va)
        const = baseline_const(X_tr, y_tr)
        oof_const[va] = const
        oof_marg_dow[va] = marg_dow
        oof_marg_hour[va] = marg_hour
        oof_add[va] = add
        # --- per-fold scores ---
        fold_tr_brier = brier_score_loss(y_tr, tr_pred)
        fold_tr_pr = average_precision_score(y_tr, tr_pred)
        fold_val_brier = brier_score_loss(y_va, oof_xgb[va])
        fold_val_pr = average_precision_score(y_va, oof_xgb[va])
        fold_joint_brier = brier_score_loss(y_va, p_lut)
        fold_joint_pr = average_precision_score(y_va, p_lut)
        fold_const_brier = brier_score_loss(y_va, oof_const[va])
        fold_const_pr = average_precision_score(y_va, oof_const[va])
        fold_marg_dow_brier = brier_score_loss(y_va, oof_marg_dow[va])
        fold_marg_dow_pr = average_precision_score(y_va, oof_marg_dow[va])
        fold_marg_hour_brier = brier_score_loss(y_va, oof_marg_hour[va])
        fold_marg_hour_pr = average_precision_score(y_va, oof_marg_hour[va])
        fold_add_brier = brier_score_loss(y_va, oof_add[va])
        fold_add_pr = average_precision_score(y_va, oof_add[va])
        tr_brier_list.append(fold_tr_brier)
        tr_pr_list.append(fold_tr_pr)
        val_brier_list.append(fold_val_brier)
        val_pr_list.append(fold_val_pr)
        joint_brier_list.append(fold_joint_brier)
        joint_pr_list.append(fold_joint_pr)
        marg_dow_brier_list.append(fold_marg_dow_brier)
        marg_dow_pr_list.append(fold_marg_dow_pr)
        marg_hour_brier_list.append(fold_marg_hour_brier)
        marg_hour_pr_list.append(fold_marg_hour_pr)
        const_brier_list.append(fold_const_brier)
        const_pr_list.append(fold_const_pr)
        add_brier_list.append(fold_add_brier)
        add_pr_list.append(fold_add_pr)
        # --- fold report ---
        tr_dates = df.iloc[tr]["cycle_start"]
        va_dates = df.iloc[va]["cycle_start"]
        print(
            f"fold {fold}: "
            f"train n={len(tr):>6} ({tr_dates.min().date()} → {tr_dates.max().date()})  "
            f"val n={len(va):>6} ({va_dates.min().date()} → {va_dates.max().date()})  \n"
            f"  train_pos={y_tr.mean():.3f}  val_pos={y_va.mean():.3f}  \n"
            f"  marg_dow_brier={fold_marg_dow_brier:.4f}  marg_dow_pr={fold_marg_dow_pr:.4f}  \n"
            f"  marg_hour_brier={fold_marg_hour_brier:.4f}  marg_hour_pr={fold_marg_hour_pr:.4f}  \n"
            f"  add_brier={fold_add_brier:.4f}  add_pr={fold_add_pr:.4f}  \n"
            f"  joint_brier={fold_joint_brier:.4f}  joint_pr={fold_joint_pr:.4f}  \n"
            f"  xgb_brier_val={fold_val_brier:.4f}  xgb_pr_val={fold_val_pr:.4f}  \n"
            f"  xgb_brier_tr={fold_tr_brier:.4f}  xgb_pr_tr={fold_tr_pr:.4f}  \n"
        )

    def _row(name, brier_list, pr_list, oof_arr):
        b_pool = brier_score_loss(y, oof_arr)
        p_pool = average_precision_score(y, oof_arr)
        return (
            f"{name:<7} brier={b_pool:.4f} ({np.mean(brier_list):.4f}±{np.std(brier_list):.4f}) lift={b_pool/y.mean()}   "
            f"pr_auc={p_pool:.4f} ({np.mean(pr_list):.4f}±{np.std(pr_list):.4f}) lift={p_pool/y.mean()}"
        )

    whale_pt_base = y[whales_pt_mask].mean()
    non_whale_pt_base = y[~whales_pt_mask].mean()
    whale_pc_base = y[whales_pc_mask].mean()
    non_whale_pc_base = y[~whales_pc_mask].mean()

    whales_brier_joint = brier_score_loss(y[whales_pc_mask], oof_joint[whales_pc_mask])
    whales_pr_joint = average_precision_score(y[whales_pt_mask], oof_joint[whales_pt_mask])
    non_whales_brier_joint = brier_score_loss(y[~whales_pc_mask], oof_joint[~whales_pc_mask])
    non_whales_pr_joint = average_precision_score(y[~whales_pt_mask], oof_joint[~whales_pt_mask])

    whales_brier_xgb = brier_score_loss(y[whales_pc_mask], oof_xgb[whales_pc_mask])
    whales_pr_xgb = average_precision_score(y[whales_pt_mask], oof_xgb[whales_pt_mask])
    non_whales_brier_xgb = brier_score_loss(y[~whales_pc_mask], oof_xgb[~whales_pc_mask])
    non_whales_pr_xgb = average_precision_score(y[~whales_pt_mask], oof_xgb[~whales_pt_mask])

    print(
        f"\nOOF  base_rate={y.mean():.4f}  n={len(y)}/{n}      "
        f"format: pooled (mean±std over folds)\n"
        f"{_row('const', const_brier_list, const_pr_list, oof_const)}\n"
        f"{_row('marg_dow',  marg_dow_brier_list,  marg_dow_pr_list,  oof_marg_dow)}\n"
        f"{_row('marg_hour',  marg_hour_brier_list,  marg_hour_pr_list,  oof_marg_hour)}\n"
        f"{_row("add",  add_brier_list,  add_pr_list,  oof_add)}\n"
        f"{_row('joint',  joint_brier_list,  joint_pr_list,  oof_joint)}\n"
        f"          whales brier={whales_brier_joint} lift={ whales_brier_joint/whale_pc_base}\n"
        f"          non whales brier={non_whales_brier_joint} lift={non_whales_brier_joint/whale_pc_base}\n"
        f"          whales pr={whales_pr_joint} lift={whales_pr_joint/whale_pt_base}\n"
        f"          non whales pr={non_whales_pr_joint} lift={non_whales_pr_joint/non_whale_pt_base}\n"
        f"{_row('xgb',   val_brier_list,   val_pr_list,   oof_xgb)}\n"
        f"          whales brier={whales_brier_xgb} lift={whales_brier_xgb/whale_pt_base}\n"
        f"          non whales brier={non_whales_brier_xgb} lift={non_whales_brier_xgb/non_whale_pt_base}\n"
        f"          whales pr={whales_pr_xgb} lift={whales_pr_xgb/whale_pc_base}\n"
        f"          non whales pr={non_whales_pr_xgb} lift={non_whales_pr_xgb/non_whale_pc_base}\n"
        f"xgb_tr  brier=  ---   ({np.mean(tr_brier_list):.4f}±{np.std(tr_brier_list):.4f})   "
        f"pr_auc=  ---   ({np.mean(tr_pr_list):.4f}±{np.std(tr_pr_list):.4f})\n"
    )
    return oof_xgb, oof_joint, oof_const,models


def train_on_disk(data: Path, models: Path):
    logger.info("Training some model...")
    df = pd.read_parquet(data)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(sgkf.split(df, y=df["has_booking"], groups=df["username"]))
    df_train, df_test = df.iloc[tr], df.iloc[te]

    train(df_train)
    logger.success("Modeling training complete.")


@app.command()
def run(
    data_path: Path = STAGE_1_PROCESSED,
    model_path: Path = MODELS_DIR / "model.pkl",
):
    train_on_disk(data_path, model_path)


if __name__ == "__main__":
    app()
