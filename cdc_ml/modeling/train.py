from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, average_precision_score

from cdc_ml.config import MODELS_DIR, PROCESSED_DATA_DIR, STAGE_1_PROCESSED
from cdc_ml.features.build_features import drop_meta_high_card_cols

app = typer.Typer()


def temporal_split(df: pd.DataFrame):
    cutoff = df["cycle_start"].quantile(0.80)

    train_data = df[df["cycle_start"] < cutoff]
    test_data = df[df["cycle_start"] >= cutoff]

    return train_data, test_data

    # train_customers = set(train_data["username"])
    # test_data = test_data[~test_data["username"].isin(train_customers)]


def time_ordered_kfold(df, time_col, n_splits=5):
    order = np.argsort(df[time_col].to_numpy(), kind="stable")  # positional
    fold_size = len(order) // (n_splits + 1)
    for i in range(n_splits):
        yield (
            order[: fold_size * (i + 1)],
            order[fold_size * (i + 1) : fold_size * (i + 2)],
        )


def baseline_const(X_tr, y_tr):
    return X_tr.assign(_y=y_tr)["_y"].mean()


def baseline_lut(X_tr, y_tr, X_va):

    base = y_tr.mean()
    lut = X_tr.assign(_y=y_tr).groupby(["polling_dow", "polling_hour"])["_y"].mean().reset_index()
    p_lut = (
        X_va[["polling_dow", "polling_hour"]]
        .merge(lut, on=["polling_dow", "polling_hour"], how="left")["_y"]
        .fillna(base)
        .to_numpy()
    )
    return p_lut


def train(df: pd.DataFrame, extra: list, seed=42):
    y = df["has_booking"].to_numpy()
    X = drop_meta_high_card_cols(df).drop(columns=(["has_booking"] + extra))
    print(X.sample())
    n = len(y)
    oof_xgb = np.full(n, np.nan)
    oof_marg = np.full(n, np.nan)
    oof_const = np.full(n, np.nan)
    tr_brier_list, tr_pr_list = [], []
    val_brier_list, val_pr_list = [], []
    marg_brier_list, marg_pr_list = [], []
    const_brier_list, const_pr_list = [], []
    models = []
    for fold, (tr, va) in enumerate(time_ordered_kfold(df, time_col="cycle_start")):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]
        # --- XGBoost ---
        model = xgb.XGBClassifier(
            n_estimators=500,
            objective="binary:logistic",
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va] = model.predict_proba(X_va)[:, 1]
        tr_pred = model.predict_proba(X_tr)[:, 1]
        models.append(model)
        # --- marg baseline (dow × hour lookup) ---
        p_lut = baseline_lut(X_tr, y_tr, X_va)
        oof_marg[va] = p_lut
        const = baseline_const(X_tr, y_tr)
        oof_const[va] = const
        # --- per-fold scores ---
        fold_tr_brier = brier_score_loss(y_tr, tr_pred)
        fold_tr_pr = average_precision_score(y_tr, tr_pred)
        fold_val_brier = brier_score_loss(y_va, oof_xgb[va])
        fold_val_pr = average_precision_score(y_va, oof_xgb[va])
        fold_marg_brier = brier_score_loss(y_va, p_lut)
        fold_marg_pr = average_precision_score(y_va, p_lut)
        fold_const_brier = brier_score_loss(y_va, oof_const[va])
        fold_const_pr = average_precision_score(y_va, oof_const[va])
        tr_brier_list.append(fold_tr_brier)
        tr_pr_list.append(fold_tr_pr)
        val_brier_list.append(fold_val_brier)
        val_pr_list.append(fold_val_pr)
        marg_brier_list.append(fold_marg_brier)
        marg_pr_list.append(fold_marg_pr)
        const_brier_list.append(fold_const_brier)
        const_pr_list.append(fold_const_pr)
        # --- fold report ---
        tr_dates = df.iloc[tr]["cycle_start"]
        va_dates = df.iloc[va]["cycle_start"]
        print(
            f"fold {fold}: "
            f"train n={len(tr):>6} ({tr_dates.min().date()} → {tr_dates.max().date()})  "
            f"val n={len(va):>6} ({va_dates.min().date()} → {va_dates.max().date()})  \n"
            f"  train_pos={y_tr.mean():.3f}  val_pos={y_va.mean():.3f}  \n"
            f"  marg_brier={fold_marg_brier:.4f}  marg_pr={fold_marg_pr:.4f}  \n"
            f"  xgb_brier_val={fold_val_brier:.4f}  xgb_pr_val={fold_val_pr:.4f}  \n"
            f"  xgb_brier_tr={fold_tr_brier:.4f}  xgb_pr_tr={fold_tr_pr:.4f}  \n"
        )

    mask = ~np.isnan(oof_xgb)
    y_s = y[mask]

    def _row(name, brier_list, pr_list, oof_arr):
        b_pool = brier_score_loss(y_s, oof_arr[mask])
        p_pool = average_precision_score(y_s, oof_arr[mask])
        return (
            f"{name:<7} brier={b_pool:.4f} ({np.mean(brier_list):.4f}±{np.std(brier_list):.4f})   "
            f"pr_auc={p_pool:.4f} ({np.mean(pr_list):.4f}±{np.std(pr_list):.4f})"
        )

    print(
        f"\nOOF  base_rate={y_s.mean():.4f}  n={mask.sum()}/{n}      "
        f"format: pooled (mean±std over folds)\n"
        f"{_row('const', const_brier_list, const_pr_list, oof_const)}\n"
        f"{_row('marg',  marg_brier_list,  marg_pr_list,  oof_marg)}\n"
        f"{_row('xgb',   val_brier_list,   val_pr_list,   oof_xgb)}\n"
        f"xgb_tr  brier=  ---   ({np.mean(tr_brier_list):.4f}±{np.std(tr_brier_list):.4f})   "
        f"pr_auc=  ---   ({np.mean(tr_pr_list):.4f}±{np.std(tr_pr_list):.4f})\n"
    )
    return oof_xgb, oof_marg, oof_const


def train_on_disk(data: Path, models: Path):
    logger.info("Training some model...")
    df = pd.read_parquet(data)
    df_train, df_test = temporal_split(df)

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
