from pathlib import Path
import json

from loguru import logger
import pandas as pd
import typer
import joblib
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

from cdc_ml.config import (
    STAGE_1_PROCESSED,
    BEST_PARAMS,
    BOOKING_MODEL,
    DEV_OOF_METRICS,
    PROD_OOF_METRICS,
)
from cdc_ml.modeling.model import BookingModel
from cdc_ml.modeling.calibrate import fit_platt_scaler, calibrate_proba
from cdc_ml.modeling.config import CHOSEN_FEATURES, CV_SPLITS, RANDOM_STATE
from cdc_ml.modeling.data import make_holdout_split

app = typer.Typer()


def fit_model(X: pd.DataFrame, y) -> XGBClassifier:
    best_params = json.loads(Path(BEST_PARAMS).read_text())
    return XGBClassifier(**best_params).fit(X, y)


def fit_calibrator(model, X, y, groups):
    """Out-of-fold probabilities (fresh clones per fold) + a Platt scaler fit on them.

    The fitted `model` passed in is NOT used for predictions here — cross_val_predict
    refits a clone on each fold's training rows, so every p_oof[i] is a held-out prediction.
    """
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    p_oof = cross_val_predict(
        clone(model), X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    platt = fit_platt_scaler(p_oof, y)
    cal_oof = calibrate_proba(p_oof, platt)
    return platt, p_oof, cal_oof


def compute_metrics(y, p_rank, p_cal, groups) -> dict:
    """Ranking metrics use raw scores; probability-quality metrics use calibrated ones."""
    return {
        "pr_auc": float(average_precision_score(y, p_rank)),
        "brier": float(brier_score_loss(y, p_cal)),
        "log_loss": float(log_loss(y, p_cal)),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def oof_pred(y_true, p_oof, cal_oof, groups):
    oof_df = pd.DataFrame(
        {
            "y_true": y_true,
            "p_oof": p_oof,
            "cal_oof": cal_oof,
            "groups": groups,
        }
    )
    return oof_df


def train(df: pd.DataFrame, dev: bool = False) -> BookingModel:
    X, y, groups = df[CHOSEN_FEATURES], df["has_booking"].to_numpy(), df["username"]

    model = fit_model(X, y)
    logger.success("Model training complete.")

    platt, p_oof, cal_oof = fit_calibrator(model, X, y, groups)
    logger.success("Calibration complete.")

    calibrated_model = BookingModel(model, platt, CHOSEN_FEATURES)
    joblib.dump(calibrated_model, BOOKING_MODEL)
    logger.success(f"Saved model to {BOOKING_MODEL}")

    to_save = DEV_OOF_METRICS if dev else PROD_OOF_METRICS
    oof = oof_pred(y, p_oof, cal_oof, groups)
    oof.to_parquet(to_save)

    return calibrated_model


def train_on_disk(data_path: Path = STAGE_1_PROCESSED, dev: bool = False):
    logger.info(f"Training for {"dev" if dev else "prod"} env")
    logger.info(f"Loading processed data from {data_path}")
    df = pd.read_parquet(data_path)
    if dev:
        df_train, _ = make_holdout_split(df)
        train(df_train, dev)
    else:
        train(df, dev)

    logger.success("Training pipeline complete.")


@app.command()
def run(data_path: Path = STAGE_1_PROCESSED, dev: bool = False):
    train_on_disk(data_path, dev)


if __name__ == "__main__":
    app()
