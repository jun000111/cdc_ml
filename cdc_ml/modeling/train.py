from pathlib import Path
import datetime

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import numpy as np
import xgboost as xgb
from sklearn.metrics import brier_score_loss, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from cdc_ml.config import MODELS_DIR, STAGE_1_PROCESSED
from cdc_ml.modeling.baseline import *
from cdc_ml.features.build_features import get_whale_users
from cdc_ml.modeling.crossval import make_holdout_split
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from cdc_ml.modeling.model import BookingModel
from cdc_ml.modeling.calibrate import fit_platt_scaler, calibrate_proba
from cdc_ml.modeling.config import FEATS_ABL_LIST, CHOSEN_FEATURES, CV_SPLITS, RANDOM_STATE
from xgboost import XGBClassifier
from cdc_ml.config import BEST_PARAMS, BOOKING_MODEL
import joblib
import json
from sklearn.linear_model import LogisticRegression

app = typer.Typer()


def fit_model(X: pd.DataFrame, y):
    best_params = json.loads(Path(BEST_PARAMS).read_text())
    model = XGBClassifier(**best_params).fit(X, y)
    return model


def fit_calibrator(model, X, y, groups):

    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    p_oof = cross_val_predict(
        clone(model), X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    platt = fit_platt_scaler(p_oof, y)
    oof_cal = calibrate_proba(p_oof, platt)
    return platt, oof_cal


def train(df: pd.DataFrame):
    X, y, groups = (
        df[FEATS_ABL_LIST[CHOSEN_FEATURES]],
        df["has_booking"].to_numpy(),
        df["username"],
    )

    model = fit_model(X, y)

    platt, oof_cal = fit_calibrator(model, X, y, groups)

    calibrated_model = BookingModel(model, platt, FEATS_ABL_LIST[CHOSEN_FEATURES])
    joblib.dump(calibrated_model, BOOKING_MODEL)
    logger.success(f"Saved model to {BOOKING_MODEL}")

    return calibrated_model, oof_cal


def train_on_disk(data: Path, models: Path):
    logger.info("Training some model...")
    df = pd.read_parquet(data)
    df_train, df_test = make_holdout_split(df)

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
