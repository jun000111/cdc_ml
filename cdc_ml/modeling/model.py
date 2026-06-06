# cdc_ml/tune.py
import json
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier
from cdc_ml.modeling.config import RANDOM_STATE, XGB_BASE, PARAM_DIST, CV_SPLITS
from cdc_ml.config import BEST_PARAMS
from cdc_ml.modeling.calibrate import calibrate_proba
import pandas as pd
import numpy as np
from loguru import logger


class BookingModel:
    def __init__(self, booster, calibrator, feature_names):
        self.booster = booster  # fitted XGBoost
        self.calibrator = calibrator  # fitted Platt LogisticRegression
        self.feature_names = feature_names  # locked order — XGBoost is silent if cols reorder

    def predict_proba(self, raw_df):
        X = raw_df[self.feature_names]
        scores = self.booster.predict_proba(X)[:, 1]
        return calibrate_proba(scores, self.calibrator)


def randSearch(X: pd.DataFrame, y: np.ndarray, groups):
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        XGBClassifier(**XGB_BASE),
        PARAM_DIST,
        n_iter=50,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=False,
        verbose=1,
    )
    search.fit(X, y, groups=groups)
    print(search.best_score_, search.best_params_)
    BEST_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS.write_text(json.dumps(search.best_params_, indent=2))
    logger.success(f"Saved best params to {BEST_PARAMS}")
