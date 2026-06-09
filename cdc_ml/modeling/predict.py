from pathlib import Path
import joblib
import pandas as pd

from loguru import logger
from tqdm import tqdm
import typer

from cdc_ml.config import MODELS_DIR, PROCESSED_DATA_DIR, BOOKING_MODEL
from cdc_ml.datasets.preference.preference import build_pref
from cdc_ml.datasets.poll.poll import build_poll_df
from cdc_ml.datasets.customer_class.customer_class import clean_df as clean_class
from cdc_ml.datasets.cycle.cycle import clean_df as clean_cycle
from cdc_ml.features.build_features import build_features
from cdc_ml.datasets.constants import TIMEZONE
from cdc_ml.modeling.config import CHOSEN_FEATURES

app = typer.Typer()


def predict(df_cus_pref: pd.DataFrame, df_cycle: pd.DataFrame):

    model = joblib.load(BOOKING_MODEL)
    df_cycle = clean_cycle(df_cycle.assign(preference=None, range=None))
    df_pref = df_cus_pref[
        [
            "id",
            "username",
            "pref_start",
            "pref_end",
            "mon",
            "tues",
            "wed",
            "thurs",
            "fri",
            "sat",
            "sun",
            "interval",
        ]
    ]
    df_class = df_cycle[["username", "class_type", "is_one_team"]]
    df_poll = build_poll_df(
        df_cycle["username"], df_cycle["id"], df_cycle["cycle_start"], df_cycle["cycle_end"]
    )
    df_pref = build_pref(df_pref)

    df_class = clean_class(df_class)
    X = build_features(df_poll, df_pref, df_class).reset_index(drop=True)

    pred = model.predict_proba(X[CHOSEN_FEATURES])

    return pd.concat([X, pd.Series(pred, name="pred")], axis=1)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
