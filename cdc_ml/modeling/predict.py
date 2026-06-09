from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from cdc_ml.config import BOOKING_MODEL, PROCESSED_DATA_DIR
from cdc_ml.datasets.constants import TIMEZONE
from cdc_ml.datasets.customer_class.customer_class import clean_df as clean_class
from cdc_ml.datasets.cycle.cycle import clean_df as clean_cycle
from cdc_ml.datasets.poll.poll import build_poll_df
from cdc_ml.datasets.preference.preference import build_pref
from cdc_ml.features.build_features import build_features
from cdc_ml.modeling.config import CHOSEN_FEATURES

app = typer.Typer()


def predict(
    df_cus_pref: pd.DataFrame,
    df_cycle: pd.DataFrame,
    model_path: Path = BOOKING_MODEL,
) -> pd.DataFrame:
    """Return feature matrix with calibrated booking probabilities appended.

    Parameters
    ----------
    df_cus_pref:
        Raw customer preference table (one row per preference window).
    df_cycle:
        Raw cycle table (one row per customer cycle).
    model_path:
        Path to the serialised model artefact (XGBoost + Platt scaler pipeline).

    Returns
    -------
    DataFrame with all feature columns plus ``booking_prob`` (calibrated probability).
    """
    model = joblib.load(model_path)
    logger.debug(f"Loaded model from {model_path}")

    df_cycle = clean_cycle(df_cycle.assign(preference=None, range=None))

    df_pref = build_pref(
        df_cus_pref[
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
    )

    df_class = clean_class(df_cycle[["username", "class_type", "is_one_team"]])

    df_poll = build_poll_df(
        df_cycle["username"],
        df_cycle["id"],
        df_cycle["cycle_start"],
        df_cycle["cycle_end"],
    )

    X = build_features(df_poll, df_pref, df_class).reset_index(drop=True)

    booking_prob = model.predict_proba(X[CHOSEN_FEATURES])

    return X.assign(pred=booking_prob)


@app.command()
def run(
    cus_pref_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "cus_pref.parquet",
        help="Path to customer preference data (parquet).",
    ),
    cycle_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "cycles.parquet",
        help="Path to cycle data (parquet).",
    ),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "predictions.parquet",
        help="Where to write predictions (parquet).",
    ),
    model_path: Path = typer.Option(
        BOOKING_MODEL,
        help="Override the default model artefact path.",
    ),
):
    logger.info(f"Loading customer preferences from {cus_pref_path}")
    df_cus_pref = pd.read_parquet(cus_pref_path)

    logger.info(f"Loading cycles from {cycle_path}")
    df_cycle = pd.read_parquet(cycle_path)

    logger.info("Running inference...")
    predictions = predict(df_cus_pref, df_cycle, model_path=model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    logger.success(f"Predictions written to {output_path}  ({len(predictions):,} rows)")


if __name__ == "__main__":
    app()
