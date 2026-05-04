from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from cdc_ml.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from cdc_ml.dataset.constants import TIMESLOTS, TIMEZONE
from cdc_ml.dataset.pseudo.schema import CleanedPseudo

app = typer.Typer()


def to_lesson_timestamp(df: pd.DataFrame) -> pd.DataFrame:

    df["lesson_date"] = df["lesson_date"].astype("str")
    df["lesson_time"] = df["lesson_time"].astype("str")
    df["lesson_timestamp"] = pd.to_datetime(
        df["lesson_date"] + " " + df["lesson_time"]
    ).dt.tz_localize(TIMEZONE)
    return df


def to_booking_timestamp(df: pd.DataFrame) -> pd.DataFrame:

    df["booking date"] = df["booking date"].astype("str")
    df["booking time"] = df["booking time"].astype("str")
    df["booking_timestamp"] = pd.to_datetime(
        df["booking date"] + " " + df["booking time"]
    ).dt.tz_localize(TIMEZONE)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = to_lesson_timestamp(df)
    df = to_booking_timestamp(df)
    return CleanedPseudo.validate(df, lazy=True)


def clean_from_disk(
    raw_input_path: Path = EXTERNAL_DATA_DIR / "pseudo_bookings.xlsx",
    interim_output_path: Path = INTERIM_DATA_DIR / "pseudo.parquet",
):
    df = pd.read_excel(raw_input_path)
    logger.info(f"Before cleaning : {len(df)} rows x {len(df.columns)}")
    df = clean_df(df)
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path, index=False)
    logger.info(f"After cleaning : {len(df)} rows x {len(df.columns)}")


@app.command()
def run(
    raw_input_path: Path = EXTERNAL_DATA_DIR / "pseudo_bookings.xlsx",
    interim_output_path: Path = INTERIM_DATA_DIR / "pseudo.parquet",
):
    logger.info("Starting...")
    clean_from_disk(raw_input_path, interim_output_path)
    logger.success(f"Saved cleaned pseudo to {interim_output_path} ")


if __name__ == "__main__":
    app()
