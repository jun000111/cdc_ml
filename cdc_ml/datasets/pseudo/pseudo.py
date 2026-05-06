from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from cdc_ml.config import (
    EXTERNAL_PSEUDO_EXCEL,
    INTERIM_PSEUDO_PARQUET,
)
from cdc_ml.datasets.constants import TIMESLOTS, TIMEZONE
from cdc_ml.datasets.pseudo.schema import CleanedPseudo

app = typer.Typer()


def to_timestamp_at(date: pd.Series, time: pd.Series) -> pd.Series:
    """convert both lesson and booking timestamp to tz-aware timestamp format"""
    date = date.astype(str)
    time = time.astype(str)
    return pd.to_datetime(date + " " + time, errors="coerce").dt.tz_localize(TIMEZONE)


def normalize_username(df: pd.DataFrame) -> pd.DataFrame:
    """add id and also rename name to username"""
    df = df.rename(columns={"name": "username"})
    df.insert(0, "id", range(len(df)))
    df["username"] = df["username"].str.lower()
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["lesson_date", "lesson_time", "booking date", "booking time"])


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_username(df)

    df["lesson_at"] = to_timestamp_at(df["lesson_date"], df["lesson_time"])
    df["booking_at"] = to_timestamp_at(df["booking date"], df["booking time"])
    df = clean_columns(df)
    return CleanedPseudo.validate(df, lazy=True)


def clean_from_disk(
    raw_input_path: Path = EXTERNAL_PSEUDO_EXCEL,
    interim_output_path: Path = INTERIM_PSEUDO_PARQUET,
):
    df = pd.read_excel(raw_input_path)
    logger.info(f"Before cleaning : {len(df)} rows x {len(df.columns)}")
    df = clean_df(df)
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path, index=False)
    logger.info(f"After cleaning : {len(df)} rows x {len(df.columns)}")


@app.command()
def run(
    raw_input_path: Path = EXTERNAL_PSEUDO_EXCEL,
    interim_output_path: Path = INTERIM_PSEUDO_PARQUET,
):
    logger.info("Starting...")
    clean_from_disk(raw_input_path, interim_output_path)
    logger.success(f"Saved cleaned pseudo to {interim_output_path} ")


if __name__ == "__main__":
    app()
