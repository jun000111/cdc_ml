from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from cdc_ml.config import (
    PSEUDO_BOOKINGS_EXTERNAL,
    PSEUDO_BOOKINGS_INTERIM,
)
from cdc_ml.datasets.constants import TIMESLOTS, TIMEZONE
from cdc_ml.datasets.pseudo_records.schema import CleanedPseudo

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
    return df.rename(columns={"type_of_booking": "booking_type", "name": "username"}).drop(
        columns=["lesson_date", "lesson_time", "booking date", "booking time", "id"]
    )


def is_valid_lessons(df: pd.DataFrame) -> None:
    lesson_times = df["lesson_at"].dt.strftime("%H:%M")
    invalid_mask = df["lesson_at"].notna() & ~lesson_times.isin(TIMESLOTS)

    if invalid_mask.any():
        invalid = df[invalid_mask]
        logger.error(f"{invalid_mask.sum()} invalid stamps")
        logger.error(invalid[["username", "lesson_at"]])


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_username(df)

    df["lesson_at"] = to_timestamp_at(df["lesson_date"], df["lesson_time"])
    df["booking_at"] = to_timestamp_at(df["booking date"], df["booking time"])
    df = clean_columns(df)
    is_valid_lessons(df)
    return CleanedPseudo.validate(df, lazy=True)


def clean_from_disk(
    raw_input_path: Path = PSEUDO_BOOKINGS_EXTERNAL,
    interim_output_path: Path = PSEUDO_BOOKINGS_INTERIM,
):
    df = pd.read_excel(raw_input_path)
    logger.info(f"Before cleaning : {len(df)} rows x {len(df.columns)}")
    df = clean_df(df)
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path, index=False)
    logger.info(f"After cleaning : {len(df)} rows x {len(df.columns)}")
    logger.success(f"Saved cleaned pseudo to {interim_output_path} ")


@app.command()
def run(
    raw_input_path: Path = PSEUDO_BOOKINGS_EXTERNAL,
    interim_output_path: Path = PSEUDO_BOOKINGS_INTERIM,
):
    clean_from_disk(raw_input_path, interim_output_path)


if __name__ == "__main__":
    app()
