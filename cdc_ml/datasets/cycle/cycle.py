from pathlib import Path
import pandas as pd

import typer
from loguru import logger

from cdc_ml.config import (
    BOOKING_CYCLES_EXTERNAL,
    BOOKING_CYCLES_PROCESSED,
)
from cdc_ml.datasets.constants import TIMEZONE
from cdc_ml.datasets.cycle.schema import CleanedCycle

app = typer.Typer()


def filter_na_date_mask(df: pd.DataFrame) -> pd.DataFrame:
    """some rows of the original dataset collected contains unknown cycle start date, drop them"""
    null_mask = pd.to_datetime(df["cycle start date"], errors="coerce").isna()
    return df.loc[~null_mask]


def to_cycle_timestamp(cycle_start_date: pd.Series, cycle_start_time) -> pd.Series:

    return pd.to_datetime(
        pd.to_datetime(cycle_start_date, format="%Y-%m-%d %H:%M:%S").dt.strftime("%Y-%m-%d")
        + " "
        + cycle_start_time.astype(str)
    ).dt.tz_localize(TIMEZONE)


def convert_column_type(df: pd.DataFrame) -> pd.DataFrame:
    """assign these columns to str to avoid typing issue"""
    df["preference"] = df["preference"].astype(str)
    df["range"] = df["range"].astype(str)

    df["username"] = df["username"].str.lower()
    df.tail()
    df.insert(0, "id", range(len(df)))

    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:

    # drop pre-modified columns
    df = df.drop(
        columns=["cycle start date", "cycle start time", "cycle end date", "cycle end time"]
    )
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    logger.info(f"Pre cleaning:{len(df)} rows x {len(df.columns)} columns")

    df = convert_column_type(df)

    df = filter_na_date_mask(df)

    # convert both cycle start and end to tz-aware timestamp
    df["cycle_start"] = to_cycle_timestamp(df["cycle start date"], df["cycle start time"])
    df["cycle_end"] = to_cycle_timestamp(df["cycle end date"], df["cycle end time"])
    df = clean_columns(df)

    return CleanedCycle.validate(df)


def clean_from_disk(
    external_input_path: Path = BOOKING_CYCLES_EXTERNAL,
    interim_output_path: Path = BOOKING_CYCLES_PROCESSED,
):

    logger.info("Starting...")
    df = pd.read_excel(external_input_path)
    df = clean_df(df)
    logger.info(f"Total:{len(df)} rows x {len(df.columns)} columns")
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path, index=False)
    logger.success(f"Cleaned and saved to {interim_output_path}")


@app.command()
def run(
    external_input_path: Path = BOOKING_CYCLES_EXTERNAL,
    interim_output_path: Path = BOOKING_CYCLES_PROCESSED,
):
    clean_from_disk(external_input_path, interim_output_path)


if __name__ == "__main__":
    app()
