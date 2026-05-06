from pathlib import Path

import pandas as pd
from loguru import logger

import typer

from cdc_ml.config import (
    INTERIM_RECORDS_PARQUET,
    INTERIM_PSEUDO_PARQUET,
    INTERIM_COMPLETE_RECORDS_PARQUET,
)
from cdc_ml.datasets.records_pseudo.schema import RefinedRecords

app = typer.Typer()


def clean_r(df_records: pd.DataFrame) -> pd.DataFrame:
    """assign booking type 0 to all proper bookings
    drop irrelevant/modified columns"""
    return df_records.assign(booking_type=0).drop(columns=["booking", "created_at", "id"])


def clean_p(df_pseudo: pd.DataFrame) -> pd.DataFrame:
    """standardize the naming to be the same as records
    drop irrelevant/modified columns"""
    return df_pseudo.rename(columns={"type_of_booking": "booking_type", "name": "username"}).drop(
        columns=["lesson_date", "lesson_time", "booking date", "booking time", "id"]
    )


def merge_records(df_records: pd.DataFrame, df_pseudo: pd.DataFrame) -> pd.DataFrame:

    # join the records
    df_complete_records = pd.concat([df_records, df_pseudo], axis=0, ignore_index=True)
    df_complete_records.insert(0, "id", range(len(df_complete_records)))

    # assign id column
    return df_complete_records


def merge_df(df_records: pd.DataFrame, df_pseudo: pd.DataFrame) -> pd.DataFrame:

    df_records = clean_r(df_records)
    df_pseudo = clean_p(df_pseudo)
    df_complete_records = merge_records(df_records, df_pseudo)

    logger.info(
        f"Total {len(df_complete_records)} rows x {len(df_complete_records.columns)} columns"
    )
    return RefinedRecords.validate(df_complete_records)


def merge_on_disk(
    interim_output_path: Path = INTERIM_COMPLETE_RECORDS_PARQUET,
    records_input_path: Path = INTERIM_RECORDS_PARQUET,
    pseudo_input_path: Path = INTERIM_PSEUDO_PARQUET,
) -> None:
    df_records = pd.read_parquet(records_input_path)
    df_pseudo = pd.read_parquet(pseudo_input_path)
    df_complete_records = merge_df(df_records, df_pseudo)
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_complete_records.to_parquet(interim_output_path, index=False)


@app.command()
def run(
    interim_output_path: Path = INTERIM_COMPLETE_RECORDS_PARQUET,
    records_input_path: Path = INTERIM_RECORDS_PARQUET,
    pseudo_input_path: Path = INTERIM_PSEUDO_PARQUET,
):
    logger.info("Merging...")
    merge_on_disk(interim_output_path, records_input_path, pseudo_input_path)
    logger.success(f"Successfully merged and saved to {interim_output_path}")


if __name__ == "__main__":
    app()
