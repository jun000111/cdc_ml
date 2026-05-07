from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from cdc_ml.config import (
    INTERIM_COMPLETE_RECORDS_PARQUET,
    INTERIM_CLASS_PARQUET,
    PROCESSED_CLASS_RECORDS_PARQUET,
)
from cdc_ml.datasets.merge.schema import MergeRecordsClass

app = typer.Typer()


def assign_class_type(df_records: pd.DataFrame, df_class: pd.DataFrame):
    """left join the records with the class type"""
    df_customer_records = df_records.merge(df_class, on="username", how="left")
    df_customer_records["is_one_team"] = df_customer_records["is_one_team"].fillna(0).astype(int)
    return df_customer_records


def merge_df(df_records: pd.DataFrame, df_class: pd.DataFrame) -> pd.DataFrame:
    df = assign_class_type(df_records, df_class)
    return MergeRecordsClass.validate(df)


def merge_on_disk(
    interim_records_path: Path = INTERIM_COMPLETE_RECORDS_PARQUET,
    interim_class_path: Path = INTERIM_CLASS_PARQUET,
    processed_output_path: Path = PROCESSED_CLASS_RECORDS_PARQUET,
):

    logger.info("Generating class records...")
    df_records = pd.read_parquet(interim_records_path)
    df_class = pd.read_parquet(interim_class_path)
    df = merge_df(df_records, df_class)
    logger.info(f"Total {len(df)} rows x {len(df.columns)} columns")
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_output_path)
    logger.success(f"Saved to {processed_output_path}")


@app.command()
def merge(
    interim_records_path: Path = INTERIM_COMPLETE_RECORDS_PARQUET,
    interim_class_path: Path = INTERIM_CLASS_PARQUET,
    processed_output_path: Path = PROCESSED_CLASS_RECORDS_PARQUET,
):
    merge_on_disk(
        interim_records_path,
        interim_class_path,
        processed_output_path,
    )


if __name__ == "__main__":
    app()
