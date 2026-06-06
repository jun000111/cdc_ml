from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from cdc_ml.config import (
    RECORDS_PROCESSED,
    PROPER_RECORDS_INTERIM,
    PSEUDO_BOOKINGS_INTERIM,
)
from cdc_ml.datasets.merge.schema import MergeRecordsPseudo

app = typer.Typer()


def all_user_included(df: pd.DataFrame) -> None:
    # missing_list = df_records.loc[~df_records["username"].isin(df_class["username"]), "username"]
    # if not missing_list.empty:
    #     missing = missing_list.unique().tolist()
    #     raise ValueError(f"Username not assigned with a class :{missing}")
    missing_list = df.loc[df["class_type"].isna()]
    if not missing_list.empty:
        missing = missing_list["username"].unique().tolist()
        raise ValueError(f"Username not assigned with a class :{missing}")


def assign_class_type(df_records: pd.DataFrame, df_class: pd.DataFrame):
    """left join the records with the class type"""
    df_customer_records = df_records.merge(df_class, on="username", how="left")
    return df_customer_records


def merge_df(
    df_records: pd.DataFrame,
    df_pseudo: pd.DataFrame,
) -> pd.DataFrame:

    # join the records
    df_complete_records = pd.concat([df_records, df_pseudo], axis=0, ignore_index=True)
    df_complete_records.insert(0, "id", range(len(df_complete_records)))

    # df = assign_class_type(df_complete_records, df_class)
    # all_user_included(df_complete_records)
    return MergeRecordsPseudo.validate(df_complete_records)


def merge_on_disk(
    interim_output_path: Path = RECORDS_PROCESSED,
    records_input_path: Path = PROPER_RECORDS_INTERIM,
    pseudo_input_path: Path = PSEUDO_BOOKINGS_INTERIM,
) -> None:

    logger.info(f"Merging proper and pseudo records...")
    df_records = pd.read_parquet(records_input_path)
    df_pseudo = pd.read_parquet(pseudo_input_path)
    df = merge_df(df_records, df_pseudo)
    logger.info(f"Total {len(df)} rows by {len(df.columns)} columns")
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path, index=False)
    logger.success(f"Saved to {interim_output_path} ")


@app.command()
def merge(
    interim_output_path: Path = RECORDS_PROCESSED,
    records_input_path: Path = PROPER_RECORDS_INTERIM,
    pseudo_input_path: Path = PSEUDO_BOOKINGS_INTERIM,
):
    merge_on_disk(interim_output_path, records_input_path, pseudo_input_path)


if __name__ == "__main__":
    app()
