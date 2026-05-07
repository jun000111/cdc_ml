from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from cdc_ml.config import (
    INTERIM_COMPLETE_RECORDS_PARQUET,
    INTERIM_RECORDS_PARQUET,
    INTERIM_PSEUDO_PARQUET,
)
from cdc_ml.datasets.merge.schema import MergeRecordsPseudo

app = typer.Typer()


def merge_df(df_records: pd.DataFrame, df_pseudo: pd.DataFrame) -> pd.DataFrame:

    # join the records
    df_complete_records = pd.concat([df_records, df_pseudo], axis=0, ignore_index=True)
    df_complete_records.insert(0, "id", range(len(df_complete_records)))

    # assign id column
    return MergeRecordsPseudo.validate(df_complete_records)


def merge_on_disk(
    interim_output_path: Path = INTERIM_COMPLETE_RECORDS_PARQUET,
    records_input_path: Path = INTERIM_RECORDS_PARQUET,
    pseudo_input_path: Path = INTERIM_PSEUDO_PARQUET,
) -> None:

    logger.info(f"Merging proper and pseudo records...")
    df_records = pd.read_parquet(records_input_path)
    df_pseudo = pd.read_parquet(pseudo_input_path)
    df_complete_records = merge_df(df_records, df_pseudo)
    logger.info(
        f"Total {len(df_complete_records)} rows by {len(df_complete_records.columns)} columns"
    )
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_complete_records.to_parquet(interim_output_path, index=False)
    logger.success(f"Saved to {interim_output_path} ")


@app.command()
def merge(
    interim_output_path: Path = INTERIM_COMPLETE_RECORDS_PARQUET,
    records_input_path: Path = INTERIM_RECORDS_PARQUET,
    pseudo_input_path: Path = INTERIM_PSEUDO_PARQUET,
):
    merge_on_disk(
        interim_output_path,
        records_input_path,
        pseudo_input_path,
    )


if __name__ == "__main__":
    app()
