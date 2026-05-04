from pathlib import Path

import pandas as pd
from loguru import logger

import typer

from cdc_ml.config import INTERIM_DATA_DIR
from cdc_ml.dataset.records_pseudo.schema import RefinedRecords

app = typer.Typer()


def clean_r(df_r: pd.DataFrame) -> pd.DataFrame:
    df_r["booking_type"] = pd.Series(0, index=df_r.index)
    df_r = df_r.drop(columns=["booking", "created_at", "id"])
    return df_r


def clean_p(df_p: pd.DataFrame) -> pd.DataFrame:
    df_p = df_p.rename(columns={"type_of_booking": "booking_type", "name": "username"})
    df_p = df_p.drop(columns=["lesson_date", "lesson_time", "booking date", "booking time"])
    return df_p


def merge(df_r: pd.DataFrame, df_p: pd.DataFrame) -> pd.DataFrame:
    df_r = df_r.copy()
    df_p = df_p.copy()

    df_r = clean_r(df_r)
    df_p = clean_p(df_p)

    df_r_p = pd.concat([df_r, df_p], axis=0, ignore_index=True)
    logger.info(f"Total {len(df_r_p)} rows x {len(df_r_p.columns)} columns")
    return RefinedRecords.validate(df_r_p)


def merge_on_disk(interim_path: Path = INTERIM_DATA_DIR / "records_refined.parquet"):
    df_records = pd.read_parquet(INTERIM_DATA_DIR / "records.parquet")
    df_pseudo = pd.read_parquet(INTERIM_DATA_DIR / "pseudo.parquet")
    df_records_psuedo = merge(df_records, df_pseudo)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df_records_psuedo.to_parquet(interim_path)


@app.command()
def run(interim_path: Path = INTERIM_DATA_DIR / "records_refined.parquet"):
    logger.info("Merging data...")
    merge_on_disk(interim_path)
    logger.success(f"Successfully merged and saved to {interim_path}")


if __name__ == "__main__":
    app()
