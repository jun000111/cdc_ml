from pathlib import Path

from loguru import logger
import pandas as pd
import typer
from sqlalchemy import create_engine, text


from cdc_ml.config import RAW_DATA_DIR, DATABASE_URL, INTERIM_DATA_DIR
from cdc_ml.dataset.records.schema import CleanedRecords

from cdc_ml.dataset.records.constants import (
    RECORDS_DATE_PATTERN1,
    RECORDS_DATE_PATTERN2,
    NAME_DIC,
    NAMES_TO_DROP,
    JUN_DATES,
)
from cdc_ml.dataset.constants import TIMESLOTS, TIMEZONE

app = typer.Typer()


def to_lesson_timestamp(series: pd.Series) -> pd.Series:
    # extract the date and time patterns for the 2 types of booking info
    ext1 = series.str.extract(RECORDS_DATE_PATTERN1)
    ext2 = series.str.extract(RECORDS_DATE_PATTERN2)
    # standardize the two type of time format
    date1 = pd.to_datetime(ext1[0], format="%Y-%m-%d", errors="coerce")
    date2 = pd.to_datetime(ext2[0], format="%d/%m/%Y", errors="coerce")
    full_date = date1.combine_first(date2)

    # only keep the lesson starting time since lesson duration is fixed
    full_time = ext1[1].combine_first(ext2[1]).str.split("-").str[0].str.strip()

    # localize to singapore time
    full_date_time = pd.to_datetime(
        full_date.astype("str") + " " + full_time, format="%Y-%m-%d %H:%M"
    ).dt.tz_localize(TIMEZONE)
    return full_date_time


def to_booking_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.tz_convert(TIMEZONE)


def flatten_name_dic(name_dic: dict[str, list[str]]) -> dict[str, str]:
    flattened = {}
    for name, name_list in name_dic.items():
        for alternate_name in name_list:
            if alternate_name not in flattened:
                flattened[alternate_name] = name

    return flattened


def normalize_username(series: pd.Series) -> pd.Series:
    normalized_username = series.str.lower()
    flattened_name_dic = flatten_name_dic(NAME_DIC)
    normalized_username = normalized_username.replace(flattened_name_dic)
    return normalized_username


def handle_special_customers(df: pd.DataFrame) -> pd.DataFrame:
    # jun is the admin and also the customer , only include actual booking as a customer
    jun_proper_booking = pd.to_datetime(JUN_DATES).date

    jun_to_filter = (df["username"] != "jun") | (
        df["booking_timestamp"].dt.date.isin(jun_proper_booking)
    )

    df = df.loc[jun_to_filter]
    return df


def drop_special_records(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_special_customers(df)
    df = df[~df["username"].isin(NAMES_TO_DROP)]
    return df


def fetch_df(raw_out_path: Path) -> None:
    engine = create_engine(DATABASE_URL)
    cutoff = pd.Timestamp("2026-05-01", tz=TIMEZONE)  # 2026-05-01 00:00 SGT
    df = pd.read_sql(
        text("SELECT * FROM records WHERE created_at < :cutoff"),
        engine,
        params={"cutoff": cutoff},
    )
    logger.info(f"Retrieved {len(df)} rows x {len(df.columns)} columns")
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_out_path, index=False)
    logger.success(f"Saved raw records data to {raw_out_path}")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info(f"Pre-Clean: {len(df)} rows")

    df["lesson_timestamp"] = to_lesson_timestamp(df["booking"])
    df["booking_timestamp"] = to_booking_timestamp(df["created_at"])

    time_to_include = pd.to_datetime(pd.Series(TIMESLOTS), format="%H:%M").dt.time
    df = df[df["lesson_timestamp"].dt.time.isin(time_to_include)]
    logger.info(f"Main customers only: {len(df)} row")

    df["username"] = normalize_username(df["username"])
    df = drop_special_records(df)
    logger.info(f"Post-Clean: {len(df)} rows")

    return CleanedRecords.validate(df, lazy=True)


def clean_from_disk(raw_input_path: Path, interim_output_path: Path) -> None:
    df = pd.read_csv(raw_input_path)
    df = clean_df(df)
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path, index=False)


@app.command()
def fetch(raw_out_path: Path = RAW_DATA_DIR / "records.csv"):
    fetch_df(raw_out_path)


@app.command()
def clean(
    raw_input_path: Path = RAW_DATA_DIR / "records.csv",
    interim_output_path: Path = INTERIM_DATA_DIR / "records.parquet",
):
    clean_from_disk(raw_input_path, interim_output_path)


@app.command()
def run(
    raw_path: Path = RAW_DATA_DIR / "records.csv",
    interim_output_path: Path = INTERIM_DATA_DIR / "records.parquet",
):
    logger.info("Fetching data from Neon...")
    fetch_df(raw_path)
    logger.info("Cleaning records...")
    clean_from_disk(raw_path, interim_output_path)
    logger.success(f"Records cleaned and saved to {interim_output_path}")


if __name__ == "__main__":
    app()
