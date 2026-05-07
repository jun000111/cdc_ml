from pathlib import Path

from loguru import logger
import pandas as pd
import typer
from sqlalchemy import create_engine, text


from cdc_ml.config import (
    RAW_DATA_DIR,
    DATABASE_URL,
    INTERIM_DATA_DIR,
    INTERIM_RECORDS_PARQUET,
    RAW_RECORDS_CSV,
)
from cdc_ml.datasets.records.schema import CleanedRecords

from cdc_ml.datasets.records.constants import (
    RECORDS_DATE_PATTERN1,
    RECORDS_DATE_PATTERN2,
    NAME_DIC,
    NAMES_TO_DROP,
    JUN_DATES,
)
from cdc_ml.datasets.constants import TIMESLOTS, TIMEZONE

app = typer.Typer()


def to_lesson_at(series: pd.Series) -> pd.Series:
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


def to_booking_at(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.tz_convert(TIMEZONE)


def flatten_name_dic(name_dic: dict[str, list[str]]) -> dict[str, str]:
    # the source dic have to be flattened out to be useable and pass into pandas
    flattened = {}
    for name, name_list in name_dic.items():
        for alternate_name in name_list:
            if alternate_name not in flattened:
                flattened[alternate_name] = name

    return flattened


def normalize_username(series: pd.Series) -> pd.Series:
    """standardize and normalize the usernames

    including convert username to lowercase and also uniform all alternate customers names
    """
    normalized_username = series.str.lower()
    flattened_name_dic = flatten_name_dic(NAME_DIC)

    # some customers have alternate names , merge them as one
    normalized_username = normalized_username.replace(flattened_name_dic)
    return normalized_username


def handle_special_customers(df: pd.DataFrame) -> pd.DataFrame:
    """some records are due to testing and is not valid , some records are weird and should not be included"""
    jun_proper_booking = pd.to_datetime(JUN_DATES).date

    jun_to_exclude = (df["username"] == "jun") & ~(
        df["booking_at"].dt.date.isin(jun_proper_booking)
    )
    pand_to_exclude = (df["username"] == "jy") & df["booking_at"].dt.date.isin(
        pd.to_datetime(["2025-09-06"]).date
    )

    brendon_to_exclude = (df["username"] == "brendon") & df["booking_at"].dt.date.isin(
        pd.to_datetime(["2025-11-05"]).date
    )

    issac_to_exclude = (df["username"] == "issac") & df["booking_at"].dt.date.isin(
        pd.to_datetime(["2025-12-03"]).date
    )
    df = df.loc[~jun_to_exclude & ~pand_to_exclude & ~brendon_to_exclude & ~issac_to_exclude]
    return df


def drop_special_records(df: pd.DataFrame) -> pd.DataFrame:
    """drop some weird and special records due to human/machine error"""
    df = handle_special_customers(df)

    # drop the rest of the records under testing/invalid names
    df = df[~df["username"].isin(NAMES_TO_DROP)]
    return df


def filter_main_customers(timeslots: list, df: pd.DataFrame):
    """only include records that is in the TIMESLOTS , these are the main customers that the model will be working on"""
    time_to_include = pd.to_datetime(pd.Series(timeslots), format="%H:%M").dt.time
    return df.loc[df["lesson_at"].dt.time.isin(time_to_include)]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """assign booking type 0 to all proper bookings
    drop irrelevant/modified columns"""
    return df.assign(booking_type=0).drop(columns=["booking", "created_at", "id"])


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info(f"Pre-Clean: {len(df)} rows")

    # convert both booking and created_at columns to standard tz aware timestamp format
    df["lesson_at"] = to_lesson_at(df["booking"])
    df["booking_at"] = to_booking_at(df["created_at"])

    df = filter_main_customers(TIMESLOTS, df)
    logger.info(f"Main customers only: {len(df)} row")

    df["username"] = normalize_username(df["username"])

    df = drop_special_records(df)
    df = clean_columns(df)
    logger.info(f"Post-Clean: {len(df)} rows")

    return CleanedRecords.validate(df, lazy=True)


def clean_from_disk(raw_input_path: Path, interim_output_path: Path) -> None:
    logger.info("Cleaning records...")
    df = pd.read_csv(raw_input_path)
    df_cleaned = clean_df(df)
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(interim_output_path, index=False)
    logger.success(f"Records cleaned and saved to {interim_output_path}")


def fetch_from_disk(raw_out_path: Path) -> None:
    logger.info("Fetching data from Neon...")
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


@app.command()
def fetch(raw_out_path: Path = RAW_RECORDS_CSV):
    fetch_from_disk(raw_out_path)


@app.command()
def clean(
    raw_input_path: Path = RAW_RECORDS_CSV, interim_output_path: Path = INTERIM_RECORDS_PARQUET
):
    clean_from_disk(raw_input_path, interim_output_path)


@app.command()
def run(
    raw_input_path: Path = RAW_RECORDS_CSV, interim_output_path: Path = INTERIM_RECORDS_PARQUET
):
    fetch_from_disk(raw_input_path)
    clean_from_disk(raw_input_path, interim_output_path)


if __name__ == "__main__":
    app()
