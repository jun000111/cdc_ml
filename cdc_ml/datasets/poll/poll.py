from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from cdc_ml.config import (
    BOOKING_CYCLES_PROCESSED,
    RECORDS_PROCESSED,
    POLLS_PROCESSED,
)

app = typer.Typer()


def build_poll_df(
    username: pd.Series, cycle_start: pd.Series, cycle_end: pd.Series
) -> pd.DataFrame:

    rows = []

    for username, cycle in zip(username, zip(cycle_start, cycle_end)):
        polling_timestamp = pd.date_range(cycle[0], cycle[1], freq="1h")
        for hour in polling_timestamp:
            rows.append(
                {
                    "username": username,
                    "cycle_start": cycle[0],
                    "cycle_end": cycle[1],
                    "polling_at": hour,
                }
            )

    return pd.DataFrame(rows)


def validate_booking_records(df_poll: pd.DataFrame, df_records: pd.DataFrame):

    cycles = df_poll[["username", "cycle_start", "cycle_end"]].drop_duplicates()

    merged = cycles.merge(df_records, on="username", how="left")

    valid = merged.loc[
        (merged["booking_at"] >= merged["cycle_start"])
        & (merged["booking_at"] <= merged["cycle_end"])
    ]
    invalid_records = df_records[~(df_records["id"].isin(valid["id"]))]
    if not invalid_records.empty:
        logger.error(f"{len(invalid_records)} in records")
        raise ValueError(
            f"{len(invalid_records)} in records"
            f"sample ids: {invalid_records["username"].head().tolist()}"
        )
    logger.info("All booking records are validated against booking cycle...")


def add_lable(df_poll: pd.DataFrame, df_records: pd.DataFrame) -> pd.DataFrame:

    df_records = df_records.assign(
        booking_hour=df_records["booking_at"].dt.floor("h"), has_booking=1
    )
    df_poll["polling_at"] = df_poll["polling_at"].dt.floor("h")
    df_poll = df_poll.merge(
        df_records[["username", "booking_hour", "has_booking"]],
        left_on=["username", "polling_at"],
        right_on=["username", "booking_hour"],
        how="outer",
    )
    df_poll["has_booking"] = df_poll["has_booking"].fillna(0).astype(bool)

    # df_poll = df_poll.assign(has_booking=polling_keys.isin(booking_keys).astype(int))
    df_poll = df_poll.assign(
        cycle_start_month=lambda row: row["cycle_start"].dt.month,
        cycle_start_day=lambda row: row["cycle_start"].dt.day,
        cycle_start_dow=lambda row: row["cycle_start"].dt.day_of_week,
        cycle_start_hour=lambda row: row["cycle_start"].dt.hour,
    )
    df_poll = df_poll.assign(
        polling_month=lambda row: row["polling_at"].dt.month,
        polling_day=lambda row: row["polling_at"].dt.day,
        polling_dow=lambda row: row["polling_at"].dt.day_of_week,
        polling_hour=lambda row: row["polling_at"].dt.hour,
    )

    df_poll = df_poll.assign(
        hours_into_cycle=(df_poll["polling_at"] - df_poll["cycle_start"]).dt.total_seconds() / 3600
    )

    df_poll.insert(0, "id", range(len(df_poll)))

    return df_poll


def generate_df(df_cycle: pd.DataFrame, df_records: pd.DataFrame) -> pd.DataFrame:
    df_cycle = df_cycle.copy()
    df_records = df_records.copy()
    df_poll = build_poll_df(df_cycle["username"], df_cycle["cycle_start"], df_cycle["cycle_end"])
    validate_booking_records(df_poll, df_records)
    # df_poll = left_join(df_poll, df_class)
    df_poll = add_lable(df_poll, df_records)
    return df_poll


def generate_on_disk(
    interim_input_cycle_path: Path = BOOKING_CYCLES_PROCESSED,
    interim_input_complete_records_path: Path = RECORDS_PROCESSED,
    processed_output_path: Path = POLLS_PROCESSED,
):

    logger.info("Start...")
    df_cycle = pd.read_parquet(interim_input_cycle_path)
    df_complete_records = pd.read_parquet(interim_input_complete_records_path)
    df = generate_df(df_cycle, df_complete_records)
    logger.info(f"Total {len(df)} rows x {len(df.columns)}")
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_output_path, index=False)
    logger.success(f"Feature build complete and saved to {processed_output_path}")


@app.command()
def run(
    interim_input_cycle_path: Path = BOOKING_CYCLES_PROCESSED,
    interim_input_complete_records_path: Path = RECORDS_PROCESSED,
    processed_output_path: Path = POLLS_PROCESSED,
):
    generate_on_disk(
        interim_input_cycle_path,
        interim_input_complete_records_path,
        processed_output_path,
    )


if __name__ == "__main__":
    app()
