from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from cdc_ml.config import (
    BOOKING_CYCLES_INTERIM,
    RECORDS_PROCESSED,
    POLLS_PROCESSED,
    CUSTOMER_CLASS_INTERIM,
)
from cdc_ml.features.add_derived_features import add_effective_mins

app = typer.Typer()


def build_poll_df(
    username: pd.Series, cycle_start: pd.Series, cycle_end: pd.Series
) -> pd.DataFrame:

    rows = []

    for username, cycle in zip(username, zip(cycle_start, cycle_end)):
        polling_hours = pd.date_range(cycle[0], cycle[1], freq="1h")
        for hour in polling_hours:
            rows.append(
                {
                    "username": username,
                    "cycle_start": cycle[0],
                    "cycle_end": cycle[1],
                    "polling_hour": hour,
                }
            )

    return pd.DataFrame(rows)


# def assign_class_type(df_records: pd.DataFrame, df_class: pd.DataFrame):
#     """left join the records with the class type"""
#     df_customer_records = df_records.merge(df_class, on="username", how="left")
#     df_customer_records["is_one_team"] = df_customer_records["is_one_team"].fillna(0).astype(int)
#     return df_customer_records


def validate_booking_records(df_poll: pd.DataFrame, df_records: pd.DataFrame):

    cycles = df_poll[["username", "cycle_start", "cycle_end"]].drop_duplicates()

    merged = cycles.merge(df_records, on="username", how="left")

    valid = merged.loc[
        (merged["booking_at"] >= merged["cycle_start"])
        & (merged["booking_at"] <= merged["cycle_end"])
    ]
    invalid_records = df_records[~df_records["id"].isin(valid["id"])]
    if not invalid_records.empty:
        logger.error(f"{len(invalid_records)} in records")
        raise ValueError(
            f"{len(invalid_records)} in records"
            f"sample ids: {invalid_records["id"].head().tolist()}"
        )
    logger.info("All booking records are validated against booking cycle...")


def add_lable(df_poll: pd.DataFrame, df_records: pd.DataFrame) -> pd.DataFrame:

    df_floored_polling_hours_cycle = df_poll.assign(
        floored_polling_hour=df_poll["polling_hour"].dt.floor("h")
    )
    df_unique_floored_records = (
        df_records.assign(floored_booking=df_records["booking_at"].dt.floor("h"))[
            ["username", "floored_booking"]
        ]
        .drop_duplicates()
        .assign(has_booking=1)
    )
    df_polling_labled = df_floored_polling_hours_cycle.merge(
        df_unique_floored_records,
        left_on=["username", "floored_polling_hour"],
        right_on=["username", "floored_booking"],
        how="left",
    )

    df_polling_labled["has_booking"] = df_polling_labled["has_booking"].fillna(0).astype(int)
    df_polling_labled = df_polling_labled.drop(columns=["floored_booking", "floored_polling_hour"])
    df_polling_labled.insert(0, "id", range(len(df_polling_labled)))

    return df_polling_labled


def left_join(df_poll: pd.DataFrame, df_class: pd.DataFrame) -> pd.DataFrame:
    return df_poll.merge(df_class, on="username", how="left")


def generate_df(
    df_cycle: pd.DataFrame, df_records: pd.DataFrame, df_class: pd.DataFrame
) -> pd.DataFrame:
    df_cycle = df_cycle.copy()
    df_records = df_records.copy()
    df_poll = build_poll_df(df_cycle["username"], df_cycle["cycle_start"], df_cycle["cycle_end"])
    validate_booking_records(df_poll, df_records)
    df_poll = left_join(df_poll, df_class)
    df_poll = add_lable(df_poll, df_records)
    return df_poll


def generate_on_disk(
    interim_input_cycle_path: Path = BOOKING_CYCLES_INTERIM,
    interim_input_complete_records_path: Path = RECORDS_PROCESSED,
    interim_customer_class_path: Path = CUSTOMER_CLASS_INTERIM,
    processed_output_path: Path = POLLS_PROCESSED,
):

    logger.info("Start...")
    df_cycle = pd.read_parquet(interim_input_cycle_path)
    df_complete_records = pd.read_parquet(interim_input_complete_records_path)
    df_class = pd.read_parquet(interim_customer_class_path)
    df = generate_df(df_cycle, df_complete_records, df_class)
    logger.info(f"Total {len(df)} rows x {len(df.columns)}")
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_output_path, index=False)
    logger.success(f"Feature build complete and saved to {processed_output_path}")


@app.command()
def run(
    interim_input_cycle_path: Path = BOOKING_CYCLES_INTERIM,
    interim_input_complete_records_path: Path = RECORDS_PROCESSED,
    interim_customer_class_path: Path = CUSTOMER_CLASS_INTERIM,
    processed_output_path: Path = POLLS_PROCESSED,
):
    generate_on_disk(
        interim_input_cycle_path,
        interim_input_complete_records_path,
        interim_customer_class_path,
        processed_output_path,
    )


if __name__ == "__main__":
    app()
