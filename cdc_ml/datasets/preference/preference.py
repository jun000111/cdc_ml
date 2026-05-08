from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from typing import cast

from cdc_ml.config import RECORDS_PROCESSED, PREFERENCE_EXTERNAL, PREFERENCE_PROCESSED
from cdc_ml.datasets.constants import TIMESLOTS, TIMEZONE
from cdc_ml.datasets.pseudo_records.schema import CleanedPseudo

app = typer.Typer()

DAY_COLS = ("mon", "tues", "wed", "thurs", "fri", "sat", "sun")


def generate_rows(
    id: int,
    username: str,
    pref_s: str,
    pref_e: str,
    cycle_rules: dict[int, list[int]],
    interval="D",
    standard_timelist=TIMESLOTS,
):
    rows = []
    pref_start = pd.Timestamp(pref_s)
    pref_end = pd.Timestamp(pref_e)
    date_range = pd.date_range(pref_start, pref_end, freq=interval)

    # {0:"08:30",1:"10:20",2:"12:45",...}
    timeslot_reference = dict(enumerate(standard_timelist))

    for date in date_range:

        # {"08:30":0,"10:20":0,"12:45":0,...}
        default_multi_hot = dict.fromkeys(TIMESLOTS, 0)

        available_slots = cycle_rules[date.day_of_week]
        if len(available_slots) == 0:
            continue
        for slot in available_slots:
            time = timeslot_reference[slot]
            default_multi_hot[time] = 1

        rows.append(
            {
                "id": id,
                "username": username,
                "day_of_week": date.day_of_week,
                "day_name": date.day_name(),
                "pref_start": pref_start,
                "pref_end": pref_end,
                "date": date,
                **default_multi_hot,
            }
        )
    return rows


def get_int_array(value: str | None) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        return ()
    return tuple(int(x) for x in value.split(","))


def all_records_valid(df_pref: pd.DataFrame, df_records: pd.DataFrame) -> None:

    TIME_COLS = {
        "t_0830": "08:30",
        "t_1020": "10:20",
        "t_1245": "12:45",
        "t_1435": "14:35",
        "t_1625": "16:25",
        "t_1850": "18:50",
        "t_2040": "20:40",
    }

    df_long = df_pref.melt(
        id_vars=["id", "username", "date"],
        value_vars=list(TIME_COLS),
        var_name="slot",
        value_name="enabled",
    ).query("enabled == 1")

    # Build all pref_at timestamps in a single vectorized call
    df_long["pref_at"] = pd.to_datetime(
        df_long["date"].astype(str) + " " + df_long["slot"].map(TIME_COLS)
    ).dt.tz_localize("Asia/Singapore")

    df_pref_time = df_long[["id", "username", "pref_at"]]

    df_merged = df_records.merge(
        df_pref_time,
        left_on=["username", "lesson_at"],
        right_on=["username", "pref_at"],
        how="left",
    )
    df_missing = df_merged.loc[
        (df_merged["pref_at"].isna()) & ~df_merged["lesson_at"].isna(),
        ["username", "lesson_at", "booking_at"],
    ].sort_values(by="username")

    if not df_missing.empty:
        raise ValueError(logger.error(f"{len(df_missing)} records not verified"), logger.error(f"{
                df_missing
            }"))
    else:
        logger.info("All records verified")


def build_pref(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df[list(DAY_COLS)].astype(str).map(get_int_array)

    new_rows = []
    for meta, rules_t in zip(
        df[["id", "username", "pref_start", "pref_end", "interval"]].itertuples(index=False),
        parsed.itertuples(index=False),
    ):
        rules = dict(enumerate(rules_t))
        new_rows.extend(
            generate_rows(
                cast(int, meta.id),
                cast(str, meta.username),
                cast(str, meta.pref_start),
                cast(str, meta.pref_end),
                rules,
                cast(str, meta.interval),
            )
        )
    df = pd.DataFrame(new_rows)
    df = df.rename(
        columns={
            "08:30": "t_0830",
            "10:20": "t_1020",
            "12:45": "t_1245",
            "14:35": "t_1435",
            "16:25": "t_1625",
            "18:50": "t_1850",
            "20:40": "t_2040",
        }
    )
    return df


def clean_df(df_pref: pd.DataFrame, df_records: pd.DataFrame) -> pd.DataFrame:
    df_pref = build_pref(df_pref)
    all_records_valid(df_pref, df_records)
    return df_pref


def clean_from_disk(
    preference_input_path: Path = PREFERENCE_EXTERNAL,
    records_input_path: Path = RECORDS_PROCESSED,
    preference_output_path: Path = PREFERENCE_PROCESSED,
):
    logger.info("Cleaning preference...")
    df_pref = pd.read_excel(preference_input_path)
    df_records = pd.read_parquet(records_input_path)
    df = clean_df(df_pref, df_records)
    logger.info(f"Total {len(df)} rows by {len(df.columns)}")
    preference_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(preference_output_path, index=False)
    logger.success("Done")


@app.command()
def run(
    preference_input_path: Path = PREFERENCE_EXTERNAL,
    records_input_path: Path = RECORDS_PROCESSED,
    preference_output_path: Path = PREFERENCE_PROCESSED,
):
    clean_from_disk(preference_input_path, records_input_path, preference_output_path)


if __name__ == "__main__":
    app()
