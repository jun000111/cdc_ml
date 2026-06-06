from pathlib import Path
from loguru import logger
import typer
import pandas as pd
import numpy as np
from cdc_ml.features.schema import FinalSet

from cdc_ml.config import (
    POLLS_PROCESSED,
    PREFERENCE_PROCESSED,
    CUSTOMER_CLASS_PROCESSED,
    STAGE_1_PROCESSED,
)

app = typer.Typer()


def assign_pref(df: pd.DataFrame, df_pref: pd.DataFrame) -> pd.DataFrame:

    # ----------------------------
    # 1. Melt preference table
    # ----------------------------
    timeslots = ["t_0830", "t_1020", "t_1245", "t_1435", "t_1625", "t_1850", "t_2040"]

    pref_long = (
        df_pref.melt(
            id_vars=["id", "username", "day_of_week", "date"],
            value_vars=timeslots,
            var_name="timeslot",
            value_name="selected",
        )
        .query("selected == 1")
        .drop_duplicates(subset=["id", "day_of_week", "timeslot"])
    )

    # ----------------------------
    # 2. Core preference stats
    # ----------------------------
    coverage = pref_long.groupby("id").size().rename("pref_coverage").reset_index()

    pref_unique_day = (
        df_pref.groupby("id")["day_of_week"].nunique().rename("pref_unique_day").reset_index()
    )

    pref_unique_timeslot = (
        pref_long.groupby("id")["timeslot"].nunique().rename("pref_unique_timeslot").reset_index()
    )

    # ----------------------------
    # 3. Day-of-week distribution
    # ----------------------------
    pref_dow_wide = (
        pref_long.groupby(["id", "day_of_week"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(7), fill_value=0)
        .add_prefix("pref_dow_count_")
        .reset_index()
    )

    # ----------------------------
    # 4. Merge all features
    # ----------------------------
    df = (
        df.merge(coverage, on="id", how="left")
        .merge(pref_unique_day, on="id", how="left")
        .merge(pref_unique_timeslot, on="id", how="left")
        .merge(pref_dow_wide, on="id", how="left")
    )

    id_dates = df_pref.groupby("id")["date"].agg(list).reset_index()

    df = df.merge(id_dates, on="id", how="left")
    df["pref_valid"] = df.apply(
        lambda row: sum(x > row["polling_at"] for x in row["date"]), axis=1
    )

    return FinalSet.validate(df)


def get_whale_users(poll_threshold: float = 0.5, booking_threshold: float = 0.5):
    """
    Returns users who collectively account for the top X% of
    total poll volume (pc) and total bookings (pt).

    Uses cumulative concentration (Pareto-style), not per-user percentile.

    poll count , positive total

    """
    df = pd.read_parquet(POLLS_PROCESSED)

    # --- Poll count whales ---
    poll_counts = df.groupby("username").size().sort_values(ascending=False)
    cumulative_polls = poll_counts.cumsum().shift(fill_value=0)
    whales_pc_users = poll_counts[cumulative_polls < poll_threshold * poll_counts.sum()].index

    # --- Booking count whales ---
    positive_total = df.groupby("username")["has_booking"].sum().sort_values(ascending=False)
    cumulative_bookings = positive_total.cumsum().shift(fill_value=0)
    whales_pt_users = positive_total[
        cumulative_bookings < booking_threshold * positive_total.sum()
    ].index

    return whales_pc_users, whales_pt_users


def assign_class_type(df_poll: pd.DataFrame, df_class: pd.DataFrame):
    """left join the records with the class type"""
    df = df_poll.merge(df_class, on="username", how="left")
    df["class_type"] = df["class_type"].map(lambda x: 1 if x == "3a" else 0).astype(int)
    return df


# def drop_meta_high_card_cols(df: pd.DataFrame) -> pd.DataFrame:
#     return df.drop(columns=["username", "cycle_start", "cycle_end", "polling_at"])


def build_features(
    df_poll: pd.DataFrame, df_preference: pd.DataFrame, df_class: pd.DataFrame
) -> pd.DataFrame:
    df = assign_class_type(df_poll, df_class)
    df = assign_pref(df, df_preference)
    df = df.sort_values(by="cycle_start")
    return df


def build_on_disk(
    df_poll_input_path: Path = POLLS_PROCESSED,
    df_pref_input_path: Path = PREFERENCE_PROCESSED,
    df_class_input_path: Path = CUSTOMER_CLASS_PROCESSED,
    features_output_path: Path = STAGE_1_PROCESSED,
):

    logger.info("Building features")
    df_poll = pd.read_parquet(df_poll_input_path)
    df_pref = pd.read_parquet(df_pref_input_path)
    df_class = pd.read_parquet(df_class_input_path)
    df = build_features(df_poll, df_pref, df_class)
    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_output_path, index=False)
    logger.info(f"Build features with {len(df)} rows x {len(df.columns)}")
    logger.success(f"Built and saved to {features_output_path}")


@app.command()
def run(
    df_poll_input_path: Path = POLLS_PROCESSED,
    df_pref_input_path: Path = PREFERENCE_PROCESSED,
    df_class_input_path: Path = CUSTOMER_CLASS_PROCESSED,
    features_output_path: Path = STAGE_1_PROCESSED,
) -> None:
    build_on_disk(
        df_poll_input_path,
        df_pref_input_path,
        df_class_input_path,
        features_output_path,
    )


if __name__ == "__main__":
    run(
        df_poll_input_path=POLLS_PROCESSED,
        df_pref_input_path=PREFERENCE_PROCESSED,
        df_class_input_path=CUSTOMER_CLASS_PROCESSED,
    )
