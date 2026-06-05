from pathlib import Path
from loguru import logger
import typer
import pandas as pd
import numpy as np

from cdc_ml.config import (
    POLLS_PROCESSED,
    PREFERENCE_PROCESSED,
    CUSTOMER_CLASS_PROCESSED,
    STAGE_1_PROCESSED,
)

app = typer.Typer()

import numpy as np
import pandas as pd


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

    # ----------------------------
    # 5. Time window features
    # ----------------------------
    pref_start_end = df_pref.drop_duplicates("id")[["id", "pref_start", "pref_end"]]
    df = df.merge(pref_start_end, on="id", how="left")

    pref_range = (df["pref_end"] - df["pref_start"]).dt.total_seconds() / 3600
    poll_range = (df["pref_end"] - df["polling_at"]).dt.total_seconds() / 3600

    df["countdown"] = np.minimum(pref_range, poll_range)

    return df


def get_whale_users():
    """Returns polling count and polling total whales"""
    df = pd.read_parquet(POLLS_PROCESSED)
    poll_counts = df.groupby(["username"]).size()
    whales_pc_threshold = poll_counts.quantile(0.8)
    whales_pc_users = poll_counts[poll_counts >= whales_pc_threshold].index

    positive_counts = df.groupby(["username"])["has_booking"].sum()
    whales_pt_threshold = positive_counts.quantile(0.8)
    whales_pt_users = positive_counts[positive_counts >= whales_pt_threshold].index

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
