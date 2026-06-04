from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from cdc_ml.config import (
    POLLS_PROCESSED,
    PREFERENCE_PROCESSED,
    CUSTOMER_CLASS_PROCESSED,
    STAGE_1_PROCESSED,
)

app = typer.Typer()


def assign_class_type(df_poll: pd.DataFrame, df_class: pd.DataFrame):
    """left join the records with the class type"""
    df = df_poll.merge(df_class, on="username", how="left")
    df["class_type"] = df["class_type"].map(lambda x: 1 if x == "3a" else 0).astype(int)
    return df


def drop_meta_high_card_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["username", "cycle_start", "cycle_end", "polling_at"])


def build_features(
    df_poll: pd.DataFrame, df_preference: pd.DataFrame, df_class: pd.DataFrame
) -> pd.DataFrame:
    df = assign_class_type(df_poll, df_class)
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
