from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from cdc_ml.datasets.customer_class.schema import CleanedClass

import typer

from cdc_ml.config import (
    DATABASE_URL,
    CUSTOMER_CLASS_RAW,
    RECORDS_PROCESSED,
    CUSTOMER_CLASS_INTERIM,
)
from cdc_ml.datasets.customer_class.constants import (
    INVALID_CLASS,
    CUSTOMER_CLASS_DIC,
    CLASS_MAPPING,
)

app = typer.Typer()


def flatten_dic(dic: dict[str, list[str]]):
    flattened = {}
    for c, alter_list in dic.items():
        for alter in alter_list:
            if alter not in flattened:
                flattened[alter] = c
    return flattened


def filter_and_keep(df_class: pd.DataFrame) -> pd.DataFrame:
    """only keep valid class like 3 and 3a"""
    # keep_main_customers = df_class["username"].isin(df_records["username"])
    keep_valid_class = ~df_class["class_type"].isin(INVALID_CLASS)
    # return df_class.loc[keep_main_customers & keep_valid_class]
    return df_class.loc[keep_valid_class]


def normalize_columns(df_class: pd.DataFrame) -> pd.DataFrame:
    df_class = df_class.rename(columns={"nickname": "username", "course_type": "class_type"})
    df_class["username"] = df_class["username"].str.lower()
    df_class = df_class.drop_duplicates(["username"])
    return df_class


def add_non_standard_records(df_class: pd.DataFrame) -> pd.DataFrame:
    """add some of the records that was not in the database"""
    add_classes = pd.DataFrame(
        list(CUSTOMER_CLASS_DIC.items()), columns=["username", "class_type"]
    )

    return pd.concat([df_class, add_classes], axis=0, ignore_index=True)


def modify_class_type(df_class: pd.DataFrame) -> pd.DataFrame:
    """add one team bool column"""
    assigned_one_team = df_class.assign(
        is_one_team=df_class["class_type"]
        .str.contains(r"\bone\s*team\b", case=False, na=False)
        .astype(int)
    )
    assigned_one_team["class_type"] = assigned_one_team["class_type"].replace(
        flatten_dic(CLASS_MAPPING)
    )
    return assigned_one_team


def clean_df(df_class: pd.DataFrame):

    df_class = normalize_columns(df_class)

    df_cleaned = filter_and_keep(df_class)

    df_cleaned = add_non_standard_records(df_cleaned)

    df_cleaned = modify_class_type(df_cleaned)

    return CleanedClass.validate(df_cleaned)


def fetch_from_disk(
    raw_output_path: Path = CUSTOMER_CLASS_RAW,
):
    logger.info("Fetching data from Neon...")
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("select nickname, course_type from customers", engine)
    logger.info(f"Fetched {len(df)} rows x {len(df.columns)}")
    df.to_csv(raw_output_path, index=False)
    logger.success(f"Saved to {raw_output_path} ")


def clean_from_disk(
    raw_input_path: Path = CUSTOMER_CLASS_RAW,
    interim_output_path: Path = CUSTOMER_CLASS_INTERIM,
):

    df_class = pd.read_csv(raw_input_path)
    df = clean_df(df_class)

    logger.info(f"Total: {len(df)} rows x {len(df.columns)}")
    interim_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_output_path)
    logger.success(f"Generated customer class at {interim_output_path}")


@app.command()
def clean(
    raw_input_path: Path = CUSTOMER_CLASS_RAW,
    interim_output_path: Path = CUSTOMER_CLASS_INTERIM,
):
    clean_from_disk(raw_input_path, interim_output_path)


@app.command()
def fetch(
    raw_output_path: Path = CUSTOMER_CLASS_RAW,
):
    fetch_from_disk(raw_output_path)


@app.command()
def run(
    raw_output_path: Path = CUSTOMER_CLASS_RAW,
    raw_input_path: Path = CUSTOMER_CLASS_RAW,
    interim_output_path: Path = CUSTOMER_CLASS_INTERIM,
):
    fetch_from_disk(raw_output_path)
    clean_from_disk(
        raw_input_path,
        interim_output_path,
    )


if __name__ == "__main__":
    app()
