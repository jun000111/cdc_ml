from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer

from cdc_ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    raw_input_path: Path = RAW_DATA_DIR / "records.csv",
    raw_ex_input_path: Path = EXTERNAL_DATA_DIR / "pseudo_bookings.csv",
    interim_output_path: Path = INTERIM_DATA_DIR / "cleaned_records.csv",
):
    logger.info("Start Cleaning...")


if __name__ == "__main__":
    app()
