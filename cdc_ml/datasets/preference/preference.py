from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from cdc_ml.config import (
    PSEUDO_BOOKINGS_EXTERNAL,
    PSEUDO_BOOKINGS_INTERIM,
)
from cdc_ml.datasets.constants import TIMESLOTS, TIMEZONE
from cdc_ml.datasets.pseudo_records.schema import CleanedPseudo

app = typer.Typer()


@app.command()
def run(
    raw_input_path: Path = PSEUDO_BOOKINGS_EXTERNAL,
    interim_output_path: Path = PSEUDO_BOOKINGS_INTERIM,
):
    #clean_from_disk(raw_input_path, interim_output_path)


if __name__ == "__main__":
    app()
