from pathlib import Path

from loguru import logger
from tqdm import tqdm
from sqlalchemy import create_engine
import pandas as pd
import typer

from cdc_ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, DATABASE_URL

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = RAW_DATA_DIR / "records.csv",
    query="SELECT * FROM records",
    # ----------------------------------------------
):
    logger.info("Connecting to database...")
    engine = create_engine(DATABASE_URL)

    logger.info("Running query...")
    df = pd.read_sql(query, engine)
    logger.info(f"Retrieved {len(df):,} rows  x {len(df.columns)} columns ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    app()
