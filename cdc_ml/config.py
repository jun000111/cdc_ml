import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATABASE_URL = os.environ["DATABASE_URL"]

RAW_RECORDS_CSV = RAW_DATA_DIR / "records.csv"
INTERIM_RECORDS_PARQUET = INTERIM_DATA_DIR / "records.parquet"

EXTERNAL_PSEUDO_EXCEL = EXTERNAL_DATA_DIR / "pseudo_bookings.xlsx"
INTERIM_PSEUDO_PARQUET = INTERIM_DATA_DIR / "pseudo.parquet"

INTERIM_COMPLETE_RECORDS_PARQUET = INTERIM_DATA_DIR / "complete_records.parquet"

EXTERNAL_CYCLE_EXCEL = EXTERNAL_DATA_DIR / "booking_cycle.xlsx"
INTERIM_CYCLE_PARQUET = INTERIM_DATA_DIR / "cycle.parquet"

RAW_CLASS_CSV = RAW_DATA_DIR / "customer_class.csv"
INTERIM_CLASS_PARQUET = INTERIM_DATA_DIR / "customer_class.parquet"
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
