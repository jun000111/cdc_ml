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
SALT = os.environ["SALT"]

PROPER_RECORDS_RAW = RAW_DATA_DIR / "proper_records.csv"
PROPER_RECORDS_INTERIM = INTERIM_DATA_DIR / "proper_records.parquet"

PSEUDO_BOOKINGS_EXTERNAL = EXTERNAL_DATA_DIR / "pseudo_records.xlsx"
PSEUDO_BOOKINGS_INTERIM = INTERIM_DATA_DIR / "pseudo_records.parquet"

BOOKING_CYCLES_EXTERNAL = EXTERNAL_DATA_DIR / "booking_cycles.xlsx"
BOOKING_CYCLES_INTERIM = INTERIM_DATA_DIR / "booking_cycles.parquet"

CUSTOMER_CLASS_RAW = RAW_DATA_DIR / "customer_class.csv"
CUSTOMER_CLASS_INTERIM = INTERIM_DATA_DIR / "customer_class.parquet"

PREFERENCE_EXTERNAL = EXTERNAL_DATA_DIR / "preference.xlsx"
PREFERENCE_PROCESSED = PROCESSED_DATA_DIR / "preference.parquet"

RECORDS_PROCESSED = PROCESSED_DATA_DIR / "records.parquet"
POLLS_PROCESSED = PROCESSED_DATA_DIR / "polls.parquet"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
