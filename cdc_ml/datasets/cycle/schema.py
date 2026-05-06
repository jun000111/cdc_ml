import pandera.pandas as pa
import pandas as pd

from pandera.typing import Series
from cdc_ml.datasets.constants import TIMEZONE


class CleanedCycle(pa.DataFrameModel):
    cycle_start: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={
            "tz": str(TIMEZONE),
        },
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
    )

    cycle_end: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={
            "tz": str(TIMEZONE),
        },
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
    )

    @pa.dataframe_check
    def start_before_end(cls, df: pd.DataFrame) -> pd.Series:
        return df["cycle_start"] < df["cycle_end"]

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
