import pandera.pandas as pa
import pandas as pd
from pandera.typing import Series

from cdc_ml.datasets.constants import TIMEZONE


class CleanedPreference(pa.DataFrameModel):
    pref_end: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={
            "tz": str(TIMEZONE),
        },
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
    )

    @pa.dataframe_check
    def start_before_end(cls, df: pd.DataFrame) -> pd.Series:
        return df["pref_start"] <= df["pref_end"]

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
