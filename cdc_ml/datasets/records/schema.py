import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from cdc_ml.datasets.constants import TIMEZONE


class CleanedRecords(pa.DataFrameModel):
    lesson_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"tz": str(TIMEZONE)})
    booking_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
    )

    @pa.dataframe_check
    def booking_before_lesson(cls, df: pd.DataFrame) -> pd.Series:
        return df["booking_at"] < df["lesson_at"]

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
