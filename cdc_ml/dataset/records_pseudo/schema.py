import pandera.pandas as pa
import pandas as pd
from pandera.typing import Series

from cdc_ml.dataset.constants import TIMEZONE


class RefinedRecords(pa.DataFrameModel):

    lesson_timestamp: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        nullable=True,
    )
    booking_timestamp: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        nullable=True,
    )

    @pa.dataframe_check
    def booking_before_lesson(cls, df: pd.DataFrame):
        mask = df["lesson_timestamp"].notna() & df["booking_timestamp"].notna()
        results = pd.Series(True, index=df.index)
        results[mask] = df.loc[mask, "booking_timestamp"] < df.loc[mask, "lesson_timestamp"]
        return results
