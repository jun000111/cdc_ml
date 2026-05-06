import pandera.pandas as pa
import pandas as pd
from pandera.typing import Series

from cdc_ml.datasets.constants import TIMEZONE


class CleanedPseudo(pa.DataFrameModel):
    lesson_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
        nullable=True,
    )

    booking_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
    )

    @pa.dataframe_check
    def booking_before_lesson(cls, df: pd.DataFrame) -> pd.Series:
        mask = df["booking_at"].notna() & df["lesson_at"].notna()
        results = pd.Series(True, index=df.index)
        results[mask] = df.loc[mask, "booking_at"] <= df.loc[mask, "lesson_at"]
        return results

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
