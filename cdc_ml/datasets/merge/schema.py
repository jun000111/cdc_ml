import pandera.pandas as pa
import pandas as pd
from pandera.typing import Series

from cdc_ml.datasets.constants import TIMEZONE


class MergeRecordsPseudo(pa.DataFrameModel):

    lesson_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        nullable=True,
    )
    booking_at: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"tz": str(TIMEZONE)},
        ge=pd.Timestamp("2025-08-01", tz=TIMEZONE),
        le=pd.Timestamp("2026-05-01", tz=TIMEZONE),
        nullable=True,
    )

    @pa.dataframe_check
    def booking_before_lesson(cls, df: pd.DataFrame):
        mask = df["lesson_at"].notna() & df["booking_at"].notna()
        results = pd.Series(True, index=df.index)
        results[mask] = df.loc[mask, "booking_at"] < df.loc[mask, "lesson_at"]
        return results


class MergeRecordsClass(pa.DataFrameModel):
    booking_type: Series[int] = pa.Field(isin=[0, 1, 2])
    class_type: Series[str] = pa.Field(isin=["3", "3a"])
    is_one_team: Series[int] = pa.Field(isin=[0, 1])

    @pa.dataframe_check
    def user_has_same_class_team(cls, df: pd.DataFrame) -> pd.Series:
        counts = df.groupby("username")[["class_type", "is_one_team"]].transform("nunique")
        return (counts <= 1).all(axis=1)
