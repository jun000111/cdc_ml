import pandera.pandas as pa
import pandas as pd
from pandera.typing import Series

from cdc_ml.datasets.constants import TIMEZONE


class FinalSet(pa.DataFrameModel):

    pref_coverage: Series[int] = pa.Field(ge=1, le=49, nullable=False)
    pref_unique_day: Series[int] = pa.Field(
        ge=1,
        le=7,
        nullable=False,
    )

    pref_unique_timeslot: Series[int] = pa.Field(
        ge=1,
        le=7,
        nullable=False,
    )

    pref_valid: Series[int] = pa.Field(
        ge=0,
    )
