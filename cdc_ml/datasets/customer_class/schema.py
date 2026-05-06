import pandera.pandas as pa
import pandas as pd
from pandera.typing import Series


class CleanedClass(pa.DataFrameModel):
    is_one_team: Series[int] = pa.Field(isin=[0, 1])

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
