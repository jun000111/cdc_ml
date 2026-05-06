import pytest
import pandas as pd
from pandas.testing import assert_series_equal

from cdc_ml.datasets.records.records import (
    flatten_name_dic,
    normalize_username,
    to_lesson_at,
    to_booking_at,
    handle_special_customers,
)


class TestFlattenDic:
    def test_empty_input(self):
        assert flatten_name_dic({}) == {}

    def test_empty_alternate_list(self):
        assert flatten_name_dic({"jun": []}) == {}


class TestToLessonTimestamp:
    def test_tz_aware(self, raw_records_df, sgt):
        converted = to_lesson_at(raw_records_df["booking"])

        assert str(converted.dt.tz) == sgt


class TestToBookingTimestamp:
    def test_tz_aware(self, raw_records_df, sgt):
        converted = to_booking_at(raw_records_df["created_at"])

        assert str(converted.dt.tz) == sgt


class TestNormalizeUsername:
    def test_all_lower(self):
        test_series = pd.Series(["AJC", "NNNC", "yO"])
        normalized = normalize_username(test_series)
        sample = test_series.str.lower()

        assert normalized.tolist() == sample.tolist()
