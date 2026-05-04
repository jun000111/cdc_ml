import pandas as pd
import pytest


@pytest.fixture
def sgt():
    return "Asia/Singapore"


@pytest.fixture
def raw_records_df():
    return pd.DataFrame(
        {
            "username": ["00860768", "Jun", "JUN", "etc"],
            "booking": [
                "Date: 2025-08-17 , Time: 10:15 - 11:05",
                "Date: 2025-08-31 , Time: 08:30 - 10:10",
                "Date: 2025-12-12 , Time: 10:20 - 12:00",
                "Slots booked:\nDate: 12/02/2026, Time: 10:20 - 12:00\nTotal Amount: S$ 89.38 Remaining Balance: S$ 312.83",
            ],
            "created_at": [
                "2024-03-10 09:00:00+00:00",
                "2024-03-11 10:00:00+00:00",
                "2024-03-12 11:00:00+00:00",
                "2024-03-13 12:00:00+00:00",
            ],
        }
    )
