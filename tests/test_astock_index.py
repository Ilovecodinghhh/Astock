from __future__ import annotations

import pandas as pd

from scripts.astock_index import add_index_trend_filter, normalize_index_frame


def test_normalize_index_frame_converts_dates_and_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "code": ["sh.000300"],
            "close": ["10.5"],
            "pctChg": ["1.2"],
        }
    )

    normalized = normalize_index_frame(frame)

    assert normalized.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert normalized.loc[0, "close"] == 10.5
    assert normalized.loc[0, "pctChg"] == 1.2


def test_add_index_trend_filter_lags_trend_by_one_trading_day() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "code": ["sh.600000"] * 4,
            "close": [10.0, 10.1, 10.2, 10.3],
        }
    )
    index_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )

    filtered = add_index_trend_filter(panel, index_frame, short_window=2, long_window=3)

    by_date = filtered.groupby("date")["index_risk_on"].first()
    assert bool(by_date.loc[pd.Timestamp("2024-01-04")]) is False
    assert bool(by_date.loc[pd.Timestamp("2024-01-05")]) is True
