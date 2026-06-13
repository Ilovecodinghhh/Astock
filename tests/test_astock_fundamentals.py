import pandas as pd

from scripts.astock_fundamentals import (
    combine_report_frames,
    merge_fundamentals,
    normalize_fundamental_frame,
    report_available_date,
)


def test_report_available_date_uses_conservative_quarter_lag() -> None:
    assert report_available_date(2023, 1) == pd.Timestamp("2023-05-01")
    assert report_available_date(2023, 2) == pd.Timestamp("2023-09-01")
    assert report_available_date(2023, 3) == pd.Timestamp("2023-11-01")
    assert report_available_date(2023, 4) == pd.Timestamp("2024-05-01")


def test_report_available_date_never_precedes_publication_date() -> None:
    assert report_available_date(2023, 1, pub_date="2023-04-20") == pd.Timestamp("2023-05-01")
    assert report_available_date(2023, 1, pub_date="2023-05-12") == pd.Timestamp("2023-05-12")


def test_normalize_fundamental_frame_parses_metrics_and_availability() -> None:
    frame = pd.DataFrame(
        {
            "code": ["sh.600000", "sh.600000"],
            "pubDate": ["2024-04-30", "2024-04-30"],
            "statDate": ["2023-12-31", "2023-12-31"],
            "roeAvg": ["0.051598", "0.051598"],
            "YOYNI": ["-0.280170", "-0.280170"],
        }
    )

    normalized = normalize_fundamental_frame(frame)

    assert normalized["code"].tolist() == ["sh.600000"]
    assert normalized.loc[0, "report_year"] == 2023
    assert normalized.loc[0, "report_quarter"] == 4
    assert normalized.loc[0, "available_date"] == pd.Timestamp("2024-05-01")
    assert normalized.loc[0, "roeAvg"] == 0.051598
    assert normalized.loc[0, "YOYNI"] == -0.280170


def test_combine_report_frames_keeps_metrics_and_latest_publication_date() -> None:
    profit = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "pubDate": ["2024-04-20"],
            "statDate": ["2023-12-31"],
            "roeAvg": ["0.12"],
        }
    )
    growth = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "pubDate": ["2024-04-30"],
            "statDate": ["2023-12-31"],
            "YOYNI": ["0.25"],
        }
    )

    combined = combine_report_frames([profit, growth])

    assert combined.loc[0, "pubDate"] == "2024-04-30"
    assert combined.loc[0, "roeAvg"] == "0.12"
    assert combined.loc[0, "YOYNI"] == "0.25"


def test_merge_fundamentals_uses_only_available_reports() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-30", "2024-05-01", "2024-05-02"]),
            "code": ["sh.600000"] * 3,
            "close": [10.0, 10.1, 10.2],
        }
    )
    fundamentals = normalize_fundamental_frame(
        pd.DataFrame(
            {
                "code": ["sh.600000"],
                "pubDate": ["2024-04-30"],
                "statDate": ["2023-12-31"],
                "roeAvg": ["0.051598"],
            }
        )
    )

    merged = merge_fundamentals(panel, fundamentals)
    by_date = merged.set_index("date")

    assert pd.isna(by_date.loc[pd.Timestamp("2024-04-30"), "roeAvg"])
    assert by_date.loc[pd.Timestamp("2024-05-01"), "roeAvg"] == 0.051598
    assert by_date.loc[pd.Timestamp("2024-05-02"), "roeAvg"] == 0.051598
