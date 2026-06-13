from __future__ import annotations

import pandas as pd

from scripts.astock_metadata import merge_metadata, normalize_basic_frame, normalize_industry_frame


def test_normalize_industry_frame_keeps_clean_group_labels() -> None:
    frame = pd.DataFrame(
        {
            "code": ["sh.600000", "sz.000001"],
            "industry": ["J66货币金融服务", ""],
            "industryClassification": ["证监会行业分类", "证监会行业分类"],
        }
    )

    normalized = normalize_industry_frame(frame)

    assert normalized.to_dict("records") == [
        {"code": "sh.600000", "industry": "J66货币金融服务", "industryClassification": "证监会行业分类"},
        {"code": "sz.000001", "industry": "unknown", "industryClassification": "证监会行业分类"},
    ]


def test_normalize_basic_frame_parses_ipo_dates() -> None:
    frame = pd.DataFrame(
        {
            "code": ["sh.600000", "sz.000001"],
            "ipoDate": ["1999-11-10", ""],
            "outDate": ["", ""],
            "status": ["1", "1"],
        }
    )

    normalized = normalize_basic_frame(frame)

    assert normalized.loc[0, "ipoDate"] == pd.Timestamp("1999-11-10")
    assert pd.isna(normalized.loc[1, "ipoDate"])


def test_merge_metadata_adds_industry_and_ipo_without_dropping_panel_rows() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "code": ["sh.600000", "sz.000001"],
            "close": [10.0, 11.0],
        }
    )
    metadata = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "industry": ["J66货币金融服务"],
            "ipoDate": [pd.Timestamp("1999-11-10")],
        }
    )

    merged = merge_metadata(panel, metadata)

    assert merged["code"].tolist() == ["sh.600000", "sz.000001"]
    assert merged.loc[0, "industry"] == "J66货币金融服务"
    assert merged.loc[1, "industry"] == "unknown"
