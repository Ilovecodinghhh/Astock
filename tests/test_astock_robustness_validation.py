from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.astock_robustness_validation import (
    RobustnessThresholds,
    aggregate_validation_rows,
    codes_from_pool_file,
    filter_panel_to_pool,
    parse_args,
    prepare_validation_panel,
    rank_aggregates,
    run_validation_on_scored_panel,
    rolling_start_dates,
    write_summary,
)


def test_rolling_start_dates_advance_by_month_step() -> None:
    starts = rolling_start_dates("2021-01-01", "2021-10-01", step_months=3)

    assert starts == [
        "2021-01-01",
        "2021-04-01",
        "2021-07-01",
        "2021-10-01",
    ]


def test_aggregate_validation_rows_enforces_concentration_and_rolling_thresholds() -> None:
    rows = [
        {
            "stock_pool": "expanded_206",
            "stock_pool_size": 206,
            "top_n": 3,
            "max_single_stock_weight_pct": 33.33,
            "start_date": "2021-01-01",
            "cagr_pct": 58.0,
            "max_drawdown_pct": 30.0,
            "trading_days": 900,
        },
        {
            "stock_pool": "expanded_206",
            "stock_pool_size": 206,
            "top_n": 3,
            "max_single_stock_weight_pct": 33.33,
            "start_date": "2021-04-01",
            "cagr_pct": 44.0,
            "max_drawdown_pct": 35.0,
            "trading_days": 840,
        },
        {
            "stock_pool": "expanded_206",
            "stock_pool_size": 206,
            "top_n": 1,
            "max_single_stock_weight_pct": 100.0,
            "start_date": "2021-01-01",
            "cagr_pct": 90.0,
            "max_drawdown_pct": 28.0,
            "trading_days": 900,
        },
    ]
    thresholds = RobustnessThresholds(
        min_median_cagr_pct=50.0,
        min_worst_cagr_pct=20.0,
        max_worst_drawdown_pct=45.0,
        max_single_stock_weight_pct=33.34,
    )

    aggregates = aggregate_validation_rows(rows, thresholds)

    diversified = next(row for row in aggregates if row["top_n"] == 3)
    concentrated = next(row for row in aggregates if row["top_n"] == 1)
    assert diversified["passes_strict_validation"] is True
    assert diversified["rolling_window_count"] == 2
    assert diversified["positive_windows"] == 2
    assert diversified["median_cagr_pct"] == 51.0
    assert diversified["worst_cagr_pct"] == 44.0
    assert concentrated["passes_strict_validation"] is False
    assert concentrated["passes_concentration"] is False


def test_rank_aggregates_prioritizes_strict_pass_then_robust_return() -> None:
    rows = [
        {
            "stock_pool": "wide",
            "top_n": 5,
            "passes_strict_validation": False,
            "positive_windows": 4,
            "median_cagr_pct": 80.0,
            "worst_cagr_pct": 30.0,
            "score": 80.0,
        },
        {
            "stock_pool": "wide",
            "top_n": 8,
            "passes_strict_validation": True,
            "positive_windows": 4,
            "median_cagr_pct": 55.0,
            "worst_cagr_pct": 25.0,
            "score": 55.0,
        },
    ]

    ranked = rank_aggregates(rows)

    assert [row["top_n"] for row in ranked] == [8, 5]


def test_write_summary_persists_rows_and_aggregates(tmp_path: Path) -> None:
    output_path = write_summary(
        results_dir=tmp_path,
        candidate={"strategy": "relative_strength_quality_score"},
        thresholds=RobustnessThresholds(),
        rows=[{"stock_pool": "wide", "top_n": 5}],
        aggregates=[{"stock_pool": "wide", "top_n": 5}],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate"]["strategy"] == "relative_strength_quality_score"
    assert payload["rows"] == [{"stock_pool": "wide", "top_n": 5}]
    assert payload["aggregates"] == [{"stock_pool": "wide", "top_n": 5}]
    assert payload["thresholds"]["min_median_cagr_pct"] == 50.0


def test_codes_from_pool_file_strips_comments_and_windows_line_endings(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.txt"
    pool_file.write_bytes(b"sh.600000\r\n# comment\r\nsz.000001\r\n\r\n")

    assert codes_from_pool_file(pool_file) == ["sh.600000", "sz.000001"]


def test_filter_panel_to_pool_keeps_only_requested_codes() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "code": ["sh.600000", "sz.000001", "sz.000002"],
            "close": [10.0, 11.0, 12.0],
        }
    )

    filtered = filter_panel_to_pool(panel, ["sz.000001", "sh.600000"])

    assert filtered["code"].tolist() == ["sh.600000", "sz.000001"]


def test_prepare_validation_panel_merges_metadata_and_index_filter() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "code": ["sh.600000"] * 4,
            "close": [10.0, 10.1, 10.2, 10.3],
            "high": [10.1, 10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1, 10.2],
            "volume": [100.0] * 4,
            "amount": [100000000.0] * 4,
            "turn": [1.0] * 4,
            "pctChg": [0.0, 1.0, 1.0, 1.0],
            "tradestatus": [1] * 4,
            "isST": [0] * 4,
        }
    )
    metadata = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "industry": ["J66货币金融服务"],
            "ipoDate": [pd.Timestamp("1999-11-10")],
        }
    )
    index_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )

    prepared = prepare_validation_panel(
        panel,
        metadata=metadata,
        index_frame=index_frame,
        index_short_window=2,
        index_long_window=3,
    )

    assert prepared.loc[0, "industry"] == "J66货币金融服务"
    assert prepared.loc[0, "ipoDate"] == pd.Timestamp("1999-11-10")
    assert bool(prepared.loc[prepared["date"] == pd.Timestamp("2024-01-05"), "index_risk_on"].iloc[0]) is True


def test_prepare_validation_panel_merges_fundamentals_by_available_date() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-30", "2024-05-01", "2024-05-02"]),
            "code": ["sh.600000"] * 3,
            "close": [10.0, 10.1, 10.2],
            "high": [10.1, 10.2, 10.3],
            "low": [9.9, 10.0, 10.1],
            "volume": [100.0] * 3,
            "amount": [100000000.0] * 3,
            "turn": [1.0] * 3,
            "pctChg": [0.0, 1.0, 1.0],
            "tradestatus": [1] * 3,
            "isST": [0] * 3,
        }
    )
    fundamentals = pd.DataFrame(
        {
            "code": ["sh.600000"],
            "available_date": [pd.Timestamp("2024-05-01")],
            "roeAvg": [0.12],
        }
    )

    prepared = prepare_validation_panel(panel, fundamentals=fundamentals)
    by_date = prepared.set_index("date")

    assert pd.isna(by_date.loc[pd.Timestamp("2024-04-30"), "roeAvg"])
    assert by_date.loc[pd.Timestamp("2024-05-01"), "roeAvg"] == 0.12


def test_run_validation_on_scored_panel_reuses_existing_scores() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    rows = []
    for code, drift, industry in [
        ("sh.600000", 0.5, "bank"),
        ("sz.000001", -0.2, "software"),
    ]:
        price = 10.0
        for index, date in enumerate(dates):
            price = price * (1.0 + drift / 100.0)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": price,
                    "pctChg": drift,
                    "tradestatus": 1,
                    "isST": 0,
                    "turn": 1.0,
                    "amount": 100000000.0,
                    "industry": industry,
                    "custom_score": float(index) if code == "sh.600000" else 0.0,
                }
            )
    scored = pd.DataFrame(rows)

    validation_rows = run_validation_on_scored_panel(
        scored,
        stock_pool="tiny",
        candidate={
            "strategy": "custom_score",
            "risk_filter": "none",
            "execution_lag_days": 0,
            "rebalance_interval_days": 1,
            "group_column": "industry",
            "max_group_weight": 1.0,
            "preserve_target_cash": False,
        },
        top_n_values=[1],
        start_dates=["2024-01-02"],
        end_date="2024-01-31",
    )

    assert validation_rows[0]["strategy"] == "custom_score"
    assert validation_rows[0]["stock_pool_size"] == 2
    assert validation_rows[0]["top_n"] == 1


def test_parse_args_accepts_conservative_portfolio_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "astock_robustness_validation.py",
            "--data-dir",
            "user_data/astock_baostock_wide_500",
            "--metadata-file",
            "user_data/astock_metadata.feather",
            "--fundamentals-file",
            "user_data/astock_fundamentals.feather",
            "--index-file",
            "user_data/astock_indexes/sh.000300-d.feather",
            "--risk-filter",
            "index_risk_on",
            "--group-column",
            "industry",
            "--max-group-weight",
            "0.25",
            "--preserve-target-cash",
        ],
    )

    args = parse_args()

    assert args.metadata_file == Path("user_data/astock_metadata.feather")
    assert args.fundamentals_file == Path("user_data/astock_fundamentals.feather")
    assert args.index_file == Path("user_data/astock_indexes/sh.000300-d.feather")
    assert args.risk_filter == "index_risk_on"
    assert args.group_column == "industry"
    assert args.max_group_weight == 0.25
    assert args.preserve_target_cash is True
