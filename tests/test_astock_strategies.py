import pandas as pd

from scripts.astock_strategies import add_strategy_scores


def test_strategy_scores_are_lagged_one_trading_day() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "code": ["sh.600000"] * 4,
            "close": [10.0, 11.0, 12.0, 13.0],
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [10.0, 11.0, 12.0, 13.0],
            "volume": [100.0, 110.0, 120.0, 130.0],
            "amount": [100000000.0, 110000000.0, 120000000.0, 130000000.0],
            "turn": [1.0, 1.1, 1.2, 1.3],
            "pctChg": [0.0, 10.0, 9.09, 8.33],
            "tradestatus": [1, 1, 1, 1],
            "isST": [0, 0, 0, 0],
        }
    )

    scored = add_strategy_scores(panel, momentum_window=2, volatility_window=2)

    by_date = scored.set_index("date")
    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-03"), "momentum_lowvol_score"])
    assert by_date.loc[pd.Timestamp("2024-01-05"), "momentum_lowvol_score"] == by_date.loc[
        pd.Timestamp("2024-01-04"), "raw_momentum_lowvol_score"
    ]


def test_strategy_scores_include_multiple_candidate_columns() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "code": ["sh.600000"] * 4,
            "close": [10.0, 10.5, 10.7, 11.2],
            "high": [10.1, 10.6, 10.8, 11.3],
            "low": [9.9, 10.4, 10.6, 11.0],
            "volume": [100.0, 150.0, 130.0, 180.0],
            "amount": [100000000.0, 150000000.0, 130000000.0, 180000000.0],
            "turn": [1.0, 1.5, 1.3, 1.8],
            "pctChg": [0.0, 5.0, 1.9, 4.67],
            "tradestatus": [1, 1, 1, 1],
            "isST": [0, 0, 0, 0],
            "peTTM": [20.0, 19.0, 18.0, 17.0],
            "pbMRQ": [2.0, 1.9, 1.8, 1.7],
            "psTTM": [3.0, 2.9, 2.8, 2.7],
            "pcfNcfTTM": [15.0, 14.0, 13.0, 12.0],
        }
    )

    scored = add_strategy_scores(panel, momentum_window=2, volatility_window=2)

    assert {
        "momentum_lowvol_score",
        "trend_strength_score",
        "breakout_score",
        "reversal_quality_score",
        "absolute_trend_score",
        "relative_strength_quality_score",
        "risk_adjusted_trend_score",
        "steady_uptrend_score",
        "trend_pullback_score",
        "squeeze_breakout_score",
        "risk_on",
        "valuation_quality_score",
        "fundamental_quality_score",
        "growth_value_score",
        "value_trend_score",
        "turnover_accumulation_score",
        "value_reversal_score",
        "small_mid_momentum_score",
        "limit_followthrough_score",
        "industry_rotation_score",
            "industry_leader_score",
            "industry_reversal_score",
            "large_lowvol_value_score",
            "low_turnover_trend_score",
            "volatility_contraction_trend_score",
            "large_value_recovery_score",
            "gap_reversal_score",
            "lower_shadow_reversal_score",
            "quiet_high_base_score",
            "value_event_composite_score",
            "defensive_event_composite_score",
            "high_beta_breakout_score",
            "volume_price_acceleration_score",
            "drawdown_reacceleration_score",
            "smallcap_rs_acceleration_score",
            "quiet_value_trend_score",
            "anti_chase_reversal_score",
            "factor_reinforced_score",
            "factor_consensus_score",
        }.issubset(scored.columns)


def test_new_strategy_scores_are_lagged_one_trading_day() -> None:
    dates = pd.bdate_range("2024-01-02", periods=140)
    panel = pd.DataFrame(
        {
            "date": dates,
            "code": ["sh.600000"] * len(dates),
            "close": [10.0 + index * 0.1 for index in range(len(dates))],
            "high": [10.1 + index * 0.1 for index in range(len(dates))],
            "low": [9.9 + index * 0.1 for index in range(len(dates))],
            "volume": [1000000.0 + index * 1000.0 for index in range(len(dates))],
            "amount": [100000000.0 + index * 100000.0 for index in range(len(dates))],
            "turn": [1.0] * len(dates),
            "pctChg": [1.0] * len(dates),
            "tradestatus": [1] * len(dates),
            "isST": [0] * len(dates),
        }
    )

    scored = add_strategy_scores(panel, momentum_window=20, volatility_window=10)

    by_date = scored.set_index("date")
    for raw_column, score_column in [
        ("raw_absolute_trend_score", "absolute_trend_score"),
        ("raw_relative_strength_quality_score", "relative_strength_quality_score"),
        ("raw_risk_adjusted_trend_score", "risk_adjusted_trend_score"),
        ("raw_steady_uptrend_score", "steady_uptrend_score"),
        ("raw_trend_pullback_score", "trend_pullback_score"),
        ("raw_squeeze_breakout_score", "squeeze_breakout_score"),
    ]:
        assert by_date.loc[dates[-1], score_column] == by_date.loc[dates[-2], raw_column]


def test_risk_on_uses_lagged_market_breadth() -> None:
    rows = []
    for code in ["sh.600000", "sz.000001"]:
        for date, close in zip(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), [10.0, 11.0, 12.0]):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "high": close,
                    "low": close,
                    "volume": 100.0,
                    "amount": 100000000.0,
                    "turn": 1.0,
                    "pctChg": 1.0,
                    "tradestatus": 1,
                    "isST": 0,
                }
            )
    panel = pd.DataFrame(rows)

    scored = add_strategy_scores(panel, momentum_window=1, volatility_window=1, breadth_threshold=0.5)

    by_date = scored.groupby("date")["risk_on"].first()
    assert bool(by_date.loc[pd.Timestamp("2024-01-02")]) is False
    assert bool(by_date.loc[pd.Timestamp("2024-01-04")]) is True


def test_strict_risk_on_uses_lagged_market_conditions() -> None:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=25)
    for code in ["sh.600000", "sz.000001"]:
        for index, date in enumerate(dates):
            close = 10.0 + index
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "high": close,
                    "low": close,
                    "volume": 100.0,
                    "amount": 100000000.0,
                    "turn": 1.0,
                    "pctChg": 1.0,
                    "tradestatus": 1,
                    "isST": 0,
                }
            )
    panel = pd.DataFrame(rows)

    scored = add_strategy_scores(panel, momentum_window=2, volatility_window=2, breadth_threshold=0.5)

    by_date = scored.groupby("date")["risk_on_strict"].first()
    assert bool(by_date.loc[dates[0]]) is False
    assert bool(by_date.loc[dates[-1]]) is True


def test_recovery_risk_on_uses_lagged_market_conditions() -> None:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=25)
    for code in ["sh.600000", "sz.000001", "sz.000002"]:
        for index, date in enumerate(dates):
            close = 10.0 + index * 0.4
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "high": close,
                    "low": close,
                    "volume": 100.0,
                    "amount": 100000000.0,
                    "turn": 1.0,
                    "pctChg": 1.0,
                    "tradestatus": 1,
                    "isST": 0,
                }
            )
    panel = pd.DataFrame(rows)

    scored = add_strategy_scores(panel, momentum_window=2, volatility_window=2, breadth_threshold=0.5)

    by_date = scored.groupby("date")["risk_on_recovery"].first()
    assert bool(by_date.loc[dates[0]]) is False
    assert bool(by_date.loc[dates[-1]]) is True


def test_conservative_scores_use_lagged_size_proxy_and_exclude_young_listings() -> None:
    dates = pd.bdate_range("2024-01-02", periods=160)
    rows = []
    for code, start_index in [("sh.600000", 0), ("sz.300999", 120)]:
        for index, date in enumerate(dates[start_index:], start=start_index):
            close = 10.0 + index * 0.05
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": 1000000.0,
                    "amount": 100000000.0 + index * 100000.0,
                    "turn": 1.0,
                    "pctChg": 0.5,
                    "tradestatus": 1,
                    "isST": 0,
                }
            )
    panel = pd.DataFrame(rows)

    scored = add_strategy_scores(panel, momentum_window=20, volatility_window=10)

    older = scored.loc[scored["code"] == "sh.600000"].set_index("date")
    young = scored.loc[scored["code"] == "sz.300999"]
    assert older.loc[dates[-1], "float_mkt_value_proxy"] == older.loc[dates[-1], "amount"] / 0.01
    assert older.loc[dates[-1], "balanced_core_score"] == older.loc[dates[-2], "raw_balanced_core_score"]
    assert older.loc[dates[-1], "defensive_core_score"] == older.loc[dates[-2], "raw_defensive_core_score"]
    assert young["balanced_core_score"].dropna().empty


def test_conservative_scores_use_ipo_date_when_available() -> None:
    dates = pd.bdate_range("2024-01-02", periods=160)
    panel = pd.DataFrame(
        {
            "date": dates,
            "code": ["sh.600000"] * len(dates),
            "close": [10.0 + index * 0.05 for index in range(len(dates))],
            "high": [10.1 + index * 0.05 for index in range(len(dates))],
            "low": [9.9 + index * 0.05 for index in range(len(dates))],
            "volume": [1000000.0] * len(dates),
            "amount": [100000000.0] * len(dates),
            "turn": [1.0] * len(dates),
            "pctChg": [0.5] * len(dates),
            "tradestatus": [1] * len(dates),
            "isST": [0] * len(dates),
            "ipoDate": [dates[-30]] * len(dates),
        }
    )

    scored = add_strategy_scores(panel, momentum_window=20, volatility_window=10)

    assert scored["balanced_core_score"].dropna().empty


def test_valuation_and_fundamental_scores_are_lagged_one_trading_day() -> None:
    dates = pd.bdate_range("2024-01-02", periods=160)
    panel = pd.DataFrame(
        {
            "date": dates,
            "code": ["sh.600000"] * len(dates),
            "close": [10.0 + index * 0.05 for index in range(len(dates))],
            "high": [10.1 + index * 0.05 for index in range(len(dates))],
            "low": [9.9 + index * 0.05 for index in range(len(dates))],
            "volume": [1000000.0] * len(dates),
            "amount": [100000000.0] * len(dates),
            "turn": [1.0] * len(dates),
            "pctChg": [0.5] * len(dates),
            "tradestatus": [1] * len(dates),
            "isST": [0] * len(dates),
            "peTTM": [20.0 - index * 0.01 for index in range(len(dates))],
            "pbMRQ": [2.0 - index * 0.001 for index in range(len(dates))],
            "psTTM": [3.0 - index * 0.002 for index in range(len(dates))],
            "pcfNcfTTM": [15.0 - index * 0.01 for index in range(len(dates))],
            "roeAvg": [0.15] * len(dates),
            "npMargin": [0.10] * len(dates),
            "YOYNI": [0.20] * len(dates),
            "YOYAsset": [0.12] * len(dates),
            "CFOToNP": [1.10] * len(dates),
            "liabilityToAsset": [0.45] * len(dates),
            "industry": ["J66货币金融服务"] * len(dates),
        }
    )

    scored = add_strategy_scores(panel, momentum_window=20, volatility_window=10)
    by_date = scored.set_index("date")

    for raw_column, score_column in [
        ("raw_valuation_quality_score", "valuation_quality_score"),
        ("raw_fundamental_quality_score", "fundamental_quality_score"),
        ("raw_growth_value_score", "growth_value_score"),
        ("raw_value_trend_score", "value_trend_score"),
        ("raw_turnover_accumulation_score", "turnover_accumulation_score"),
        ("raw_value_reversal_score", "value_reversal_score"),
        ("raw_small_mid_momentum_score", "small_mid_momentum_score"),
        ("raw_limit_followthrough_score", "limit_followthrough_score"),
        ("raw_industry_rotation_score", "industry_rotation_score"),
        ("raw_industry_leader_score", "industry_leader_score"),
        ("raw_industry_reversal_score", "industry_reversal_score"),
        ("raw_large_lowvol_value_score", "large_lowvol_value_score"),
        ("raw_low_turnover_trend_score", "low_turnover_trend_score"),
        ("raw_volatility_contraction_trend_score", "volatility_contraction_trend_score"),
        ("raw_large_value_recovery_score", "large_value_recovery_score"),
        ("raw_gap_reversal_score", "gap_reversal_score"),
        ("raw_lower_shadow_reversal_score", "lower_shadow_reversal_score"),
        ("raw_quiet_high_base_score", "quiet_high_base_score"),
        ("raw_value_event_composite_score", "value_event_composite_score"),
        ("raw_defensive_event_composite_score", "defensive_event_composite_score"),
        ("raw_high_beta_breakout_score", "high_beta_breakout_score"),
        ("raw_volume_price_acceleration_score", "volume_price_acceleration_score"),
        ("raw_drawdown_reacceleration_score", "drawdown_reacceleration_score"),
        ("raw_smallcap_rs_acceleration_score", "smallcap_rs_acceleration_score"),
        ("raw_quiet_value_trend_score", "quiet_value_trend_score"),
        ("raw_anti_chase_reversal_score", "anti_chase_reversal_score"),
        ("raw_factor_reinforced_score", "factor_reinforced_score"),
        ("raw_factor_consensus_score", "factor_consensus_score"),
    ]:
        score = by_date.loc[dates[-1], score_column]
        raw = by_date.loc[dates[-2], raw_column]
        assert score == raw or (pd.isna(score) and pd.isna(raw))
