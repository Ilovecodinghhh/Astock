import pandas as pd
from pathlib import Path

from scripts.astock_strategy_scan import evaluate_strategies, parse_args


def test_evaluate_strategies_returns_ranked_candidates() -> None:
    dates = pd.bdate_range("2024-01-02", periods=90)
    rows = []
    for code, drift in [("sh.600000", 0.01), ("sz.000001", -0.002), ("sz.000002", 0.004)]:
        price = 10.0
        for date in dates:
            previous = price
            price = price * (1.0 + drift)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": previous,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000000.0,
                    "amount": 100000000.0,
                    "turn": 2.0,
                    "pctChg": drift * 100,
                    "tradestatus": 1,
                    "isST": 0,
                }
            )
    panel = pd.DataFrame(rows)

    ranked = evaluate_strategies(
        panel,
        top_n_values=[1],
        execution_lag_days_values=[0, 1],
        rebalance_interval_days_values=[1, 2],
        start_date="2024-01-02",
        end_date="2024-05-31",
    )

    assert ranked
    assert {
        "strategy",
        "top_n",
        "risk_filter",
        "execution_lag_days",
        "rebalance_interval_days",
        "cagr_pct",
        "max_drawdown_pct",
        "score",
    }.issubset(ranked[0])
    assert {row["risk_filter"] for row in ranked} == {"none", "risk_on", "risk_on_strict"}
    assert {row["execution_lag_days"] for row in ranked} == {0, 1}
    assert {row["rebalance_interval_days"] for row in ranked} == {1, 2}
    assert {
        "absolute_trend_score",
        "relative_strength_quality_score",
        "risk_adjusted_trend_score",
        "steady_uptrend_score",
        "trend_pullback_score",
        "squeeze_breakout_score",
        "balanced_core_score",
        "defensive_core_score",
        "valuation_quality_score",
        "fundamental_quality_score",
        "growth_value_score",
        "value_trend_score",
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
        "turnover_accumulation_score",
        "value_reversal_score",
        "small_mid_momentum_score",
        "limit_followthrough_score",
        "industry_rotation_score",
        "industry_leader_score",
        "industry_reversal_score",
    }.issubset({row["strategy"] for row in ranked})


def test_parse_args_accepts_fundamentals_file(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "astock_strategy_scan.py",
            "--data-dir",
            "user_data/astock_baostock_wide_500",
            "--fundamentals-file",
            "user_data/astock_fundamentals.feather",
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2026-05-31",
        ],
    )

    args = parse_args()

    assert args.fundamentals_file == Path("user_data/astock_fundamentals.feather")
