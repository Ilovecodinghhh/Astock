import pandas as pd

from scripts.astock_backtester import (
    BacktestConfig,
    calculate_performance,
    run_portfolio_backtest,
    select_daily_positions,
)


def make_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rows = []
    for code, closes, pct_chg in [
        ("sh.600000", [10.0, 10.9, 12.0, 13.0], [0.0, 9.0, 10.09, 8.33]),
        ("sz.000001", [10.0, 9.0, 8.0, 7.0], [0.0, -10.0, -11.11, -12.5]),
    ]:
        for date, close, pct in zip(dates, closes, pct_chg):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": close,
                    "pctChg": pct,
                    "tradestatus": 1,
                    "isST": 0,
                    "turn": 3.0,
                    "amount": 100000000.0,
                    "score": 1.0 if code == "sh.600000" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_backtest_uses_previous_day_selection_for_next_day_return() -> None:
    panel = make_panel()

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(top_n=1, fee_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0),
    )

    first_trade_day = result.equity.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert first_trade_day["daily_return"] == 0.09


def test_backtest_can_trade_same_day_when_signal_is_already_lagged() -> None:
    panel = make_panel()
    panel["score"] = 0.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-03")) & (panel["code"] == "sh.600000"), "score"] = 1.0

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(
            top_n=1,
            fee_rate=0.0,
            stamp_tax_rate=0.0,
            slippage_rate=0.0,
            execution_lag_days=0,
        ),
    )

    first_trade_day = result.equity.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert first_trade_day["daily_return"] == 0.09


def test_zero_execution_lag_uses_previous_day_liquidity_for_selection() -> None:
    panel = make_panel()
    panel["score"] = 0.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-03")) & (panel["code"] == "sh.600000"), "score"] = 1.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-02")) & (panel["code"] == "sh.600000"), "amount"] = 1_000_000.0

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(
            top_n=1,
            fee_rate=0.0,
            stamp_tax_rate=0.0,
            slippage_rate=0.0,
            execution_lag_days=0,
        ),
    )

    selected = result.daily_positions.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert selected["code"] == "sz.000001"


def test_backtest_can_hold_between_rebalance_dates() -> None:
    panel = make_panel()
    panel["score"] = 0.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-03")) & (panel["code"] == "sh.600000"), "score"] = 1.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-04")) & (panel["code"] == "sz.000001"), "score"] = 1.0

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(
            top_n=1,
            fee_rate=0.0,
            stamp_tax_rate=0.0,
            slippage_rate=0.0,
            execution_lag_days=0,
            rebalance_interval_days=2,
        ),
    )

    equity = result.equity.set_index("date")
    assert equity.loc[pd.Timestamp("2024-01-03"), "turnover"] == 1.0
    assert equity.loc[pd.Timestamp("2024-01-04"), "turnover"] == 0.0
    assert equity.loc[pd.Timestamp("2024-01-04"), "daily_return"] == 10.09 / 100.0


def test_backtest_blocks_limit_up_buys() -> None:
    panel = make_panel()
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-03")) & (panel["code"] == "sh.600000"), "pctChg"] = 10.0

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(top_n=1, fee_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0),
    )

    first_trade_day = result.equity.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert first_trade_day["daily_return"] == 0.0


def test_backtest_keeps_existing_holding_when_it_limits_up() -> None:
    panel = make_panel()
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-04")) & (panel["code"] == "sh.600000"), "pctChg"] = 10.0

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(top_n=1, fee_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0),
    )

    limit_up_day = result.equity.set_index("date").loc[pd.Timestamp("2024-01-04")]
    assert limit_up_day["daily_return"] == 0.10


def test_backtest_keeps_existing_holding_when_it_limits_down_and_target_changes() -> None:
    panel = make_panel()
    panel["score"] = 0.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-02")) & (panel["code"] == "sh.600000"), "score"] = 1.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-03")) & (panel["code"] == "sz.000001"), "score"] = 1.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-04")) & (panel["code"] == "sh.600000"), "pctChg"] = -10.0
    panel.loc[(panel["date"] == pd.Timestamp("2024-01-04")) & (panel["code"] == "sz.000001"), "pctChg"] = 5.0

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(top_n=1, fee_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0),
    )

    limit_down_day = result.equity.set_index("date").loc[pd.Timestamp("2024-01-04")]
    assert limit_down_day["daily_return"] == -0.10
    assert limit_down_day["position_count"] == 1
    assert limit_down_day["turnover"] == 0.0


def test_calculate_performance_reports_drawdown_and_cagr() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "equity": [1.0, 1.2, 0.9],
            "daily_return": [0.0, 0.2, -0.25],
        }
    )

    metrics = calculate_performance(equity)

    assert metrics["total_return_pct"] == -10.0
    assert metrics["max_drawdown_pct"] == 25.0
    assert metrics["win_rate_pct"] == 50.0


def test_backtest_does_not_charge_full_roundtrip_when_holdings_do_not_change() -> None:
    panel = make_panel()

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(top_n=1, fee_rate=0.001, stamp_tax_rate=0.001, slippage_rate=0.0),
    )

    equity = result.equity.set_index("date")
    assert equity.loc[pd.Timestamp("2024-01-03"), "turnover"] == 1.0
    assert equity.loc[pd.Timestamp("2024-01-04"), "turnover"] == 0.0


def test_backtest_can_stay_in_cash_when_market_filter_is_false() -> None:
    panel = make_panel()
    panel["risk_on"] = panel["date"] == pd.Timestamp("2024-01-02")

    result = run_portfolio_backtest(
        panel,
        score_column="score",
        config=BacktestConfig(top_n=1, fee_rate=0.0, stamp_tax_rate=0.0, slippage_rate=0.0, market_filter_column="risk_on"),
    )

    equity = result.equity.set_index("date")
    assert equity.loc[pd.Timestamp("2024-01-03"), "position_count"] == 1
    assert equity.loc[pd.Timestamp("2024-01-04"), "position_count"] == 0


def test_select_daily_positions_caps_group_exposure_and_keeps_cash_denominator() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 5,
            "code": ["a1", "a2", "a3", "b1", "b2"],
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
            "industry": ["tech", "tech", "tech", "bank", "bank"],
            "tradestatus": [1] * 5,
            "isST": [0] * 5,
            "amount": [100000000.0] * 5,
            "turn": [1.0] * 5,
        }
    )

    positions = select_daily_positions(
        frame,
        "score",
        BacktestConfig(
            top_n=4,
            group_column="industry",
            max_group_weight=0.5,
            preserve_target_cash=True,
        ),
    )

    assert positions["code"].tolist() == ["a1", "a2", "b1", "b2"]
    assert positions["target_weight"].tolist() == [0.25, 0.25, 0.25, 0.25]


def test_backtest_keeps_partial_cash_when_group_cap_limits_available_targets() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    rows = []
    for code, score in [("a1", 3.0), ("a2", 2.0), ("a3", 1.0)]:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "close": 10.0,
                    "pctChg": 10.0 if date == dates[1] else 0.0,
                    "tradestatus": 1,
                    "isST": 0,
                    "turn": 1.0,
                    "amount": 100000000.0,
                    "industry": "tech",
                    "score": score if date == dates[1] else 0.0,
                }
            )
    panel = pd.DataFrame(rows)

    result = run_portfolio_backtest(
        panel,
        "score",
        BacktestConfig(
            top_n=4,
            fee_rate=0.0,
            stamp_tax_rate=0.0,
            slippage_rate=0.0,
            execution_lag_days=0,
            group_column="industry",
            max_group_weight=0.5,
            preserve_target_cash=True,
            limit_threshold_pct=20.0,
        ),
    )

    second_day = result.equity.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert second_day["position_count"] == 2
    assert second_day["daily_return"] == 0.05
