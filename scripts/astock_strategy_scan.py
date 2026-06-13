from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.astock_backtester import BacktestConfig, run_portfolio_backtest
from scripts.astock_fundamentals import merge_fundamentals
from scripts.astock_strategies import add_strategy_scores


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "user_data" / "astock_baostock"
DEFAULT_RESULTS_DIR = ROOT / "scan_results" / "astock"
STRATEGY_COLUMNS = (
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
)


def load_cached_panel(data_dir: Path) -> pd.DataFrame:
    frames = [pd.read_feather(path) for path in sorted(data_dir.glob("*.feather"))]
    if not frames:
        raise ValueError(f"No feather files found in {data_dir}")
    panel = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "tradestatus",
        "isST",
    ]
    for column in numeric_columns:
        if column in panel:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def score_metrics(metrics: dict) -> float:
    return round(
        float(metrics["cagr_pct"]) * 1.0
        - float(metrics["max_drawdown_pct"]) * 0.7
        + float(metrics["win_rate_pct"]) * 0.12,
        3,
    )


def evaluate_strategies(
    panel: pd.DataFrame,
    *,
    top_n_values: Iterable[int],
    execution_lag_days_values: Iterable[int] = (1,),
    rebalance_interval_days_values: Iterable[int] = (1,),
    start_date: str,
    end_date: str,
) -> list[dict]:
    data = add_strategy_scores(panel)
    mask = (data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))
    data = data.loc[mask].copy()

    rows: list[dict] = []
    for strategy_column in STRATEGY_COLUMNS:
        for top_n in top_n_values:
            for execution_lag_days in execution_lag_days_values:
                for rebalance_interval_days in rebalance_interval_days_values:
                    for risk_filter, market_filter_column in [
                        ("none", None),
                        ("risk_on", "risk_on"),
                        ("risk_on_strict", "risk_on_strict"),
                    ]:
                        result = run_portfolio_backtest(
                            data,
                            score_column=strategy_column,
                            config=BacktestConfig(
                                top_n=top_n,
                                market_filter_column=market_filter_column,
                                execution_lag_days=execution_lag_days,
                                rebalance_interval_days=rebalance_interval_days,
                            ),
                        )
                        row = {
                            "strategy": strategy_column,
                            "top_n": top_n,
                            "execution_lag_days": execution_lag_days,
                            "rebalance_interval_days": rebalance_interval_days,
                            "risk_filter": risk_filter,
                            **result.metrics,
                        }
                        row["score"] = score_metrics(result.metrics)
                        rows.append(row)
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan A-share strategy candidates.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fundamentals-file", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--top-n", type=int, action="append")
    parser.add_argument("--execution-lag-days", type=int, action="append", default=[])
    parser.add_argument("--rebalance-interval-days", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    top_n_values = args.top_n or [10, 20, 30]
    execution_lag_days_values = args.execution_lag_days or [1]
    rebalance_interval_days_values = args.rebalance_interval_days or [1]
    panel = load_cached_panel(args.data_dir)
    if args.fundamentals_file:
        panel = merge_fundamentals(panel, pd.read_feather(args.fundamentals_file))
    ranked = evaluate_strategies(
        panel,
        top_n_values=top_n_values,
        execution_lag_days_values=execution_lag_days_values,
        rebalance_interval_days_values=rebalance_interval_days_values,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": ranked,
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for row in ranked:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
