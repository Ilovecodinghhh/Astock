from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.astock_backtester import BacktestConfig, run_portfolio_backtest
from scripts.astock_fundamentals import merge_fundamentals
from scripts.astock_index import add_index_trend_filter
from scripts.astock_metadata import merge_metadata
from scripts.astock_strategies import add_strategy_scores
from scripts.astock_strategy_scan import load_cached_panel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "scan_results" / "astock_robustness_validation"


@dataclass(frozen=True)
class RobustnessThresholds:
    min_median_cagr_pct: float = 50.0
    min_worst_cagr_pct: float = 20.0
    max_worst_drawdown_pct: float = 45.0
    max_single_stock_weight_pct: float = 33.34


def rolling_start_dates(first_start_date: str, last_start_date: str, *, step_months: int) -> list[str]:
    if step_months < 1:
        raise ValueError("step_months must be positive")

    starts = []
    current = pd.Timestamp(first_start_date)
    last = pd.Timestamp(last_start_date)
    while current <= last:
        starts.append(current.strftime("%Y-%m-%d"))
        current = current + pd.DateOffset(months=step_months)
    return starts


def codes_from_pool_file(pool_file: Path) -> list[str]:
    codes = []
    for line in pool_file.read_text(encoding="utf-8").splitlines():
        code = line.strip()
        if code and not code.startswith("#"):
            codes.append(code)
    return sorted(dict.fromkeys(codes))


def filter_panel_to_pool(panel: pd.DataFrame, codes: Iterable[str]) -> pd.DataFrame:
    allowed = set(codes)
    return panel.loc[panel["code"].isin(allowed)].copy().sort_values(["date", "code"]).reset_index(drop=True)


def prepare_validation_panel(
    panel: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    index_frame: pd.DataFrame | None = None,
    index_short_window: int = 60,
    index_long_window: int = 120,
) -> pd.DataFrame:
    prepared = panel.copy()
    if metadata is not None:
        prepared = merge_metadata(prepared, metadata)
    if fundamentals is not None:
        prepared = merge_fundamentals(prepared, fundamentals)
    if index_frame is not None:
        prepared = add_index_trend_filter(
            prepared,
            index_frame,
            short_window=index_short_window,
            long_window=index_long_window,
        )
    return prepared.sort_values(["date", "code"]).reset_index(drop=True)


def _median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 2) if values else 0.0


def _minimum(values: list[float]) -> float:
    return round(float(min(values)), 2) if values else 0.0


def _maximum(values: list[float]) -> float:
    return round(float(max(values)), 2) if values else 0.0


def aggregate_validation_rows(rows: Iterable[dict], thresholds: RobustnessThresholds) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["stock_pool"],
            row["stock_pool_size"],
            row.get("strategy", "unknown"),
            row["top_n"],
            row.get("risk_filter", "none"),
            row.get("execution_lag_days", 1),
            row.get("rebalance_interval_days", 1),
            row.get("group_column"),
            row.get("max_group_weight"),
            row.get("preserve_target_cash", False),
            row["max_single_stock_weight_pct"],
        )
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (
        stock_pool,
        stock_pool_size,
        strategy,
        top_n,
        risk_filter,
        execution_lag_days,
        rebalance_interval_days,
        group_column,
        max_group_weight,
        preserve_target_cash,
        max_single_stock_weight_pct,
    ), group_rows in grouped.items():
        cagrs = [float(row["cagr_pct"]) for row in group_rows]
        drawdowns = [float(row["max_drawdown_pct"]) for row in group_rows]
        trading_days = [int(row["trading_days"]) for row in group_rows]
        positive_windows = sum(1 for cagr in cagrs if cagr > 0.0)
        median_cagr = _median(cagrs)
        worst_cagr = _minimum(cagrs)
        worst_drawdown = _maximum(drawdowns)
        passes_concentration = float(max_single_stock_weight_pct) <= thresholds.max_single_stock_weight_pct
        passes_returns = (
            median_cagr >= thresholds.min_median_cagr_pct
            and worst_cagr >= thresholds.min_worst_cagr_pct
            and worst_drawdown <= thresholds.max_worst_drawdown_pct
            and positive_windows == len(group_rows)
        )
        score = median_cagr + worst_cagr * 0.8 - worst_drawdown * 0.55 - float(max_single_stock_weight_pct) * 0.1
        aggregates.append(
            {
                "stock_pool": stock_pool,
                "stock_pool_size": stock_pool_size,
                "strategy": strategy,
                "top_n": top_n,
                "risk_filter": risk_filter,
                "execution_lag_days": execution_lag_days,
                "rebalance_interval_days": rebalance_interval_days,
                "group_column": group_column,
                "max_group_weight": max_group_weight,
                "preserve_target_cash": preserve_target_cash,
                "max_single_stock_weight_pct": max_single_stock_weight_pct,
                "passes_strict_validation": bool(passes_concentration and passes_returns),
                "passes_concentration": bool(passes_concentration),
                "rolling_window_count": len(group_rows),
                "positive_windows": positive_windows,
                "median_cagr_pct": median_cagr,
                "worst_cagr_pct": worst_cagr,
                "worst_drawdown_pct": worst_drawdown,
                "median_trading_days": int(statistics.median(trading_days)) if trading_days else 0,
                "score": round(score, 3),
            }
        )
    return rank_aggregates(aggregates)


def rank_aggregates(rows: Iterable[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row["passes_strict_validation"]),
            bool(row.get("passes_concentration", True)),
            int(row["positive_windows"]),
            float(row["median_cagr_pct"]),
            float(row["worst_cagr_pct"]),
            float(row["score"]),
        ),
        reverse=True,
    )


def run_validation(
    panel: pd.DataFrame,
    *,
    stock_pool: str,
    candidate: dict,
    top_n_values: Iterable[int],
    start_dates: Iterable[str],
    end_date: str,
) -> list[dict]:
    scored = add_strategy_scores(panel)
    return run_validation_on_scored_panel(
        scored,
        stock_pool=stock_pool,
        candidate=candidate,
        top_n_values=top_n_values,
        start_dates=start_dates,
        end_date=end_date,
    )


def run_validation_on_scored_panel(
    scored: pd.DataFrame,
    *,
    stock_pool: str,
    candidate: dict,
    top_n_values: Iterable[int],
    start_dates: Iterable[str],
    end_date: str,
) -> list[dict]:
    rows = []
    stock_pool_size = int(scored["code"].nunique())
    strategy = candidate["strategy"]
    risk_filter = candidate.get("risk_filter", "none")
    market_filter_column = None if risk_filter == "none" else risk_filter
    execution_lag_days = int(candidate.get("execution_lag_days", 1))
    rebalance_interval_days = int(candidate.get("rebalance_interval_days", 1))
    group_column = candidate.get("group_column")
    max_group_weight = candidate.get("max_group_weight")
    preserve_target_cash = bool(candidate.get("preserve_target_cash", False))

    for top_n in top_n_values:
        max_weight = round(100.0 / int(top_n), 2)
        for start_date in start_dates:
            mask = (scored["date"] >= pd.Timestamp(start_date)) & (scored["date"] <= pd.Timestamp(end_date))
            data = scored.loc[mask].copy()
            result = run_portfolio_backtest(
                data,
                score_column=strategy,
                config=BacktestConfig(
                    top_n=int(top_n),
                    market_filter_column=market_filter_column,
                    execution_lag_days=execution_lag_days,
                    rebalance_interval_days=rebalance_interval_days,
                    group_column=group_column,
                    max_group_weight=max_group_weight,
                    preserve_target_cash=preserve_target_cash,
                ),
            )
            rows.append(
                {
                    "stock_pool": stock_pool,
                    "stock_pool_size": stock_pool_size,
                    "strategy": strategy,
                    "top_n": int(top_n),
                    "risk_filter": risk_filter,
                    "execution_lag_days": execution_lag_days,
                    "rebalance_interval_days": rebalance_interval_days,
                    "group_column": group_column,
                    "max_group_weight": max_group_weight,
                    "preserve_target_cash": preserve_target_cash,
                    "max_single_stock_weight_pct": max_weight,
                    "start_date": start_date,
                    "end_date": end_date,
                    **result.metrics,
                }
            )
    return rows


def write_summary(
    *,
    results_dir: Path,
    candidate: dict,
    thresholds: RobustnessThresholds,
    rows: list[dict],
    aggregates: list[dict],
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "summary.json"
    payload = {
        "candidate": candidate,
        "thresholds": asdict(thresholds),
        "rows": rows,
        "aggregates": aggregates,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate A-share candidates with stricter robustness checks.")
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--pool-name", action="append")
    parser.add_argument("--pool-file", type=Path, action="append", default=[])
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--fundamentals-file", type=Path)
    parser.add_argument("--index-file", type=Path)
    parser.add_argument("--index-short-window", type=int, default=60)
    parser.add_argument("--index-long-window", type=int, default=120)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--strategy", default="relative_strength_quality_score")
    parser.add_argument("--risk-filter", default="risk_on_strict")
    parser.add_argument("--execution-lag-days", type=int, default=0)
    parser.add_argument("--rebalance-interval-days", type=int, default=5)
    parser.add_argument("--group-column")
    parser.add_argument("--max-group-weight", type=float)
    parser.add_argument("--preserve-target-cash", action="store_true")
    parser.add_argument("--top-n", type=int, action="append", default=[])
    parser.add_argument("--first-start-date", default="2021-01-01")
    parser.add_argument("--last-start-date", default="2024-01-01")
    parser.add_argument("--start-step-months", type=int, default=3)
    parser.add_argument("--end-date", default="2026-05-31")
    parser.add_argument("--min-median-cagr-pct", type=float, default=50.0)
    parser.add_argument("--min-worst-cagr-pct", type=float, default=20.0)
    parser.add_argument("--max-worst-drawdown-pct", type=float, default=45.0)
    parser.add_argument("--max-single-stock-weight-pct", type=float, default=33.34)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    top_n_values = args.top_n or [1, 2, 3, 5, 8, 10]
    pool_names = args.pool_name or [path.name for path in args.data_dir]
    if len(pool_names) != len(args.data_dir):
        raise SystemExit("--pool-name must be supplied once per --data-dir")
    if args.pool_file and len(args.pool_file) != len(args.data_dir):
        raise SystemExit("--pool-file must be supplied once per --data-dir")

    candidate = {
        "strategy": args.strategy,
        "risk_filter": args.risk_filter,
        "execution_lag_days": args.execution_lag_days,
        "rebalance_interval_days": args.rebalance_interval_days,
        "group_column": args.group_column,
        "max_group_weight": args.max_group_weight,
        "preserve_target_cash": args.preserve_target_cash,
    }
    thresholds = RobustnessThresholds(
        min_median_cagr_pct=args.min_median_cagr_pct,
        min_worst_cagr_pct=args.min_worst_cagr_pct,
        max_worst_drawdown_pct=args.max_worst_drawdown_pct,
        max_single_stock_weight_pct=args.max_single_stock_weight_pct,
    )
    start_dates = rolling_start_dates(
        args.first_start_date,
        args.last_start_date,
        step_months=args.start_step_months,
    )

    rows = []
    metadata = pd.read_feather(args.metadata_file) if args.metadata_file else None
    fundamentals = pd.read_feather(args.fundamentals_file) if args.fundamentals_file else None
    index_frame = pd.read_feather(args.index_file) if args.index_file else None
    pool_files = args.pool_file or [None] * len(args.data_dir)
    for data_dir, pool_name, pool_file in zip(args.data_dir, pool_names, pool_files):
        panel = load_cached_panel(data_dir)
        if pool_file is not None:
            panel = filter_panel_to_pool(panel, codes_from_pool_file(pool_file))
        panel = prepare_validation_panel(
            panel,
            metadata=metadata,
            fundamentals=fundamentals,
            index_frame=index_frame,
            index_short_window=args.index_short_window,
            index_long_window=args.index_long_window,
        )
        rows.extend(
            run_validation(
                panel,
                stock_pool=pool_name,
                candidate=candidate,
                top_n_values=top_n_values,
                start_dates=start_dates,
                end_date=args.end_date,
            )
        )

    aggregates = aggregate_validation_rows(rows, thresholds)
    output_path = write_summary(
        results_dir=args.results_dir,
        candidate=candidate,
        thresholds=thresholds,
        rows=rows,
        aggregates=aggregates,
    )
    for row in aggregates:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps({"summary": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
