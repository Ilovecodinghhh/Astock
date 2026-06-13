from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.astock_fundamentals import merge_fundamentals
from scripts.astock_metadata import merge_metadata
from scripts.astock_robustness_validation import (
    RobustnessThresholds,
    aggregate_validation_rows,
    codes_from_pool_file,
    filter_panel_to_pool,
    prepare_validation_panel,
    rolling_start_dates,
    run_validation_on_scored_panel,
    write_summary,
)
from scripts.astock_strategies import add_strategy_scores
from scripts.astock_strategy_scan import load_cached_panel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "scan_results" / "astock_batch_robustness_validation"


def strategy_results_dir(
    results_dir: Path,
    *,
    strategy: str,
    risk_filter: str,
    rebalance_interval_days: int,
) -> Path:
    return results_dir / f"{strategy}_{risk_filter}_{rebalance_interval_days}d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multiple scored A-share candidates in one batch.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pool-name")
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--fundamentals-file", type=Path)
    parser.add_argument("--index-file", type=Path)
    parser.add_argument("--index-short-window", type=int, default=60)
    parser.add_argument("--index-long-window", type=int, default=120)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--strategy", action="append", required=True)
    parser.add_argument("--risk-filter", default="none")
    parser.add_argument("--execution-lag-days", type=int, default=0)
    parser.add_argument("--rebalance-interval-days", type=int, default=20)
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
    top_n_values = args.top_n or [10, 20, 30]
    stock_pool = args.pool_name or args.data_dir.name
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

    panel = load_cached_panel(args.data_dir)
    if args.pool_file:
        panel = filter_panel_to_pool(panel, codes_from_pool_file(args.pool_file))
    metadata = pd.read_feather(args.metadata_file) if args.metadata_file else None
    fundamentals = pd.read_feather(args.fundamentals_file) if args.fundamentals_file else None
    index_frame = pd.read_feather(args.index_file) if args.index_file else None
    panel = prepare_validation_panel(
        panel,
        metadata=metadata,
        fundamentals=fundamentals,
        index_frame=index_frame,
        index_short_window=args.index_short_window,
        index_long_window=args.index_long_window,
    )
    scored = add_strategy_scores(panel)

    for strategy in args.strategy:
        candidate = {
            "strategy": strategy,
            "risk_filter": args.risk_filter,
            "execution_lag_days": args.execution_lag_days,
            "rebalance_interval_days": args.rebalance_interval_days,
            "group_column": args.group_column,
            "max_group_weight": args.max_group_weight,
            "preserve_target_cash": args.preserve_target_cash,
        }
        rows = run_validation_on_scored_panel(
            scored,
            stock_pool=stock_pool,
            candidate=candidate,
            top_n_values=top_n_values,
            start_dates=start_dates,
            end_date=args.end_date,
        )
        aggregates = aggregate_validation_rows(rows, thresholds)
        output_path = write_summary(
            results_dir=strategy_results_dir(
                args.results_dir,
                strategy=strategy,
                risk_filter=args.risk_filter,
                rebalance_interval_days=args.rebalance_interval_days,
            ),
            candidate=candidate,
            thresholds=thresholds,
            rows=rows,
            aggregates=aggregates,
        )
        for row in aggregates:
            print(json.dumps(row, ensure_ascii=False))
        print(json.dumps({"strategy": strategy, "summary": str(output_path)}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
