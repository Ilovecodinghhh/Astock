from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
USER_DATA = ROOT / "user_data"
RESULTS = ROOT / "scan_results" / "walk_forward"
DEFAULT_FREQTRADE = Path(
    "C:/Users/byf/Documents/New project/binance-freqtrade-lab/.conda/freqtrade-binance/Scripts/freqtrade.exe"
)

PERIODS = {
    "train_2021_2022": "20210101-20221231",
    "validation_2023": "20230101-20231231",
    "validation_2024": "20240101-20241231",
    "validation_2025_2026": "20250101-20260531",
    "full_2021_2026": "20210101-20260531",
}
VALIDATION_PERIODS = ("validation_2023", "validation_2024", "validation_2025_2026")

STRATEGY_GROUPS = {
    "spot": [
        ("BreakoutTrendStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutBaselineStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutBtcRegimeStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutStrictBtcRegimeStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutEthRegimeStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutBtcEthRegimeStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutPairDailyTrendStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutStrictPairDailyTrendStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutRelativeStrengthStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutTop4RelativeStrengthStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutVolatilityStakeStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutChandelierExitStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutFastRsiExitStrategy", "4h", "user_data/config_binance_spot.json"),
        ("BreakoutStrictDefensiveStrategy", "4h", "user_data/config_binance_spot.json"),
        ("PullbackTrendDeepStrategy", "4h", "user_data/config_binance_spot.json"),
    ],
    "futures": [
        ("FuturesTrendLongShortStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesTrendLongShort3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesBreakoutLongShortStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesBreakoutLongShort3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesBollingerFundingFadeLongOnlyStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesHighConvictionBtcTrendStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesMegaCapTrend3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesBtcEthTrend3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesCrossSectionalLongShortStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesCrossSectionalLongShort3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesMarketNeutralMomentumStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesMarketNeutralMomentum3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesStrictRegimeTrendStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesCounterTrendMeanReversionStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesRegimeScalpLongShortStrategy", "1h", "user_data/config_binance_futures.json"),
        ("FuturesRegimeScalpLongShort3xStrategy", "1h", "user_data/config_binance_futures.json"),
        ("FuturesBollingerFundingFadeStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesCarryMomentumStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRegimeSwitchCrossSectionalStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesRegimeSwitchCrossSectional3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesBullBearBreakoutStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesBollingerFundingFadeLongOnlyTightStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesBollingerFundingFadeLongOnlyFastStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
    ],
    "alternative": [
        ("FuturesVolatilityExpansionTrendStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesVolatilityExpansionTrend3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesCrashReversalMeanReversionStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesCrashReversalMeanReversion3xStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesRangeBreakoutAdaptiveStrategy", "4h", "user_data/config_binance_futures.json"),
        ("FuturesExpandedUniverseTrendStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesExpandedUniverseTrend3xStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesCorePairsDailyTrendStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRiskOffShortOnlyStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesExtremeFundingReversalStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesCarryTrendRelaxedStrategy", "1h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotation3xStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationTop2Strategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationTop1Strategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationLooseStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationLoose2xStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationLooseGuardStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesRelativeStrengthLongRotationLooseGuardHighStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesPrecomputedRelativeStrengthLooseGuardStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesPrecomputedRelativeStrengthLooseGuardHighStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesPrecomputedRelativeStrengthLooseGuardDefensiveStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesPrecomputedRelativeStrengthLooseGuardStrictStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesSinglePairMomentumGuardStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("FuturesTrainWinnersMomentumGuardStrategy", "4h", "user_data/config_binance_futures_expanded.json"),
        ("SpotRelativeStrengthDefensiveRotationStrategy", "4h", "user_data/config_binance_spot.json"),
        ("SpotQualityPullbackRotationStrategy", "4h", "user_data/config_binance_spot.json"),
    ],
}


def read_backtest_summary(strategy: str, period_name: str, timerange: str, zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        json_name = next(name for name in archive.namelist() if name.endswith(".json") and "_config" not in name)
        data = json.loads(archive.read(json_name).decode("utf-8"))

    summary = data["strategy_comparison"][0]
    return {
        "strategy": strategy,
        "period": period_name,
        "timerange": timerange,
        "trades": int(summary["trades"]),
        "profit_total_pct": round(float(summary["profit_total_pct"]), 4),
        "cagr_pct": round(float(summary["cagr"]) * 100, 2),
        "drawdown_pct": round(float(summary["max_drawdown_account"]) * 100, 2),
        "profit_factor": round(float(summary["profit_factor"]), 3),
        "sharpe": round(float(summary["sharpe"]), 3),
        "sortino": round(float(summary["sortino"]), 3),
    }


def evaluate_candidate(strategy: str, validation_rows: Iterable[dict]) -> dict:
    rows = [row for row in validation_rows if row["period"] in VALIDATION_PERIODS]
    cagrs = [float(row["cagr_pct"]) for row in rows]
    drawdowns = [float(row["drawdown_pct"]) for row in rows]
    profit_factors = [float(row["profit_factor"]) for row in rows]
    trades = [int(row["trades"]) for row in rows]
    positive_windows = sum(1 for cagr in cagrs if cagr > 0)
    median_cagr = round(statistics.median(cagrs), 2) if cagrs else 0.0
    worst_cagr = round(min(cagrs), 2) if cagrs else 0.0
    worst_drawdown = round(max(drawdowns), 2) if drawdowns else 0.0
    min_profit_factor = round(min(profit_factors), 3) if profit_factors else 0.0
    min_trades = min(trades) if trades else 0

    passes = (
        len(rows) == len(VALIDATION_PERIODS)
        and positive_windows == len(VALIDATION_PERIODS)
        and median_cagr >= 50.0
        and worst_cagr >= 20.0
        and worst_drawdown <= 45.0
        and min_profit_factor >= 1.15
        and min_trades >= 25
    )
    score = median_cagr + worst_cagr * 0.8 - worst_drawdown * 0.55 + min_profit_factor * 5.0
    return {
        "strategy": strategy,
        "passes_walk_forward": passes,
        "required_median_cagr_pct": 50.0,
        "required_worst_cagr_pct": 20.0,
        "positive_windows": positive_windows,
        "median_validation_cagr_pct": median_cagr,
        "worst_validation_cagr_pct": worst_cagr,
        "worst_validation_drawdown_pct": worst_drawdown,
        "min_validation_profit_factor": min_profit_factor,
        "min_validation_trades": min_trades,
        "score": round(score, 3),
    }


def sort_evaluations(evaluations: Iterable[dict]) -> list[dict]:
    return sorted(
        evaluations,
        key=lambda item: (
            bool(item["passes_walk_forward"]),
            float(item["median_validation_cagr_pct"]),
            float(item["score"]),
        ),
        reverse=True,
    )


def retryable_backtest_error(stderr: str) -> bool:
    retryable_fragments = (
        "ExchangeNotAvailable",
        "TemporaryError",
        "exchangeInfo",
        "TimeoutError",
        "NetworkError",
    )
    return any(fragment in stderr for fragment in retryable_fragments)


def run_backtest(
    *,
    freqtrade: Path,
    results_dir: Path,
    strategy: str,
    timeframe: str,
    config: Path,
    period_name: str,
    timerange: str,
    force: bool,
) -> dict:
    outdir = results_dir / f"{strategy}_{period_name}"
    if outdir.exists() and force:
        shutil.rmtree(outdir)
    if outdir.exists():
        existing = sorted(outdir.glob("backtest-result-*.zip"))
        if existing:
            return read_backtest_summary(strategy, period_name, timerange, existing[-1])
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        str(freqtrade),
        "backtesting",
        "--config",
        str(config),
        "--userdir",
        str(USER_DATA),
        "--strategy",
        strategy,
        "--timeframe",
        timeframe,
        "--timerange",
        timerange,
        "--export",
        "trades",
        "--cache",
        "none",
        "--enable-protections",
        "--backtest-directory",
        str(outdir),
    ]
    for attempt in range(1, 4):
        try:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            break
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            retrying = attempt < 3 and retryable_backtest_error(stderr)
            print(
                json.dumps(
                    {
                        "strategy": strategy,
                        "period": period_name,
                        "timerange": timerange,
                        "attempt": attempt,
                        "retrying": retrying,
                        "returncode": exc.returncode,
                        "stderr_tail": stderr[-3000:],
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            if not retrying:
                raise
            time.sleep(5 * attempt)

    zip_path = sorted(outdir.glob("backtest-result-*.zip"))[-1]
    return read_backtest_summary(strategy, period_name, timerange, zip_path)


def selected_candidates(
    groups: Iterable[str],
    limit: int | None,
    strategy_names: Iterable[str] | None = None,
) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    allowed = set(strategy_names or [])
    for group in groups:
        candidates.extend(STRATEGY_GROUPS[group])
    if allowed:
        candidates = [candidate for candidate in candidates if candidate[0] in allowed]
        found = {candidate[0] for candidate in candidates}
        missing = sorted(allowed - found)
        if missing:
            raise ValueError(f"Unknown strategy for selected groups: {', '.join(missing)}")
    if limit is not None:
        return candidates[:limit]
    return candidates


def selected_periods(period_names: Iterable[str] | None, skip_full: bool) -> dict[str, str]:
    names = list(period_names or [])
    if names:
        unknown = [name for name in names if name not in PERIODS]
        if unknown:
            raise ValueError(f"Unknown period: {', '.join(unknown)}")
        return {name: PERIODS[name] for name in names}
    periods = {key: PERIODS[key] for key in VALIDATION_PERIODS}
    if not skip_full:
        periods["full_2021_2026"] = PERIODS["full_2021_2026"]
    return periods


def write_summary(rows: list[dict], evaluations: list[dict], results_dir: Path = RESULTS) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "selection_rule": (
            "A strategy passes only when all validation windows are positive, median validation CAGR is at least 50%, "
            "worst validation CAGR is at least 20%, worst validation drawdown is at most 45%, "
            "minimum validation profit factor is at least 1.15, and every validation window has at least 25 trades."
        ),
        "periods": PERIODS,
        "validation_periods": list(VALIDATION_PERIODS),
        "rows": rows,
        "evaluations": sort_evaluations(evaluations),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed walk-forward Freqtrade backtests.")
    parser.add_argument("--freqtrade", type=Path, default=DEFAULT_FREQTRADE)
    parser.add_argument("--group", choices=sorted(STRATEGY_GROUPS), action="append", default=[])
    parser.add_argument("--strategy", action="append", default=[], help="Only run the named strategy. Can be repeated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument(
        "--period",
        choices=sorted(PERIODS),
        action="append",
        default=[],
        help="Period name to run. Can be passed multiple times. Defaults to all periods except full when --skip-full is set.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Directory for per-run backtest artifacts and summary.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = args.group or sorted(STRATEGY_GROUPS)
    candidates = selected_candidates(groups, args.limit, args.strategy)
    periods = selected_periods(args.period, args.skip_full)
    rows: list[dict] = []
    evaluations: list[dict] = []

    for strategy, timeframe, config_name in candidates:
        strategy_rows: list[dict] = []
        config = ROOT / config_name
        for period_name, timerange in periods.items():
            row = run_backtest(
                freqtrade=args.freqtrade,
                results_dir=args.results_dir,
                strategy=strategy,
                timeframe=timeframe,
                config=config,
                period_name=period_name,
                timerange=timerange,
                force=args.force,
            )
            rows.append(row)
            strategy_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        evaluation = evaluate_candidate(strategy, strategy_rows)
        evaluations.append(evaluation)
        print(json.dumps(evaluation, ensure_ascii=False), flush=True)

    write_summary(rows, evaluations, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
