import json
import zipfile
from pathlib import Path

import pytest

from scripts.evaluate_backtests import (
    PERIODS,
    STRATEGY_GROUPS,
    evaluate_candidate,
    parse_args,
    read_backtest_summary,
    selected_candidates,
    selected_periods,
    sort_evaluations,
    write_summary,
)


def make_result_zip(path: Path, summary: dict) -> None:
    payload = {"strategy_comparison": [summary]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("backtest-result.json", json.dumps(payload))
        archive.writestr("backtest-result_config.json", "{}")


def test_read_backtest_summary_extracts_freqtrade_metrics(tmp_path: Path) -> None:
    zip_path = tmp_path / "backtest-result.zip"
    make_result_zip(
        zip_path,
        {
            "trades": 42,
            "profit_total_pct": 62.75,
            "cagr": 0.514,
            "max_drawdown_account": 0.187,
            "profit_factor": 1.64,
            "sharpe": 1.25,
            "sortino": 1.92,
        },
    )

    row = read_backtest_summary("DemoStrategy", "validation", "20250101-20260531", zip_path)

    assert row == {
        "strategy": "DemoStrategy",
        "period": "validation",
        "timerange": "20250101-20260531",
        "trades": 42,
        "profit_total_pct": 62.75,
        "cagr_pct": 51.4,
        "drawdown_pct": 18.7,
        "profit_factor": 1.64,
        "sharpe": 1.25,
        "sortino": 1.92,
    }


def test_evaluate_candidate_requires_all_validation_windows_to_pass() -> None:
    rows = [
        {
            "period": "validation_2023",
            "cagr_pct": 62.0,
            "drawdown_pct": 31.0,
            "profit_factor": 1.35,
            "trades": 90,
        },
        {
            "period": "validation_2024",
            "cagr_pct": 55.0,
            "drawdown_pct": 26.0,
            "profit_factor": 1.42,
            "trades": 110,
        },
        {
            "period": "validation_2025_2026",
            "cagr_pct": 44.0,
            "drawdown_pct": 22.0,
            "profit_factor": 1.25,
            "trades": 80,
        },
    ]

    evaluation = evaluate_candidate("DemoStrategy", rows)

    assert evaluation["passes_walk_forward"] is True
    assert evaluation["median_validation_cagr_pct"] == 55.0
    assert evaluation["worst_validation_cagr_pct"] == 44.0


def test_evaluate_candidate_rejects_fragile_single_year_winner() -> None:
    rows = [
        {
            "period": "validation_2023",
            "cagr_pct": -12.0,
            "drawdown_pct": 36.0,
            "profit_factor": 0.82,
            "trades": 120,
        },
        {
            "period": "validation_2024",
            "cagr_pct": 98.0,
            "drawdown_pct": 28.0,
            "profit_factor": 1.95,
            "trades": 140,
        },
        {
            "period": "validation_2025_2026",
            "cagr_pct": 6.0,
            "drawdown_pct": 18.0,
            "profit_factor": 1.05,
            "trades": 75,
        },
    ]

    evaluation = evaluate_candidate("FragileStrategy", rows)

    assert evaluation["passes_walk_forward"] is False
    assert evaluation["positive_windows"] == 2
    assert evaluation["worst_validation_cagr_pct"] == -12.0


def test_sort_evaluations_ranks_passing_candidates_first() -> None:
    passing = {
        "strategy": "PassingStrategy",
        "passes_walk_forward": True,
        "median_validation_cagr_pct": 55.0,
        "score": 70.0,
    }
    shiny_failure = {
        "strategy": "ShinyFailure",
        "passes_walk_forward": False,
        "median_validation_cagr_pct": 200.0,
        "score": 200.0,
    }

    assert sort_evaluations([shiny_failure, passing]) == [passing, shiny_failure]


def test_strategy_groups_include_spot_and_futures_candidates() -> None:
    assert "spot" in STRATEGY_GROUPS
    assert "futures" in STRATEGY_GROUPS
    assert "alternative" in STRATEGY_GROUPS
    assert STRATEGY_GROUPS["spot"]
    assert STRATEGY_GROUPS["futures"]
    assert STRATEGY_GROUPS["alternative"]

    alternative_names = {candidate[0] for candidate in STRATEGY_GROUPS["alternative"]}
    assert {
        "FuturesVolatilityExpansionTrendStrategy",
        "FuturesCrashReversalMeanReversionStrategy",
        "SpotRelativeStrengthDefensiveRotationStrategy",
        "FuturesExpandedUniverseTrendStrategy",
        "FuturesCorePairsDailyTrendStrategy",
        "FuturesRiskOffShortOnlyStrategy",
        "FuturesExtremeFundingReversalStrategy",
        "FuturesCarryTrendRelaxedStrategy",
        "FuturesRelativeStrengthLongRotationStrategy",
        "FuturesRelativeStrengthLongRotation3xStrategy",
        "FuturesRelativeStrengthLongRotationTop2Strategy",
        "FuturesRelativeStrengthLongRotationTop1Strategy",
        "FuturesRelativeStrengthLongRotationLooseStrategy",
        "FuturesRelativeStrengthLongRotationLoose2xStrategy",
        "FuturesRelativeStrengthLongRotationLooseGuardStrategy",
        "FuturesRelativeStrengthLongRotationLooseGuardHighStrategy",
        "FuturesPrecomputedRelativeStrengthLooseGuardStrategy",
        "FuturesPrecomputedRelativeStrengthLooseGuardHighStrategy",
        "FuturesPrecomputedRelativeStrengthLooseGuardDefensiveStrategy",
        "FuturesPrecomputedRelativeStrengthLooseGuardStrictStrategy",
        "FuturesSinglePairMomentumGuardStrategy",
        "FuturesTrainWinnersMomentumGuardStrategy",
    }.issubset(alternative_names)


def test_parse_args_accepts_custom_results_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_backtests.py", "--results-dir", "scan_results/walk_forward_alt"],
    )

    args = parse_args()

    assert args.results_dir == Path("scan_results/walk_forward_alt")


def test_write_summary_uses_custom_results_dir(tmp_path: Path) -> None:
    rows = [
        {
            "strategy": "DemoStrategy",
            "period": "validation_2023",
            "timerange": "20230101-20231231",
            "trades": 42,
            "profit_total_pct": 8.5,
            "cagr_pct": 8.5,
            "drawdown_pct": 4.0,
            "profit_factor": 1.2,
            "sharpe": 1.0,
            "sortino": 1.3,
        }
    ]
    evaluations = [
        {
            "strategy": "DemoStrategy",
            "passes_walk_forward": False,
            "required_median_cagr_pct": 50.0,
            "required_worst_cagr_pct": 20.0,
            "positive_windows": 1,
            "median_validation_cagr_pct": 8.5,
            "worst_validation_cagr_pct": 8.5,
            "worst_validation_drawdown_pct": 4.0,
            "min_validation_profit_factor": 1.2,
            "min_validation_trades": 42,
            "score": 20.3,
        }
    ]

    write_summary(rows, evaluations, tmp_path)

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["rows"] == rows
    assert payload["evaluations"][0]["strategy"] == "DemoStrategy"


def test_selected_periods_can_run_train_only() -> None:
    periods = selected_periods(["train_2021_2022"], skip_full=False)

    assert periods == {"train_2021_2022": PERIODS["train_2021_2022"]}


def test_selected_periods_default_excludes_train_and_full_when_skipping_full() -> None:
    periods = selected_periods([], skip_full=True)

    assert "train_2021_2022" not in periods
    assert "full_2021_2026" not in periods
    assert set(periods) == {"validation_2023", "validation_2024", "validation_2025_2026"}


def test_selected_periods_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown period"):
        selected_periods(["future_window"], skip_full=False)


def test_selected_candidates_can_filter_by_strategy_name() -> None:
    candidates = selected_candidates(["alternative"], limit=None, strategy_names=["FuturesRelativeStrengthLongRotation3xStrategy"])

    assert candidates == [
        (
            "FuturesRelativeStrengthLongRotation3xStrategy",
            "4h",
            "user_data/config_binance_futures_expanded.json",
        )
    ]
