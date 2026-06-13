from pathlib import Path

from scripts.astock_batch_robustness_validation import parse_args, strategy_results_dir


def test_parse_args_accepts_multiple_strategies(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "astock_batch_robustness_validation.py",
            "--data-dir",
            "user_data/astock_baostock_wide_500",
            "--pool-file",
            "user_data/astock_pools/baostock_wide_500.txt",
            "--strategy",
            "low_turnover_trend_score",
            "--strategy",
            "large_value_recovery_score",
            "--top-n",
            "10",
            "--top-n",
            "30",
        ],
    )

    args = parse_args()

    assert args.data_dir == Path("user_data/astock_baostock_wide_500")
    assert args.pool_file == Path("user_data/astock_pools/baostock_wide_500.txt")
    assert args.strategy == ["low_turnover_trend_score", "large_value_recovery_score"]
    assert args.top_n == [10, 30]


def test_strategy_results_dir_is_stable_for_summary_paths() -> None:
    path = strategy_results_dir(
        Path("scan_results/batch"),
        strategy="low_turnover_trend_score",
        risk_filter="none",
        rebalance_interval_days=20,
    )

    assert path == Path("scan_results/batch/low_turnover_trend_score_none_20d")
