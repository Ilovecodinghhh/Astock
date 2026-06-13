import json
from pathlib import Path

from scripts.rank_backtest_summaries import best_evaluation_per_strategy, rank_evaluations


def test_best_evaluation_per_strategy_keeps_highest_score_for_duplicate_strategy(tmp_path: Path) -> None:
    first = tmp_path / "first" / "summary.json"
    second = tmp_path / "second" / "summary.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "strategy": "DemoStrategy",
                        "passes_walk_forward": False,
                        "positive_windows": 3,
                        "median_validation_cagr_pct": 10.0,
                        "score": 12.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "strategy": "DemoStrategy",
                        "passes_walk_forward": False,
                        "positive_windows": 3,
                        "median_validation_cagr_pct": 20.0,
                        "score": 22.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    best = best_evaluation_per_strategy([first, second])

    assert best["DemoStrategy"]["median_validation_cagr_pct"] == 20.0
    assert best["DemoStrategy"]["source"] == str(second)


def test_rank_evaluations_prioritizes_passes_then_robust_return() -> None:
    rows = [
        {
            "strategy": "HighMedianFail",
            "passes_walk_forward": False,
            "positive_windows": 3,
            "median_validation_cagr_pct": 80.0,
            "score": 80.0,
        },
        {
            "strategy": "Passing",
            "passes_walk_forward": True,
            "positive_windows": 3,
            "median_validation_cagr_pct": 50.0,
            "score": 50.0,
        },
    ]

    ranked = rank_evaluations(rows)

    assert [row["strategy"] for row in ranked] == ["Passing", "HighMedianFail"]
