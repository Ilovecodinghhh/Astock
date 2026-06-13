from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = ROOT / "scan_results"


def evaluation_sort_key(row: dict) -> tuple:
    return (
        bool(row.get("passes_walk_forward", False)),
        int(row.get("positive_windows", 0)),
        float(row.get("median_validation_cagr_pct", -999.0)),
        float(row.get("score", -999.0)),
    )


def best_evaluation_per_strategy(summary_paths: Iterable[Path]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload.get("evaluations", []):
            candidate = dict(row)
            candidate["source"] = str(summary_path)
            strategy = candidate["strategy"]
            if strategy not in best or evaluation_sort_key(candidate) > evaluation_sort_key(best[strategy]):
                best[strategy] = candidate
    return best


def rank_evaluations(rows: Iterable[dict]) -> list[dict]:
    return sorted(rows, key=evaluation_sort_key, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank walk-forward summary files.")
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_paths = sorted(args.scan_dir.glob("**/summary.json"))
    ranked = rank_evaluations(best_evaluation_per_strategy(summary_paths).values())
    for row in ranked[: args.top]:
        print(
            json.dumps(
                {
                    "strategy": row["strategy"],
                    "passes_walk_forward": row.get("passes_walk_forward", False),
                    "positive_windows": row.get("positive_windows", 0),
                    "median_validation_cagr_pct": row.get("median_validation_cagr_pct"),
                    "worst_validation_cagr_pct": row.get("worst_validation_cagr_pct"),
                    "worst_validation_drawdown_pct": row.get("worst_validation_drawdown_pct"),
                    "min_validation_profit_factor": row.get("min_validation_profit_factor"),
                    "min_validation_trades": row.get("min_validation_trades"),
                    "score": row.get("score"),
                    "source": row.get("source"),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
