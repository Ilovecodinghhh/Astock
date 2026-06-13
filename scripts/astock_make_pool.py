from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.astock_baostock import baostock_result_to_frame, import_baostock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "user_data" / "astock_pools" / "baostock_wide_500.txt"


def a_share_codes_from_frame(frame: pd.DataFrame) -> list[str]:
    codes = sorted(str(code) for code in frame["code"].dropna().unique())
    return [
        code
        for code in codes
        if code.startswith(("sh.600", "sh.601", "sh.603", "sh.605", "sh.688", "sz.000", "sz.001", "sz.002", "sz.003", "sz.300"))
    ]


def codes_from_cache_dir(cache_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix("-d.feather") for path in cache_dir.glob("*-d.feather"))


def evenly_spaced_codes(codes: list[str], limit: int) -> list[str]:
    if limit < 1:
        return []
    unique = sorted(dict.fromkeys(codes))
    if len(unique) <= limit:
        return unique
    if limit == 1:
        return [unique[0]]
    step = (len(unique) - 1) / (limit - 1)
    indexes = [round(index * step) for index in range(limit)]
    return [unique[index] for index in indexes]


def merge_required_and_sampled_codes(all_codes: list[str], required_codes: list[str], *, limit: int) -> list[str]:
    required = sorted(code for code in dict.fromkeys(required_codes) if code in set(all_codes))
    remaining_limit = max(limit - len(required), 0)
    remaining = [code for code in all_codes if code not in set(required)]
    return required + evenly_spaced_codes(remaining, remaining_limit)


def query_all_stock_codes(date: str) -> list[str]:
    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        result = bs.query_all_stock(date)
        if result.error_code != "0":
            raise RuntimeError(f"baostock query_all_stock failed: {result.error_msg}")
        frame = baostock_result_to_frame(result)
    finally:
        bs.logout()
    return a_share_codes_from_frame(frame)


def write_pool_file(codes: list[str], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(codes) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable A-share pool file from baostock code listings.")
    parser.add_argument("--date", required=True, help="Trading date for baostock.query_all_stock, e.g. 2026-05-29.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--include-cache-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_codes = query_all_stock_codes(args.date)
    required_codes = []
    for cache_dir in args.include_cache_dir:
        required_codes.extend(codes_from_cache_dir(cache_dir))
    selected = merge_required_and_sampled_codes(all_codes, required_codes, limit=args.limit)
    output = write_pool_file(selected, args.output)
    print(
        json.dumps(
            {
                "date": args.date,
                "all_a_share_codes": len(all_codes),
                "selected_codes": len(selected),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
