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
DEFAULT_OUTPUT = ROOT / "user_data" / "astock_metadata.feather"


def normalize_industry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["industry"] = normalized["industry"].replace("", pd.NA).fillna("unknown")
    columns = ["code", "industry", "industryClassification"]
    return normalized[columns].drop_duplicates("code").sort_values("code").reset_index(drop=True)


def normalize_basic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["ipoDate"] = pd.to_datetime(normalized["ipoDate"].replace("", pd.NA), errors="coerce")
    normalized["outDate"] = pd.to_datetime(normalized["outDate"].replace("", pd.NA), errors="coerce")
    columns = ["code", "ipoDate", "outDate", "status"]
    return normalized[columns].drop_duplicates("code").sort_values("code").reset_index(drop=True)


def merge_metadata(panel: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = panel.merge(metadata, on="code", how="left")
    if "industry" not in merged:
        merged["industry"] = "unknown"
    else:
        merged["industry"] = merged["industry"].fillna("unknown")
    return merged.sort_values(["date", "code"]).reset_index(drop=True)


def query_industry_metadata(date: str) -> pd.DataFrame:
    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        industry_result = bs.query_stock_industry(date=date)
        if industry_result.error_code != "0":
            raise RuntimeError(f"baostock query_stock_industry failed: {industry_result.error_msg}")
        industry = normalize_industry_frame(baostock_result_to_frame(industry_result))
    finally:
        bs.logout()
    return industry


def query_basic_metadata() -> pd.DataFrame:
    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        basic_result = bs.query_stock_basic()
        if basic_result.error_code != "0":
            raise RuntimeError(f"baostock query_stock_basic failed: {basic_result.error_msg}")
        basic = normalize_basic_frame(baostock_result_to_frame(basic_result))
    finally:
        bs.logout()
    return basic


def query_metadata(date: str) -> pd.DataFrame:
    industry = query_industry_metadata(date)
    basic = query_basic_metadata()
    return industry.merge(basic, on="code", how="outer").sort_values("code").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download A-share industry and basic metadata from baostock.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = query_metadata(args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_feather(args.output)
    print(json.dumps({"rows": len(metadata), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
