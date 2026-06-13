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
DEFAULT_OUTPUT_DIR = ROOT / "user_data" / "astock_indexes"
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def normalize_index_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values("date").reset_index(drop=True)


def add_index_trend_filter(
    panel: pd.DataFrame,
    index_frame: pd.DataFrame,
    *,
    short_window: int = 60,
    long_window: int = 120,
    output_column: str = "index_risk_on",
) -> pd.DataFrame:
    if short_window < 1 or long_window < 1:
        raise ValueError("index windows must be positive")

    index_data = normalize_index_frame(index_frame)
    short_ma = index_data["close"].rolling(short_window, min_periods=short_window).mean()
    long_ma = index_data["close"].rolling(long_window, min_periods=long_window).mean()
    raw_signal = (index_data["close"] > long_ma) & (short_ma > long_ma)
    index_signal = raw_signal.shift(1).fillna(False)
    signal_by_date = pd.Series(index_signal.to_numpy(dtype=bool), index=index_data["date"])

    merged = panel.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    dates = pd.Index(sorted(merged["date"].unique()))
    aligned = signal_by_date.reindex(dates).ffill().fillna(False)
    merged[output_column] = merged["date"].map(aligned).fillna(False).astype(bool)
    return merged.sort_values(["date", "code"]).reset_index(drop=True)


def index_cache_path(output_dir: Path, code: str) -> Path:
    return output_dir / f"{code}-d.feather"


def download_index_history(*, code: str, start_date: str, end_date: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        result = bs.query_history_k_data_plus(
            code,
            INDEX_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        if result.error_code != "0":
            raise RuntimeError(f"baostock index query failed for {code}: {result.error_msg}")
        frame = normalize_index_frame(baostock_result_to_frame(result))
    finally:
        bs.logout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = index_cache_path(output_dir, code)
    frame.to_feather(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache A-share index history from baostock.")
    parser.add_argument("--code", default="sh.000300")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = download_index_history(
        code=args.code,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
