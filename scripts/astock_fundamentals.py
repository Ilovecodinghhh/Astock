from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.astock_baostock import baostock_result_to_frame, import_baostock, iter_codes


METADATA_COLUMNS = {"code", "pubDate", "statDate", "report_year", "report_quarter", "available_date"}
REPORT_KEY_COLUMNS = ["code", "statDate"]
REPORT_QUERY_NAMES = (
    "query_profit_data",
    "query_growth_data",
    "query_cash_flow_data",
    "query_balance_data",
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "user_data" / "astock_fundamentals.feather"


def combine_report_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame(columns=["code", "pubDate", "statDate"])

    combined = pd.concat(non_empty_frames, ignore_index=True, sort=False)
    missing = set(REPORT_KEY_COLUMNS) - set(combined.columns)
    if missing:
        raise ValueError(f"Missing report columns: {', '.join(sorted(missing))}")
    if "pubDate" not in combined:
        combined["pubDate"] = pd.NA

    metric_columns = [column for column in combined.columns if column not in {"code", "pubDate", "statDate"}]
    rows = []
    for (code, stat_date), group in combined.groupby(REPORT_KEY_COLUMNS, sort=True, dropna=False):
        row = {"code": code, "statDate": stat_date}
        pub_dates = group["pubDate"].replace("", pd.NA).dropna()
        row["pubDate"] = str(pub_dates.max()) if not pub_dates.empty else pd.NA
        for column in metric_columns:
            values = group[column].replace("", pd.NA).dropna()
            row[column] = values.iloc[-1] if not values.empty else pd.NA
        rows.append(row)
    columns = ["code", "pubDate", "statDate", *metric_columns]
    return pd.DataFrame(rows, columns=columns).sort_values(REPORT_KEY_COLUMNS).reset_index(drop=True)


def report_available_date(year: int, quarter: int, pub_date: str | pd.Timestamp | None = None) -> pd.Timestamp:
    quarter_month_day = {
        1: (5, 1),
        2: (9, 1),
        3: (11, 1),
        4: (5, 1),
    }
    if quarter not in quarter_month_day:
        raise ValueError("quarter must be 1, 2, 3, or 4")

    available_year = int(year) + 1 if int(quarter) == 4 else int(year)
    month, day = quarter_month_day[int(quarter)]
    conservative = pd.Timestamp(year=available_year, month=month, day=day)
    parsed_pub_date = pd.to_datetime(pub_date, errors="coerce")
    if pd.isna(parsed_pub_date):
        return conservative
    return max(conservative, pd.Timestamp(parsed_pub_date).normalize())


def _quarter_from_stat_date(stat_date: pd.Series) -> pd.Series:
    month_to_quarter = {3: 1, 6: 2, 9: 3, 12: 4}
    quarter = stat_date.dt.month.map(month_to_quarter)
    if quarter.isna().any():
        raise ValueError("statDate must be quarter-end dates")
    return quarter.astype(int)


def normalize_fundamental_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=sorted(METADATA_COLUMNS))

    required = {"code", "statDate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing fundamental columns: {', '.join(sorted(missing))}")

    normalized = frame.copy()
    normalized["statDate"] = pd.to_datetime(normalized["statDate"], errors="coerce")
    normalized = normalized.dropna(subset=["code", "statDate"]).copy()
    if "pubDate" in normalized:
        normalized["pubDate"] = pd.to_datetime(normalized["pubDate"].replace("", pd.NA), errors="coerce")
    else:
        normalized["pubDate"] = pd.NaT
    normalized["report_year"] = normalized["statDate"].dt.year.astype(int)
    normalized["report_quarter"] = _quarter_from_stat_date(normalized["statDate"])
    normalized["available_date"] = [
        report_available_date(year, quarter, pub_date)
        for year, quarter, pub_date in zip(
            normalized["report_year"],
            normalized["report_quarter"],
            normalized["pubDate"],
        )
    ]

    for column in normalized.columns:
        if column not in METADATA_COLUMNS and column not in {"pubDate", "statDate"}:
            normalized[column] = pd.to_numeric(normalized[column].replace("", pd.NA), errors="coerce")

    normalized = normalized.sort_values(["code", "statDate", "pubDate", "available_date"])
    normalized = normalized.drop_duplicates(["code", "statDate"], keep="last")
    return normalized.sort_values(["code", "available_date"]).reset_index(drop=True)


def merge_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        return panel.copy().sort_values(["date", "code"]).reset_index(drop=True)

    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"])
    right = fundamentals.copy()
    right["available_date"] = pd.to_datetime(right["available_date"])
    right = right.sort_values(["code", "available_date"])
    output_frames = []

    for code, panel_group in left.sort_values(["code", "date"]).groupby("code", sort=False):
        fundamentals_group = right.loc[right["code"] == code].sort_values("available_date")
        if fundamentals_group.empty:
            output_frames.append(panel_group)
            continue
        merged = pd.merge_asof(
            panel_group.sort_values("date"),
            fundamentals_group.drop(columns=["code"]).sort_values("available_date"),
            left_on="date",
            right_on="available_date",
            direction="backward",
        )
        merged["code"] = code
        output_frames.append(merged)

    return pd.concat(output_frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def read_fundamentals(path: Path) -> pd.DataFrame:
    return pd.read_feather(path)


def query_code_quarter_reports(bs, code: str, year: int, quarter: int) -> pd.DataFrame:
    frames = []
    for query_name in REPORT_QUERY_NAMES:
        result = getattr(bs, query_name)(code, year=year, quarter=quarter)
        if result.error_code != "0":
            raise RuntimeError(f"{query_name} failed for {code} {year}Q{quarter}: {result.error_msg}")
        frames.append(baostock_result_to_frame(result))
    return combine_report_frames(frames)


def query_fundamentals(
    *,
    codes: Iterable[str],
    start_year: int,
    end_year: int,
    quarters: Iterable[int] = (1, 2, 3, 4),
    sleep: float = 0.0,
) -> pd.DataFrame:
    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")

    raw_frames = []
    try:
        for code in codes:
            for year in range(start_year, end_year + 1):
                for quarter in quarters:
                    raw_frames.append(query_code_quarter_reports(bs, code, year, int(quarter)))
                    if sleep > 0:
                        time.sleep(sleep)
    finally:
        bs.logout()

    combined = combine_report_frames(raw_frames)
    return normalize_fundamental_frame(combined)


def parse_quarters(value: str) -> list[int]:
    quarters = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [quarter for quarter in quarters if quarter not in {1, 2, 3, 4}]
    if invalid:
        raise argparse.ArgumentTypeError("quarters must contain only 1, 2, 3, and 4")
    return quarters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download A-share quarterly fundamentals from baostock.")
    parser.add_argument("--code", action="append", default=[], help="Stock code, e.g. sh.600000.")
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--quarters", type=parse_quarters, default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    codes = iter_codes(args.code, args.pool_file)
    if args.limit:
        codes = codes[: args.limit]
    if not codes:
        raise SystemExit("No codes supplied. Use --code or --pool-file.")
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be greater than or equal to --start-year")

    fundamentals = query_fundamentals(
        codes=codes,
        start_year=args.start_year,
        end_year=args.end_year,
        quarters=args.quarters,
        sleep=args.sleep,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fundamentals.to_feather(args.output)
    print(json.dumps({"rows": len(fundamentals), "codes": len(codes), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
