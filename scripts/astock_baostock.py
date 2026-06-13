from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "user_data" / "astock_baostock"
FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
    "tradestatus,pctChg,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM"
)


def normalize_baostock_code(code: str) -> str:
    if code.startswith(("sh.", "sz.")):
        return code
    if code.startswith(("5", "6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def cache_path(output_dir: Path, code: str, frequency: str) -> Path:
    return output_dir / f"{normalize_baostock_code(code)}-{frequency}.feather"


def should_skip_download(output_dir: Path, code: str, frequency: str) -> bool:
    return cache_path(output_dir, code, frequency).exists()


def iter_codes(cli_codes: list[str] | None, pool_file: Path | None) -> list[str]:
    codes = list(cli_codes or [])
    if pool_file:
        for line in pool_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                codes.append(stripped)
    seen: set[str] = set()
    unique: list[str] = []
    for code in codes:
        if code not in seen:
            unique.append(code)
            seen.add(code)
    return unique


def import_baostock():
    try:
        import baostock as bs
    except ModuleNotFoundError as exc:
        raise RuntimeError("baostock is not installed. Install it with `pip install baostock`.") from exc
    return bs


def baostock_result_to_frame(result) -> pd.DataFrame:
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def download_history(
    *,
    code: str,
    start_date: str,
    end_date: str,
    output_dir: Path = DEFAULT_OUTPUT,
    frequency: str = "d",
    adjustflag: str = "2",
) -> Path:
    bs = import_baostock()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        normalized = normalize_baostock_code(code)
        result = bs.query_history_k_data_plus(
            normalized,
            FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )
        if result.error_code != "0":
            raise RuntimeError(f"baostock query failed for {normalized}: {result.error_msg}")
        frame = baostock_result_to_frame(result)
    finally:
        bs.logout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(output_dir, code, frequency)
    frame.to_feather(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download A-share history from baostock.")
    parser.add_argument("--code", action="append", default=[], help="Stock code, e.g. 600000 or sh.600000.")
    parser.add_argument("--pool-file", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", default="d")
    parser.add_argument("--adjustflag", default="2", help="2 means front-adjusted in baostock.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    codes = iter_codes(args.code, args.pool_file)
    if not codes:
        raise SystemExit("No codes supplied. Use --code or --pool-file.")

    written = []
    failed = []
    for code in codes:
        if args.skip_existing and should_skip_download(args.output_dir, code, args.frequency):
            written.append(str(cache_path(args.output_dir, code, args.frequency)))
            continue
        try:
            written.append(
                str(
                    download_history(
                        code=code,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        output_dir=args.output_dir,
                        frequency=args.frequency,
                        adjustflag=args.adjustflag,
                    )
                )
            )
        except Exception as exc:
            failed.append({"code": code, "error": str(exc)})
        time.sleep(args.sleep)
    print(json.dumps({"files_written": written, "failed": failed}, ensure_ascii=False))
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
