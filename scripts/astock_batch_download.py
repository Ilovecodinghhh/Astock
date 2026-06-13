from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.astock_baostock import cache_path, iter_codes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "user_data" / "astock_baostock_liquid_60"


def pending_codes(codes: list[str], output_dir: Path, frequency: str) -> list[str]:
    return [code for code in codes if not cache_path(output_dir, code, frequency).exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download baostock data one code per subprocess.")
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", default="d")
    parser.add_argument("--per-code-timeout", type=int, default=60)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    codes = pending_codes(iter_codes([], args.pool_file), args.output_dir, args.frequency)
    if args.limit:
        codes = codes[: args.limit]

    results = []
    script = ROOT / "scripts" / "astock_baostock.py"
    for code in codes:
        cmd = [
            sys.executable,
            str(script),
            "--code",
            code,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--output-dir",
            str(args.output_dir),
            "--frequency",
            args.frequency,
            "--skip-existing",
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.per_code_timeout,
            )
            results.append(
                {
                    "code": code,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-500:],
                    "stderr_tail": completed.stderr[-500:],
                }
            )
        except subprocess.TimeoutExpired:
            results.append({"code": code, "returncode": "timeout", "stdout_tail": "", "stderr_tail": ""})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    failures = [row for row in results if row["returncode"] != 0]
    print(json.dumps({"attempted": len(results), "failures": len(failures)}, ensure_ascii=False), flush=True)
    return 1 if failures and len(failures) == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
