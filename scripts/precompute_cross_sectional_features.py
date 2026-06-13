from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "user_data" / "config_binance_futures_expanded.json"
DEFAULT_DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
DEFAULT_OUTPUT_DIR = ROOT / "user_data" / "cross_sectional_features"
DEFAULT_TIMEFRAME = "4h"


def pair_to_slug(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def pair_to_futures_ohlcv_path(data_dir: Path, pair: str, timeframe: str) -> Path:
    return data_dir / f"{pair_to_slug(pair)}-{timeframe}-futures.feather"


def pair_to_feature_path(output_dir: Path, pair: str, timeframe: str) -> Path:
    return output_dir / f"{pair_to_slug(pair)}-{timeframe}-cross_sectional.feather"


def load_pair_whitelist(config_path: Path) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return list(config["exchange"]["pair_whitelist"])


def prepare_pair_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe[["date", "close"]].copy()
    prepared["momentum_42"] = prepared["close"] / prepared["close"].shift(42) - 1.0
    prepared["momentum_126"] = prepared["close"] / prepared["close"].shift(126) - 1.0
    prepared["volatility_63"] = prepared["close"].pct_change().rolling(63, min_periods=63).std()
    prepared["rs_score"] = (
        prepared["momentum_42"] * 0.65 + prepared["momentum_126"] * 0.35
    ) / prepared["volatility_63"].replace(0, pd.NA)
    prepared["ema_100"] = prepared["close"].ewm(span=100, adjust=False, min_periods=100).mean()
    prepared["trend_flag"] = (prepared["close"] > prepared["ema_100"]).astype(float)
    return prepared[["date", "rs_score", "trend_flag"]]


def load_prepared_frames(pairs: list[str], data_dir: Path, timeframe: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        path = pair_to_futures_ohlcv_path(data_dir, pair, timeframe)
        if not path.exists():
            continue
        frames[pair] = prepare_pair_frame(pd.read_feather(path))
    return frames


def compute_cross_sectional_features(pair_frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    scores = pd.DataFrame()
    trend_flags = pd.DataFrame()
    for pair, frame in pair_frames.items():
        indexed = frame.set_index("date").sort_index()
        scores[pair] = indexed["rs_score"].shift(1)
        trend_flags[pair] = indexed["trend_flag"].shift(1)

    if scores.empty:
        return {}

    ranks = scores.rank(axis=1, ascending=False, pct=True, method="first")
    breadth = trend_flags.mean(axis=1, skipna=True).fillna(0.0)

    features: dict[str, pd.DataFrame] = {}
    for pair in pair_frames:
        pair_features = pd.DataFrame(
            {
                "date": ranks.index,
                "rs_rank_pct": ranks[pair].to_numpy(),
                "market_breadth": breadth.to_numpy(),
            }
        )
        features[pair] = pair_features
    return features


def write_feature_files(features: Mapping[str, pd.DataFrame], output_dir: Path, timeframe: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for pair, dataframe in features.items():
        path = pair_to_feature_path(output_dir, pair, timeframe)
        dataframe.to_feather(path)
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute lagged cross-sectional futures features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_pair_whitelist(args.config)
    frames = load_prepared_frames(pairs, args.data_dir, args.timeframe)
    features = compute_cross_sectional_features(frames)
    written = write_feature_files(features, args.output_dir, args.timeframe)
    print(
        json.dumps(
            {
                "pairs_requested": len(pairs),
                "pairs_loaded": len(frames),
                "files_written": len(written),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
