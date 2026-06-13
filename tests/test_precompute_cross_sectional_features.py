from pathlib import Path

import pandas as pd

from scripts.precompute_cross_sectional_features import (
    compute_cross_sectional_features,
    pair_to_futures_ohlcv_path,
    write_feature_files,
)


def test_pair_to_futures_ohlcv_path_uses_freqtrade_futures_filename() -> None:
    path = pair_to_futures_ohlcv_path(Path("user_data/data/binance/futures"), "BTC/USDT:USDT", "4h")

    assert path == Path("user_data/data/binance/futures/BTC_USDT_USDT-4h-futures.feather")


def test_compute_cross_sectional_features_uses_prior_candle_only() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    pair_a = pd.DataFrame(
        {
            "date": dates,
            "rs_score": [1.0, 1.0, -100.0],
            "trend_flag": [1.0, 1.0, 0.0],
        }
    )
    pair_b = pd.DataFrame(
        {
            "date": dates,
            "rs_score": [0.0, 0.0, 100.0],
            "trend_flag": [1.0, 1.0, 0.0],
        }
    )

    features = compute_cross_sectional_features({"A/USDT:USDT": pair_a, "B/USDT:USDT": pair_b})

    a_features = features["A/USDT:USDT"].set_index("date")
    assert a_features.loc[dates[2], "rs_rank_pct"] == 0.5
    assert a_features.loc[dates[2], "market_breadth"] == 1.0


def test_write_feature_files_outputs_one_file_per_pair(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
    features = {
        "BTC/USDT:USDT": pd.DataFrame(
            {
                "date": dates,
                "rs_rank_pct": [1.0, 0.5],
                "market_breadth": [0.0, 1.0],
            }
        ),
        "ETH/USDT:USDT": pd.DataFrame(
            {
                "date": dates,
                "rs_rank_pct": [1.0, 1.0],
                "market_breadth": [0.0, 1.0],
            }
        ),
    }

    written = write_feature_files(features, tmp_path, "4h")

    assert sorted(path.name for path in written) == [
        "BTC_USDT_USDT-4h-cross_sectional.feather",
        "ETH_USDT_USDT-4h-cross_sectional.feather",
    ]
    output = pd.read_feather(tmp_path / "BTC_USDT_USDT-4h-cross_sectional.feather")
    assert list(output.columns) == ["date", "rs_rank_pct", "market_breadth"]
