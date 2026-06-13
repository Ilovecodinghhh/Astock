from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FUTURES_DATA = ROOT / "user_data" / "data" / "binance" / "futures"

PERIODS = {
    "train_2021_2022": ("2021-01-01", "2022-12-31"),
    "validation_2023": ("2023-01-01", "2023-12-31"),
    "validation_2024": ("2024-01-01", "2024-12-31"),
    "validation_2025_2026": ("2025-01-01", "2026-05-31"),
}


def cagr(first: float, last: float, days: int) -> float:
    if first <= 0 or last <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    return (last / first) ** (1 / years) - 1


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values / values.iloc[0]
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def pair_name(path: Path) -> str:
    return path.name.split("-")[0].replace("_USDT_USDT", "/USDT:USDT")


def load_daily_futures() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(FUTURES_DATA.glob("*-1d-futures.feather")):
        frame = pd.read_feather(path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        data[pair_name(path)] = frame.sort_values("date")
    return data


def summarize_buy_hold(data: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for pair, frame in data.items():
        row: dict[str, str | float] = {"pair": pair}
        for name, (start, end) in PERIODS.items():
            window = frame[(frame["date"] >= start) & (frame["date"] <= end)]
            if len(window) < 2:
                row[f"{name}_cagr_pct"] = math.nan
                row[f"{name}_dd_pct"] = math.nan
                continue
            days = int((window["date"].iloc[-1] - window["date"].iloc[0]).days)
            row[f"{name}_cagr_pct"] = round(cagr(float(window["close"].iloc[0]), float(window["close"].iloc[-1]), days) * 100, 2)
            row[f"{name}_dd_pct"] = round(max_drawdown(window["close"]) * 100, 2)
        rows.append(row)
    return rows


def summarize_equal_weight(data: dict[str, pd.DataFrame]) -> list[dict]:
    returns = []
    for pair, frame in data.items():
        pair_returns = frame[["date", "close"]].copy()
        pair_returns[pair] = pair_returns["close"].pct_change()
        returns.append(pair_returns[["date", pair]])
    merged = returns[0]
    for frame in returns[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date")
    merged["portfolio_return"] = merged.drop(columns=["date"]).mean(axis=1, skipna=True).fillna(0.0)
    merged["equity"] = (1.0 + merged["portfolio_return"]).cumprod()

    rows = []
    for name, (start, end) in PERIODS.items():
        window = merged[(merged["date"] >= start) & (merged["date"] <= end)]
        if len(window) < 2:
            continue
        days = int((window["date"].iloc[-1] - window["date"].iloc[0]).days)
        rows.append(
            {
                "period": name,
                "equal_weight_cagr_pct": round(cagr(float(window["equity"].iloc[0]), float(window["equity"].iloc[-1]), days) * 100, 2),
                "equal_weight_dd_pct": round(max_drawdown(window["equity"]) * 100, 2),
            }
        )
    return rows


def summarize_daily_trend(data: dict[str, pd.DataFrame], leverage: float) -> list[dict]:
    returns = []
    for pair, frame in data.items():
        signal_frame = frame[["date", "close"]].copy()
        signal_frame["ema_50"] = signal_frame["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        signal_frame["ema_200"] = signal_frame["close"].ewm(span=200, adjust=False, min_periods=200).mean()
        signal_frame["position"] = 0.0
        signal_frame.loc[signal_frame["ema_50"] > signal_frame["ema_200"], "position"] = 1.0
        signal_frame.loc[signal_frame["ema_50"] < signal_frame["ema_200"], "position"] = -1.0
        signal_frame[pair] = signal_frame["position"].shift(1).fillna(0.0) * signal_frame["close"].pct_change().fillna(0.0) * leverage
        returns.append(signal_frame[["date", pair]])
    merged = returns[0]
    for frame in returns[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date")
    merged["portfolio_return"] = merged.drop(columns=["date"]).mean(axis=1, skipna=True).fillna(0.0)
    merged["portfolio_return"] = merged["portfolio_return"].clip(lower=-0.35, upper=0.35)
    merged["equity"] = (1.0 + merged["portfolio_return"]).cumprod()

    rows = []
    for name, (start, end) in PERIODS.items():
        window = merged[(merged["date"] >= start) & (merged["date"] <= end)]
        if len(window) < 2:
            continue
        days = int((window["date"].iloc[-1] - window["date"].iloc[0]).days)
        rows.append(
            {
                "period": name,
                "daily_trend_leverage": leverage,
                "daily_trend_cagr_pct": round(cagr(float(window["equity"].iloc[0]), float(window["equity"].iloc[-1]), days) * 100, 2),
                "daily_trend_dd_pct": round(max_drawdown(window["equity"]) * 100, 2),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect broad market returns in local Binance futures data.")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    data = load_daily_futures()
    buy_hold = summarize_buy_hold(data)
    print("Equal-weight futures buy-and-hold:")
    for row in summarize_equal_weight(data):
        print(row)

    print("\nEqual-weight daily EMA50/EMA200 long-short trend:")
    for row in summarize_daily_trend(data, leverage=1.0):
        print(row)
    for row in summarize_daily_trend(data, leverage=3.0):
        print(row)

    print("\nTop pairs by 2023 CAGR:")
    for row in sorted(buy_hold, key=lambda item: item["validation_2023_cagr_pct"], reverse=True)[: args.top]:
        print(row)

    print("\nTop pairs by 2025-2026 CAGR:")
    for row in sorted(buy_hold, key=lambda item: item["validation_2025_2026_cagr_pct"], reverse=True)[: args.top]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
