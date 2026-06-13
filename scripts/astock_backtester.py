from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    top_n: int = 20
    fee_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    limit_threshold_pct: float = 9.8
    min_amount: float = 20_000_000.0
    min_turnover: float = 0.3
    market_filter_column: str | None = None
    execution_lag_days: int = 1
    rebalance_interval_days: int = 1
    group_column: str | None = None
    max_group_weight: float | None = None
    preserve_target_cash: bool = False


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    metrics: dict
    daily_positions: pd.DataFrame


def tradable_universe(frame: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    return (
        (frame["tradestatus"].astype(int) == 1)
        & (frame["isST"].astype(int) == 0)
        & (frame["amount"].astype(float) >= config.min_amount)
        & (frame["turn"].astype(float) >= config.min_turnover)
    )


def select_daily_positions(frame: pd.DataFrame, score_column: str, config: BacktestConfig) -> pd.DataFrame:
    universe = tradable_universe(frame, config)
    if config.market_filter_column:
        universe = universe & frame[config.market_filter_column].fillna(False).astype(bool)
    tradable = frame.loc[universe].copy()
    tradable = tradable.dropna(subset=[score_column])
    if tradable.empty:
        return pd.DataFrame(columns=["date", "code", "target_weight"])

    ranked = tradable.sort_values(["date", score_column, "amount"], ascending=[True, False, False])
    if config.group_column and config.max_group_weight is not None:
        nominal_weight = 1.0 / config.top_n
        max_per_group = int(np.floor(config.max_group_weight / nominal_weight + 1e-12))
        selected_rows = []
        for _, daily in ranked.groupby("date", sort=True):
            group_counts: dict[object, int] = {}
            kept = []
            for row in daily.itertuples(index=False):
                group_value = getattr(row, config.group_column)
                if group_counts.get(group_value, 0) >= max_per_group:
                    continue
                kept.append(row._asdict())
                group_counts[group_value] = group_counts.get(group_value, 0) + 1
                if len(kept) >= config.top_n:
                    break
            selected_rows.extend(kept)
        selected = pd.DataFrame(selected_rows, columns=ranked.columns)
    else:
        selected = ranked.groupby("date", group_keys=False).head(config.top_n).copy()
    if selected.empty:
        return pd.DataFrame(columns=["date", "code", "target_weight"])
    if config.preserve_target_cash:
        selected["target_weight"] = 1.0 / config.top_n
    else:
        counts = selected.groupby("date")["code"].transform("count")
        selected["target_weight"] = 1.0 / counts
    return selected[["date", "code", "target_weight"]]


def previous_day_selection_filters(frame: pd.DataFrame) -> pd.DataFrame:
    shifted = frame.copy()
    shifted = shifted.sort_values(["code", "date"])
    grouped = shifted.groupby("code", group_keys=False)
    for column in ["tradestatus", "isST", "amount", "turn"]:
        shifted[column] = grouped[column].shift(1)
    shifted["tradestatus"] = shifted["tradestatus"].fillna(0)
    shifted["isST"] = shifted["isST"].fillna(1)
    return shifted.sort_values(["date", "code"])


def shift_position_dates(positions: pd.DataFrame, dates: list[pd.Timestamp], lag_days: int) -> pd.DataFrame:
    if lag_days == 0:
        return positions.copy()
    shifted = positions.copy()
    shifted["date"] = shifted["date"].map(dict(zip(dates[:-lag_days], dates[lag_days:])))
    return shifted.dropna(subset=["date"])


def apply_rebalance_interval(target_weights: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    if interval_days <= 1 or target_weights.empty:
        return target_weights

    has_target = target_weights.sum(axis=1) > 0.0
    if not has_target.any():
        return target_weights

    anchor_index = int(np.flatnonzero(has_target.to_numpy())[0])
    rebalance_rows = list(range(anchor_index, len(target_weights), interval_days))
    rebalanced = pd.DataFrame(np.nan, index=target_weights.index, columns=target_weights.columns)
    if anchor_index > 0:
        rebalanced.iloc[:anchor_index] = 0.0
    rebalanced.iloc[rebalance_rows] = target_weights.iloc[rebalance_rows]
    return rebalanced.ffill().fillna(0.0)


def calculate_performance(equity: pd.DataFrame) -> dict:
    if equity.empty:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "trading_days": 0,
        }

    final_equity = float(equity["equity"].iloc[-1])
    total_return = final_equity - 1.0
    trading_days = len(equity)
    years = max(trading_days / 252.0, 1 / 252.0)
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
    drawdown = equity["equity"] / equity["equity"].cummax() - 1.0
    nonzero_returns = equity.loc[equity["daily_return"] != 0, "daily_return"]
    win_rate = float((nonzero_returns > 0).mean()) if not nonzero_returns.empty else 0.0
    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(abs(float(drawdown.min())) * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "trading_days": trading_days,
    }


def run_portfolio_backtest(panel: pd.DataFrame, score_column: str, config: BacktestConfig) -> BacktestResult:
    if config.execution_lag_days < 0:
        raise ValueError("execution_lag_days must be non-negative")
    if config.rebalance_interval_days < 1:
        raise ValueError("rebalance_interval_days must be positive")

    required = {
        "date",
        "code",
        "close",
        "pctChg",
        "tradestatus",
        "isST",
        "turn",
        "amount",
        score_column,
    }
    if config.market_filter_column:
        required.add(config.market_filter_column)
    if config.group_column:
        required.add(config.group_column)
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "code"])
    selection_data = previous_day_selection_filters(data) if config.execution_lag_days == 0 else data
    positions = select_daily_positions(selection_data, score_column, config)

    returns = data[["date", "code", "pctChg", "tradestatus"]].copy()
    returns["asset_return"] = returns["pctChg"].astype(float) / 100.0
    returns["can_buy"] = returns["pctChg"].astype(float) < config.limit_threshold_pct
    returns["can_sell"] = returns["pctChg"].astype(float) > -config.limit_threshold_pct
    returns["is_open"] = returns["tradestatus"].astype(int) == 1

    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())
    shifted_positions = shift_position_dates(positions, dates, config.execution_lag_days)
    raw_target = shifted_positions.pivot_table(index="date", columns="code", values="target_weight", fill_value=0.0)
    raw_target = raw_target.reindex(index=dates, columns=codes, fill_value=0.0)
    raw_target = apply_rebalance_interval(raw_target, config.rebalance_interval_days)
    asset_return = returns.pivot_table(index="date", columns="code", values="asset_return", fill_value=0.0)
    asset_return = asset_return.reindex(index=dates, columns=codes, fill_value=0.0)
    can_buy = returns.pivot_table(index="date", columns="code", values="can_buy", fill_value=False)
    can_buy = can_buy.reindex(index=dates, columns=codes, fill_value=False).astype(bool)
    can_sell = returns.pivot_table(index="date", columns="code", values="can_sell", fill_value=False)
    can_sell = can_sell.reindex(index=dates, columns=codes, fill_value=False).astype(bool)
    is_open = returns.pivot_table(index="date", columns="code", values="is_open", fill_value=False)
    is_open = is_open.reindex(index=dates, columns=codes, fill_value=False).astype(bool)

    target_values = raw_target.to_numpy(dtype=float)
    asset_return_values = asset_return.to_numpy(dtype=float)
    can_buy_values = can_buy.to_numpy(dtype=bool)
    can_sell_values = can_sell.to_numpy(dtype=bool)
    is_open_values = is_open.to_numpy(dtype=bool)

    previous_weight = np.zeros(len(codes), dtype=float)
    daily_rows = []
    for row_index, date in enumerate(dates):
        target = target_values[row_index]
        held = previous_weight > 0.0
        sell_blocked = held & (~is_open_values[row_index] | ~can_sell_values[row_index])
        blocked_weight = np.where(sell_blocked, previous_weight, 0.0)
        remaining_weight = max(1.0 - float(blocked_weight.sum()), 0.0)

        target_candidates = (target > 0.0) & ~sell_blocked & is_open_values[row_index] & (held | can_buy_values[row_index])
        target_weight = np.where(target_candidates, target, 0.0)
        target_sum = float(target_weight.sum())
        if target_sum > 0.0 and remaining_weight > 0.0:
            if config.preserve_target_cash:
                scale = min(1.0, remaining_weight / target_sum)
                executable_weight = blocked_weight + target_weight * scale
            else:
                executable_weight = blocked_weight + target_weight / target_sum * remaining_weight
        else:
            executable_weight = blocked_weight.copy()

        weighted_return = executable_weight * asset_return_values[row_index]
        delta = executable_weight - previous_weight
        daily_rows.append(
            {
                "date": date,
                "gross_return": float(weighted_return.sum()),
                "position_count": int((executable_weight > 0.0).sum()),
                "buy_turnover": float(np.clip(delta, 0.0, None).sum()),
                "sell_turnover": float((-np.clip(delta, None, 0.0)).sum()),
            }
        )
        previous_weight = executable_weight

    daily = pd.DataFrame(daily_rows)
    daily["turnover"] = daily["buy_turnover"] + daily["sell_turnover"]
    daily["cost"] = (
        daily["buy_turnover"] * (config.fee_rate + config.slippage_rate)
        + daily["sell_turnover"] * (config.fee_rate + config.stamp_tax_rate + config.slippage_rate)
    )
    daily["daily_return"] = daily["gross_return"] - daily["cost"]
    daily.loc[daily.index == 0, "daily_return"] = 0.0
    daily["equity"] = (1.0 + daily["daily_return"]).cumprod()
    equity = daily[["date", "daily_return", "equity", "position_count", "turnover", "cost"]]
    return BacktestResult(equity=equity, metrics=calculate_performance(equity), daily_positions=positions)
