from __future__ import annotations

import pandas as pd


def _rank_by_date(dataframe: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    return dataframe.groupby("date")[column].rank(pct=True, ascending=ascending)


def _rolling_mean(grouped: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    return grouped.rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)


def _rolling_std(grouped: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    return grouped.rolling(window, min_periods=window).std().reset_index(level=0, drop=True)


def _numeric_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe:
        return pd.Series(pd.NA, index=dataframe.index, dtype="Float64")
    return pd.to_numeric(dataframe[column], errors="coerce")


def add_strategy_scores(
    panel: pd.DataFrame,
    *,
    momentum_window: int = 60,
    volatility_window: int = 20,
    breakout_window: int = 20,
    reversal_window: int = 10,
    breadth_threshold: float = 0.45,
    min_listing_days: int = 120,
) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["code", "date"])
    grouped = data.groupby("code", group_keys=False)

    momentum = grouped["close"].pct_change(momentum_window)
    momentum_20 = grouped["close"].pct_change(20)
    momentum_60 = grouped["close"].pct_change(60)
    momentum_120 = grouped["close"].pct_change(120)
    positive_momentum = momentum > 0
    breadth = positive_momentum.groupby(data["date"]).mean()
    lagged_breadth = breadth.shift(1).reindex(data["date"]).to_numpy()
    data["market_breadth"] = lagged_breadth
    data["risk_on"] = data["market_breadth"].fillna(0.0) >= breadth_threshold
    breadth_20 = (momentum_20 > 0).groupby(data["date"]).mean()
    mean_momentum_20 = momentum_20.groupby(data["date"]).mean()
    strict_risk_on = (breadth >= breadth_threshold) & (breadth_20 >= 0.50) & (mean_momentum_20 > 0)
    lagged_strict_risk_on = strict_risk_on.shift(1).reindex(data["date"]).fillna(False).to_numpy()
    data["risk_on_strict"] = lagged_strict_risk_on
    recovery_risk_on = (breadth_20 >= 0.45) & (mean_momentum_20 > 0) & (breadth >= breadth_threshold * 0.60)
    lagged_recovery_risk_on = recovery_risk_on.shift(1).reindex(data["date"]).fillna(False).to_numpy()
    data["risk_on_recovery"] = lagged_recovery_risk_on
    daily_return = grouped["close"].pct_change()
    volatility = daily_return.rolling(volatility_window, min_periods=volatility_window).std().reset_index(level=0, drop=True)
    volatility_60 = daily_return.rolling(60, min_periods=60).std().reset_index(level=0, drop=True)
    turnover_mean = _rolling_mean(grouped["turn"], volatility_window)
    cached_listing_age = grouped.cumcount() + 1
    if "ipoDate" in data:
        ipo_date = pd.to_datetime(data["ipoDate"], errors="coerce")
        calendar_listing_age = (data["date"] - ipo_date).dt.days + 1
        data["listing_age_days"] = calendar_listing_age.where(ipo_date.notna(), cached_listing_age)
    else:
        data["listing_age_days"] = cached_listing_age
    turn_ratio = data["turn"].astype(float) / 100.0
    data["float_mkt_value_proxy"] = data["amount"].astype(float) / turn_ratio.where(turn_ratio > 0)
    size_rank = data["float_mkt_value_proxy"].groupby(data["date"]).rank(pct=True, ascending=True)
    liquidity_rank_high = data["amount"].groupby(data["date"]).rank(pct=True, ascending=True)
    mature_liquid_gate = (data["listing_age_days"] >= min_listing_days) & (size_rank >= 0.20) & (data["turn"] > 0)
    data["raw_momentum_lowvol_score"] = momentum / volatility.replace(0, pd.NA) + turnover_mean.rank(pct=True) * 0.1
    data["raw_trend_strength_score"] = momentum_20 * 0.2 + momentum_60 * 0.5 + momentum_120 * 0.3

    previous_high = grouped["high"].rolling(breakout_window, min_periods=breakout_window).max().reset_index(level=0, drop=True)
    previous_high = previous_high.groupby(data["code"]).shift(1)
    volume_mean = grouped["volume"].rolling(volatility_window, min_periods=volatility_window).mean().reset_index(level=0, drop=True)
    breakout_strength = data["close"] / previous_high - 1.0
    volume_expansion = data["volume"] / volume_mean.replace(0, pd.NA)
    data["raw_breakout_score"] = breakout_strength * 5.0 + volume_expansion

    drawdown = data["close"] / grouped["close"].rolling(reversal_window, min_periods=reversal_window).max().reset_index(
        level=0, drop=True
    ) - 1.0
    bounce = grouped["close"].pct_change(3)
    liquidity_rank = _rank_by_date(data, "amount", ascending=False)
    data["raw_reversal_quality_score"] = -drawdown + bounce * 0.5 + liquidity_rank * 0.1

    ma60 = _rolling_mean(grouped["close"], 60)
    ma120 = _rolling_mean(grouped["close"], 120)
    ma20 = _rolling_mean(grouped["close"], 20)
    absolute_trend_gate = (data["close"] > ma60) & (data["close"] > ma120) & (momentum_60 > 0) & (momentum_120 > 0)
    absolute_trend = momentum_20 * 0.2 + momentum_60 * 0.35 + momentum_120 * 0.45
    data["raw_absolute_trend_score"] = absolute_trend.where(absolute_trend_gate)

    momentum_60_rank = momentum_60.groupby(data["date"]).rank(pct=True)
    momentum_120_rank = momentum_120.groupby(data["date"]).rank(pct=True)
    volatility_rank = volatility_60.groupby(data["date"]).rank(pct=True, ascending=True)
    low_volatility_rank = volatility_60.groupby(data["date"]).rank(pct=True, ascending=False)
    liquidity_rank = _rank_by_date(data, "amount", ascending=False)
    data["raw_relative_strength_quality_score"] = (
        momentum_60_rank * 0.35 + momentum_120_rank * 0.35 + volatility_rank * 0.15 + liquidity_rank * 0.15
    ).where(absolute_trend_gate)

    recent_drawdown = data["close"] / grouped["close"].rolling(60, min_periods=60).max().reset_index(level=0, drop=True) - 1.0
    data["raw_risk_adjusted_trend_score"] = (
        (momentum_60 * 0.45 + momentum_120 * 0.55) / volatility_60.replace(0, pd.NA)
        + recent_drawdown * 0.5
        + liquidity_rank * 0.1
    ).where(absolute_trend_gate)

    momentum_20_rank = momentum_20.groupby(data["date"]).rank(pct=True)
    steady_trend_gate = absolute_trend_gate & (data["close"] > ma20) & (ma20 > ma60)
    data["raw_steady_uptrend_score"] = (
        momentum_20_rank * 0.20
        + momentum_60_rank * 0.30
        + momentum_120_rank * 0.25
        + volatility_rank * 0.15
        + liquidity_rank * 0.10
        + recent_drawdown.clip(lower=-0.30, upper=0.0) * 0.30
    ).where(steady_trend_gate)

    pullback_to_ma20 = (ma20 / data["close"] - 1.0).clip(lower=-0.10, upper=0.20)
    data["raw_trend_pullback_score"] = (
        momentum_60_rank * 0.30
        + momentum_120_rank * 0.25
        + volatility_rank * 0.15
        + liquidity_rank * 0.10
        + pullback_to_ma20 * 1.50
        + recent_drawdown.clip(lower=-0.25, upper=0.0).abs() * 0.60
    ).where(absolute_trend_gate & (data["close"] > ma120))

    volatility_ratio = volatility / volatility_60.replace(0, pd.NA)
    compression_rank = volatility_ratio.groupby(data["date"]).rank(pct=True, ascending=True)
    price_range_20 = grouped["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    price_range_low_20 = grouped["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    range_compression = (price_range_20 / price_range_low_20.replace(0, pd.NA) - 1.0).groupby(data["date"]).rank(
        pct=True,
        ascending=True,
    )
    data["raw_squeeze_breakout_score"] = (
        breakout_strength * 4.0
        + volume_expansion.groupby(data["date"]).rank(pct=True) * 0.20
        + compression_rank * 0.40
        + range_compression * 0.20
        + momentum_20_rank * 0.20
    ).where((data["close"] > ma60) & (momentum_20 > 0))

    data["raw_volatility_contraction_trend_score"] = (
        compression_rank * 0.30
        + range_compression * 0.25
        + momentum_60_rank * 0.20
        + momentum_20_rank * 0.15
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & (data["close"] > ma60) & (momentum_20 > 0) & (momentum_60 > 0) & (volatility_ratio < 1.0))

    conservative_trend_gate = mature_liquid_gate & (data["close"] > ma120) & (momentum_60 > 0)
    data["raw_balanced_core_score"] = (
        momentum_60_rank * 0.25
        + momentum_120_rank * 0.20
        + volatility_rank * 0.25
        + size_rank * 0.20
        + liquidity_rank_high * 0.10
    ).where(conservative_trend_gate)

    defensive_trend_gate = mature_liquid_gate & (data["close"] > ma60) & (momentum_20 > 0)
    data["raw_defensive_core_score"] = (
        volatility_rank * 0.35
        + size_rank * 0.25
        + liquidity_rank_high * 0.15
        + momentum_20_rank * 0.15
        + momentum_60_rank * 0.10
    ).where(defensive_trend_gate)

    pe = _numeric_column(data, "peTTM")
    pb = _numeric_column(data, "pbMRQ")
    ps = _numeric_column(data, "psTTM")
    pcf = _numeric_column(data, "pcfNcfTTM")
    value_pe_rank = pe.where(pe > 0).groupby(data["date"]).rank(pct=True, ascending=False)
    value_pb_rank = pb.where(pb > 0).groupby(data["date"]).rank(pct=True, ascending=False)
    value_ps_rank = ps.where(ps > 0).groupby(data["date"]).rank(pct=True, ascending=False)
    value_pcf_rank = pcf.where(pcf > 0).groupby(data["date"]).rank(pct=True, ascending=False)
    value_score = (
        value_pe_rank * 0.30
        + value_pb_rank * 0.30
        + value_ps_rank * 0.20
        + value_pcf_rank * 0.20
    )
    data["raw_valuation_quality_score"] = (
        value_score * 0.45
        + volatility_rank * 0.20
        + size_rank * 0.20
        + liquidity_rank_high * 0.15
    ).where(mature_liquid_gate & (pb > 0) & (pe > 0))

    roe = _numeric_column(data, "roeAvg")
    net_margin = _numeric_column(data, "npMargin")
    cfo_to_np = _numeric_column(data, "CFOToNP")
    liability_to_asset = _numeric_column(data, "liabilityToAsset")
    roe_rank = roe.groupby(data["date"]).rank(pct=True)
    margin_rank = net_margin.groupby(data["date"]).rank(pct=True)
    cash_conversion_rank = cfo_to_np.clip(lower=-2.0, upper=3.0).groupby(data["date"]).rank(pct=True)
    leverage_rank = liability_to_asset.groupby(data["date"]).rank(pct=True, ascending=True)
    data["raw_fundamental_quality_score"] = (
        roe_rank * 0.35
        + margin_rank * 0.20
        + cash_conversion_rank * 0.20
        + leverage_rank * 0.15
        + value_pb_rank * 0.10
    ).where(mature_liquid_gate & roe.notna())

    yoy_net_income = _numeric_column(data, "YOYNI")
    yoy_asset = _numeric_column(data, "YOYAsset")
    yoy_profit_rank = yoy_net_income.clip(lower=-1.0, upper=2.0).groupby(data["date"]).rank(pct=True)
    yoy_asset_rank = yoy_asset.clip(lower=-1.0, upper=1.5).groupby(data["date"]).rank(pct=True)
    data["raw_growth_value_score"] = (
        yoy_profit_rank * 0.35
        + yoy_asset_rank * 0.15
        + roe_rank * 0.20
        + value_score * 0.20
        + momentum_60_rank * 0.10
    ).where(mature_liquid_gate & yoy_net_income.notna() & (pe > 0))

    data["raw_value_trend_score"] = (
        value_score * 0.40
        + momentum_60_rank * 0.25
        + momentum_120_rank * 0.15
        + volatility_rank * 0.10
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & (pe > 0) & (pb > 0) & (data["close"] > ma60) & (momentum_60 > 0))

    data["raw_large_lowvol_value_score"] = (
        size_rank * 0.35
        + low_volatility_rank * 0.25
        + value_score * 0.25
        + liquidity_rank_high * 0.15
    ).where(mature_liquid_gate & (size_rank >= 0.65) & (pe > 0) & (pb > 0) & volatility_60.notna())

    amount_mean_20 = _rolling_mean(grouped["amount"], 20)
    amount_mean_60 = _rolling_mean(grouped["amount"], 60)
    turnover_mean_20 = _rolling_mean(grouped["turn"], 20)
    turnover_mean_60 = _rolling_mean(grouped["turn"], 60)
    amount_acceleration = (amount_mean_20 / amount_mean_60.replace(0, pd.NA)).groupby(data["date"]).rank(pct=True)
    turnover_acceleration = (turnover_mean_20 / turnover_mean_60.replace(0, pd.NA)).groupby(data["date"]).rank(pct=True)
    data["raw_turnover_accumulation_score"] = (
        amount_acceleration * 0.30
        + turnover_acceleration * 0.25
        + momentum_20_rank * 0.20
        + value_score * 0.15
        + volatility_rank * 0.10
    ).where(mature_liquid_gate & (amount_mean_60 > 0) & (turnover_mean_60 > 0) & (momentum_20 > 0))

    low_turnover_rank = turnover_mean_20.groupby(data["date"]).rank(pct=True, ascending=False)
    turnover_contraction_rank = (turnover_mean_20 / turnover_mean_60.replace(0, pd.NA)).groupby(data["date"]).rank(
        pct=True,
        ascending=False,
    )
    data["raw_low_turnover_trend_score"] = (
        low_turnover_rank * 0.25
        + turnover_contraction_rank * 0.15
        + momentum_60_rank * 0.25
        + momentum_20_rank * 0.15
        + low_volatility_rank * 0.10
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & (turnover_mean_60 > 0) & (data["close"] > ma60) & (momentum_60 > 0))

    drawdown_120 = data["close"] / grouped["close"].rolling(120, min_periods=120).max().reset_index(level=0, drop=True) - 1.0
    moderate_reversal_rank = drawdown_120.clip(lower=-0.45, upper=-0.05).abs().groupby(data["date"]).rank(pct=True)
    rebound_rank = grouped["close"].pct_change(5).groupby(data["date"]).rank(pct=True)
    data["raw_value_reversal_score"] = (
        value_score * 0.35
        + moderate_reversal_rank * 0.25
        + rebound_rank * 0.15
        + volatility_rank * 0.15
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & (pe > 0) & (pb > 0) & (drawdown_120 < -0.05) & (momentum_20 > 0))

    data["raw_large_value_recovery_score"] = (
        value_score * 0.30
        + size_rank * 0.25
        + moderate_reversal_rank * 0.20
        + rebound_rank * 0.15
        + low_volatility_rank * 0.10
    ).where(mature_liquid_gate & (size_rank >= 0.65) & (pe > 0) & (pb > 0) & (drawdown_120 < -0.05) & (momentum_20 > 0))

    small_mid_size_preference = (1.0 - ((size_rank - 0.35).abs() / 0.35)).clip(lower=0.0, upper=1.0)
    data["raw_small_mid_momentum_score"] = (
        small_mid_size_preference * 0.35
        + momentum_60_rank * 0.30
        + momentum_20_rank * 0.15
        + volatility_rank * 0.10
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & (size_rank >= 0.15) & (size_rank <= 0.70) & (momentum_60 > 0) & (data["close"] > ma60))

    close_location = (data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, pd.NA)
    close_location_rank = close_location.clip(lower=0.0, upper=1.0).groupby(data["date"]).rank(pct=True)
    pct_chg = pd.to_numeric(data["pctChg"], errors="coerce")
    strong_day_rank = pct_chg.clip(lower=0.0, upper=10.0).groupby(data["date"]).rank(pct=True)
    volume_expansion_rank = volume_expansion.groupby(data["date"]).rank(pct=True)
    open_price = _numeric_column(data, "open").astype("float64")
    previous_close = grouped["close"].shift(1)
    gap_down = open_price / previous_close.replace(0, pd.NA) - 1.0
    intraday_recovery = data["close"] / open_price.replace(0, pd.NA) - 1.0
    gap_down_rank = gap_down.where(gap_down < 0).abs().groupby(data["date"]).rank(pct=True)
    intraday_recovery_rank = intraday_recovery.groupby(data["date"]).rank(pct=True)
    candle_range = (data["high"] - data["low"]).replace(0, pd.NA)
    lower_shadow = (pd.concat([open_price, data["close"]], axis=1).min(axis=1) - data["low"]) / candle_range
    lower_shadow_rank = lower_shadow.clip(lower=0.0, upper=1.0).groupby(data["date"]).rank(pct=True)
    high_120 = grouped["high"].rolling(120, min_periods=120).max().reset_index(level=0, drop=True)
    near_high = data["close"] / high_120.replace(0, pd.NA)
    near_high_rank = near_high.groupby(data["date"]).rank(pct=True)
    data["raw_limit_followthrough_score"] = (
        strong_day_rank * 0.35
        + close_location_rank * 0.25
        + volume_expansion_rank * 0.20
        + momentum_20_rank * 0.10
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & (pct_chg >= 5.0) & (pct_chg <= 9.7) & (close_location >= 0.70))

    data["raw_gap_reversal_score"] = (
        gap_down_rank * 0.30
        + intraday_recovery_rank * 0.25
        + close_location_rank * 0.15
        + rebound_rank * 0.10
        + value_score * 0.10
        + liquidity_rank_high * 0.10
    ).where(
        mature_liquid_gate
        & (gap_down <= -0.02)
        & (gap_down >= -0.09)
        & (intraday_recovery > 0)
        & (close_location >= 0.60)
        & (pct_chg < 9.7)
    )

    data["raw_lower_shadow_reversal_score"] = (
        lower_shadow_rank * 0.30
        + close_location_rank * 0.20
        + moderate_reversal_rank * 0.20
        + rebound_rank * 0.15
        + value_score * 0.10
        + liquidity_rank_high * 0.05
    ).where(
        mature_liquid_gate
        & (drawdown_120 < -0.05)
        & (lower_shadow >= 0.35)
        & (close_location >= 0.55)
        & (pct_chg > -5.0)
    )

    data["raw_quiet_high_base_score"] = (
        near_high_rank * 0.25
        + compression_rank * 0.25
        + range_compression * 0.20
        + momentum_60_rank * 0.15
        + low_volatility_rank * 0.10
        + liquidity_rank_high * 0.05
    ).where(
        mature_liquid_gate
        & (data["close"] > ma60)
        & (momentum_60 > 0)
        & (near_high >= 0.88)
        & (volatility_ratio < 1.10)
    )

    small_elasticity_rank = (1.0 - size_rank).clip(lower=0.0, upper=1.0)
    breakout_rank = breakout_strength.groupby(data["date"]).rank(pct=True)
    data["raw_high_beta_breakout_score"] = (
        breakout_rank * 0.30
        + momentum_20_rank * 0.20
        + momentum_60_rank * 0.20
        + volume_expansion_rank * 0.15
        + volatility_rank * 0.10
        + small_elasticity_rank * 0.05
    ).where(
        mature_liquid_gate
        & (data["close"] > previous_high)
        & (momentum_20 > 0)
        & (momentum_60 > 0)
        & (pct_chg < 9.7)
    )

    data["raw_volume_price_acceleration_score"] = (
        amount_acceleration * 0.25
        + turnover_acceleration * 0.20
        + strong_day_rank * 0.20
        + close_location_rank * 0.15
        + momentum_20_rank * 0.10
        + volatility_rank * 0.10
    ).where(
        mature_liquid_gate
        & (amount_mean_60 > 0)
        & (turnover_mean_60 > 0)
        & (momentum_20 > 0)
        & (pct_chg >= 2.0)
        & (pct_chg < 9.7)
        & (volume_expansion >= 1.30)
        & (close_location >= 0.65)
    )

    data["raw_drawdown_reacceleration_score"] = (
        moderate_reversal_rank * 0.25
        + rebound_rank * 0.20
        + amount_acceleration * 0.20
        + momentum_20_rank * 0.15
        + small_elasticity_rank * 0.10
        + volatility_rank * 0.10
    ).where(
        mature_liquid_gate
        & (drawdown_120 < -0.05)
        & (drawdown_120 > -0.45)
        & (momentum_20 > 0)
        & (amount_mean_60 > 0)
        & ((amount_mean_20 / amount_mean_60.replace(0, pd.NA)) >= 1.10)
    )

    data["raw_smallcap_rs_acceleration_score"] = (
        small_elasticity_rank * 0.25
        + momentum_60_rank * 0.25
        + momentum_20_rank * 0.20
        + amount_acceleration * 0.15
        + volatility_rank * 0.15
    ).where(
        mature_liquid_gate
        & (size_rank <= 0.55)
        & (data["close"] > ma60)
        & (momentum_20 > 0)
        & (momentum_60 > 0)
        & (amount_mean_60 > 0)
    )

    quiet_volume_rank = volume_expansion.groupby(data["date"]).rank(pct=True, ascending=False)
    quiet_turnover_rank = turnover_acceleration.groupby(data["date"]).rank(pct=True, ascending=False)
    muted_day_rank = strong_day_rank.groupby(data["date"]).rank(pct=True, ascending=False)
    data["raw_quiet_value_trend_score"] = (
        value_score * 0.35
        + momentum_60_rank * 0.15
        + momentum_120_rank * 0.10
        + low_volatility_rank * 0.15
        + quiet_volume_rank * 0.10
        + quiet_turnover_rank * 0.10
        + liquidity_rank_high * 0.05
    ).where(
        mature_liquid_gate
        & (pe > 0)
        & (pb > 0)
        & (data["close"] > ma60)
        & (momentum_60 > 0)
        & (pct_chg >= -3.0)
        & (pct_chg <= 3.0)
        & (volume_expansion <= 1.30)
    )

    data["raw_anti_chase_reversal_score"] = (
        moderate_reversal_rank * 0.25
        + lower_shadow_rank * 0.20
        + close_location_rank * 0.15
        + value_score * 0.15
        + low_volatility_rank * 0.10
        + muted_day_rank * 0.10
        + liquidity_rank_high * 0.05
    ).where(
        mature_liquid_gate
        & (pe > 0)
        & (pb > 0)
        & (drawdown_120 < -0.04)
        & (lower_shadow >= 0.25)
        & (close_location >= 0.50)
        & (pct_chg <= 3.0)
        & (pct_chg > -6.0)
    )

    component_ranks = {
        column: data[column].groupby(data["date"]).rank(pct=True).fillna(0.0)
        for column in [
            "raw_valuation_quality_score",
            "raw_large_lowvol_value_score",
            "raw_value_trend_score",
            "raw_lower_shadow_reversal_score",
            "raw_gap_reversal_score",
            "raw_limit_followthrough_score",
            "raw_low_turnover_trend_score",
            "raw_quiet_value_trend_score",
            "raw_anti_chase_reversal_score",
        ]
    }
    data["raw_value_event_composite_score"] = (
        component_ranks["raw_valuation_quality_score"] * 0.35
        + component_ranks["raw_value_trend_score"] * 0.25
        + component_ranks["raw_lower_shadow_reversal_score"] * 0.15
        + component_ranks["raw_gap_reversal_score"] * 0.10
        + component_ranks["raw_limit_followthrough_score"] * 0.05
        + component_ranks["raw_low_turnover_trend_score"] * 0.10
    ).where(mature_liquid_gate & (pe > 0) & (pb > 0))

    data["raw_defensive_event_composite_score"] = (
        component_ranks["raw_valuation_quality_score"] * 0.30
        + component_ranks["raw_large_lowvol_value_score"] * 0.25
        + component_ranks["raw_gap_reversal_score"] * 0.15
        + component_ranks["raw_lower_shadow_reversal_score"] * 0.10
        + component_ranks["raw_low_turnover_trend_score"] * 0.10
        + low_volatility_rank.fillna(0.0) * 0.10
    ).where(mature_liquid_gate & (size_rank >= 0.50) & (pe > 0) & (pb > 0))

    reinforcement_columns = [
        "raw_valuation_quality_score",
        "raw_value_trend_score",
        "raw_large_lowvol_value_score",
        "raw_lower_shadow_reversal_score",
        "raw_gap_reversal_score",
        "raw_limit_followthrough_score",
        "raw_low_turnover_trend_score",
        "raw_quiet_value_trend_score",
        "raw_anti_chase_reversal_score",
    ]
    consensus_sum = sum(component_ranks[column] for column in reinforcement_columns)
    consensus_count = sum((component_ranks[column] >= 0.70).astype(float) for column in reinforcement_columns)
    consensus_average = consensus_sum / len(reinforcement_columns)
    data["raw_factor_consensus_score"] = (
        consensus_average * 0.70 + (consensus_count / len(reinforcement_columns)) * 0.30
    ).where(mature_liquid_gate & (pe > 0) & (pb > 0))

    universe_return = daily_return.groupby(data["date"]).mean()
    weighted_sum = pd.Series(0.0, index=data.index)
    weight_sum = pd.Series(0.0, index=data.index)
    for column in reinforcement_columns:
        rank = component_ranks[column]
        prior_rank = rank.groupby(data["code"]).shift(1)
        top_return = daily_return.where(prior_rank >= 0.80).groupby(data["date"]).mean()
        factor_payoff = (top_return - universe_return).fillna(0.0)
        factor_strength = factor_payoff.rolling(60, min_periods=20).mean().clip(lower=0.0)
        row_weight = pd.Series(factor_strength.reindex(data["date"]).to_numpy(), index=data.index).fillna(0.0)
        weighted_sum = weighted_sum + rank * row_weight
        weight_sum = weight_sum + row_weight
    reinforced_core = weighted_sum / weight_sum.where(weight_sum > 0)
    data["raw_factor_reinforced_score"] = reinforced_core.where(weight_sum > 0, data["raw_factor_consensus_score"]).where(
        mature_liquid_gate & (pe > 0) & (pb > 0)
    )

    has_industry = "industry" in data
    industry_group = (
        data["industry"].replace("", pd.NA).fillna("unknown").astype(str)
        if has_industry
        else pd.Series("unknown", index=data.index)
    )
    industry_frame = pd.DataFrame(
        {
            "date": data["date"],
            "industry": industry_group,
            "code": data["code"],
            "momentum_60": momentum_60,
            "momentum_20_positive": momentum_20 > 0,
            "amount_acceleration": amount_mean_20 / amount_mean_60.replace(0, pd.NA),
        }
    )
    industry_daily = industry_frame.groupby(["date", "industry"], dropna=False).agg(
        industry_momentum_60=("momentum_60", "mean"),
        industry_breadth_20=("momentum_20_positive", "mean"),
        industry_amount_acceleration=("amount_acceleration", "mean"),
        industry_member_count=("code", "count"),
    )
    industry_daily["industry_momentum_rank"] = industry_daily.groupby(level=0)["industry_momentum_60"].rank(pct=True)
    industry_daily["industry_breadth_rank"] = industry_daily.groupby(level=0)["industry_breadth_20"].rank(pct=True)
    industry_daily["industry_amount_rank"] = industry_daily.groupby(level=0)["industry_amount_acceleration"].rank(pct=True)
    industry_metrics = industry_daily.reset_index()
    industry_features = (
        pd.DataFrame({"row_index": data.index, "date": data["date"], "industry": industry_group})
        .merge(industry_metrics, on=["date", "industry"], how="left")
        .set_index("row_index")
        .reindex(data.index)
    )
    industry_momentum_rank = industry_features["industry_momentum_rank"]
    industry_breadth_rank = industry_features["industry_breadth_rank"]
    industry_amount_rank = industry_features["industry_amount_rank"]
    industry_momentum = industry_features["industry_momentum_60"]
    industry_member_count = industry_features["industry_member_count"].fillna(0)
    industry_gate = pd.Series(bool(has_industry), index=data.index) & (industry_member_count >= 3) & (industry_momentum > 0)
    data["raw_industry_rotation_score"] = (
        industry_momentum_rank * 0.40
        + industry_breadth_rank * 0.25
        + industry_amount_rank * 0.15
        + momentum_60_rank * 0.10
        + liquidity_rank_high * 0.10
    ).where(mature_liquid_gate & industry_gate)

    within_industry_momentum_rank = momentum_60.groupby([data["date"], industry_group]).rank(pct=True)
    within_industry_liquidity_rank = data["amount"].groupby([data["date"], industry_group]).rank(pct=True)
    data["raw_industry_leader_score"] = (
        industry_momentum_rank * 0.30
        + industry_breadth_rank * 0.15
        + within_industry_momentum_rank * 0.25
        + within_industry_liquidity_rank * 0.15
        + volatility_rank * 0.15
    ).where(mature_liquid_gate & industry_gate & (momentum_60 > 0) & (data["close"] > ma60))

    weak_industry_rank = (1.0 - industry_momentum_rank).clip(lower=0.0, upper=1.0)
    industry_reversal_gate = (
        pd.Series(bool(has_industry), index=data.index)
        & (industry_member_count >= 3)
        & ((industry_momentum < 0) | (industry_momentum_rank <= 0.45))
        & (momentum_20 > 0)
    )
    data["raw_industry_reversal_score"] = (
        weak_industry_rank * 0.30
        + rebound_rank * 0.20
        + momentum_20_rank * 0.15
        + value_score * 0.20
        + low_volatility_rank * 0.10
        + liquidity_rank_high * 0.05
    ).where(mature_liquid_gate & industry_reversal_gate & (pe > 0) & (pb > 0))

    for raw_column, score_column in [
        ("raw_momentum_lowvol_score", "momentum_lowvol_score"),
        ("raw_trend_strength_score", "trend_strength_score"),
        ("raw_breakout_score", "breakout_score"),
        ("raw_reversal_quality_score", "reversal_quality_score"),
        ("raw_absolute_trend_score", "absolute_trend_score"),
        ("raw_relative_strength_quality_score", "relative_strength_quality_score"),
        ("raw_risk_adjusted_trend_score", "risk_adjusted_trend_score"),
        ("raw_steady_uptrend_score", "steady_uptrend_score"),
        ("raw_trend_pullback_score", "trend_pullback_score"),
        ("raw_squeeze_breakout_score", "squeeze_breakout_score"),
        ("raw_balanced_core_score", "balanced_core_score"),
        ("raw_defensive_core_score", "defensive_core_score"),
        ("raw_valuation_quality_score", "valuation_quality_score"),
        ("raw_fundamental_quality_score", "fundamental_quality_score"),
        ("raw_growth_value_score", "growth_value_score"),
        ("raw_value_trend_score", "value_trend_score"),
        ("raw_large_lowvol_value_score", "large_lowvol_value_score"),
        ("raw_turnover_accumulation_score", "turnover_accumulation_score"),
        ("raw_low_turnover_trend_score", "low_turnover_trend_score"),
        ("raw_volatility_contraction_trend_score", "volatility_contraction_trend_score"),
        ("raw_value_reversal_score", "value_reversal_score"),
        ("raw_large_value_recovery_score", "large_value_recovery_score"),
        ("raw_small_mid_momentum_score", "small_mid_momentum_score"),
        ("raw_limit_followthrough_score", "limit_followthrough_score"),
        ("raw_gap_reversal_score", "gap_reversal_score"),
        ("raw_lower_shadow_reversal_score", "lower_shadow_reversal_score"),
        ("raw_quiet_high_base_score", "quiet_high_base_score"),
        ("raw_value_event_composite_score", "value_event_composite_score"),
        ("raw_defensive_event_composite_score", "defensive_event_composite_score"),
        ("raw_high_beta_breakout_score", "high_beta_breakout_score"),
        ("raw_volume_price_acceleration_score", "volume_price_acceleration_score"),
        ("raw_drawdown_reacceleration_score", "drawdown_reacceleration_score"),
        ("raw_smallcap_rs_acceleration_score", "smallcap_rs_acceleration_score"),
        ("raw_quiet_value_trend_score", "quiet_value_trend_score"),
        ("raw_anti_chase_reversal_score", "anti_chase_reversal_score"),
        ("raw_factor_reinforced_score", "factor_reinforced_score"),
        ("raw_factor_consensus_score", "factor_consensus_score"),
        ("raw_industry_rotation_score", "industry_rotation_score"),
        ("raw_industry_leader_score", "industry_leader_score"),
        ("raw_industry_reversal_score", "industry_reversal_score"),
    ]:
        data[score_column] = grouped[raw_column].shift(1)

    return data.sort_values(["date", "code"]).reset_index(drop=True)
