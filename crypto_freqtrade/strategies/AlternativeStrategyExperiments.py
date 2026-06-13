from functools import reduce
from pathlib import Path

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame


class AlternativeFuturesBase(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    informative_timeframe = "1d"
    startup_candle_count = 720
    can_short = True

    minimal_roi = {
        "0": 0.42,
        "96": 0.18,
        "360": 0.06,
        "900": 0,
    }

    stoploss = -0.14
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.12
    trailing_only_offset_is_reached = True

    btc_pair = "BTC/USDT:USDT"
    leverage_value = 2.0
    target_trade_volatility = 0.08
    min_stake_fraction = 0.20
    max_stake_fraction = 0.85

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 12,
                "required_profit": 0.0,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 90,
                "trade_limit": 18,
                "stop_duration_candles": 18,
                "max_allowed_drawdown": 0.24,
                "calculation_mode": "equity",
            },
        ]

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return max(1.0, min(float(self.leverage_value), float(max_leverage)))

    def informative_pairs(self):
        pairs = {(self.btc_pair, self.informative_timeframe)}
        if self.dp:
            pairs.update((pair, self.informative_timeframe) for pair in self.dp.current_whitelist())
        return sorted(pairs)

    @staticmethod
    def _daily_context(dataframe: DataFrame, prefix: str) -> DataFrame:
        dataframe[f"{prefix}_ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe[f"{prefix}_ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe[f"{prefix}_ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe[f"{prefix}_rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe[f"{prefix}_momentum_30"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0
        dataframe[f"{prefix}_momentum_90"] = dataframe["close"] / dataframe["close"].shift(90) - 1.0
        dataframe[f"{prefix}_volatility_30"] = dataframe["close"].pct_change().rolling(30, min_periods=30).std()
        dataframe[f"{prefix}_range_20"] = (
            dataframe["high"].rolling(20, min_periods=20).max()
            / dataframe["low"].rolling(20, min_periods=20).min()
            - 1.0
        )
        return dataframe

    @staticmethod
    def _long_regime(dataframe: DataFrame, prefix: str) -> pd.Series:
        return (
            (dataframe["close"] > dataframe[f"{prefix}_ema_100"])
            & (dataframe[f"{prefix}_ema_50"] > dataframe[f"{prefix}_ema_200"])
            & (dataframe[f"{prefix}_momentum_30"] > -0.02)
            & (dataframe[f"{prefix}_volatility_30"] < 0.095)
        )

    @staticmethod
    def _short_regime(dataframe: DataFrame, prefix: str) -> pd.Series:
        return (
            (dataframe["close"] < dataframe[f"{prefix}_ema_100"])
            & (dataframe[f"{prefix}_ema_50"] < dataframe[f"{prefix}_ema_200"])
            & (dataframe[f"{prefix}_momentum_30"] < 0.02)
            & (dataframe[f"{prefix}_volatility_30"] < 0.11)
        )

    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["btc_long_regime_1d"] = 1
            dataframe["btc_short_regime_1d"] = 1
            dataframe["pair_long_regime_1d"] = 1
            dataframe["pair_short_regime_1d"] = 1
            dataframe["pair_daily_rsi_1d"] = 50
            return dataframe

        btc_daily = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=self.informative_timeframe)
        btc_daily = self._daily_context(btc_daily, "btc")
        btc_daily["btc_long_regime"] = self._long_regime(btc_daily, "btc").astype(int)
        btc_daily["btc_short_regime"] = self._short_regime(btc_daily, "btc").astype(int)
        dataframe = merge_informative_pair(
            dataframe,
            btc_daily[["date", "btc_long_regime", "btc_short_regime"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        pair_daily = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        pair_daily = self._daily_context(pair_daily, "pair")
        pair_daily["pair_long_regime"] = self._long_regime(pair_daily, "pair").astype(int)
        pair_daily["pair_short_regime"] = self._short_regime(pair_daily, "pair").astype(int)
        pair_daily["pair_daily_rsi"] = pair_daily["pair_rsi"]
        dataframe = merge_informative_pair(
            dataframe,
            pair_daily[["date", "pair_long_regime", "pair_short_regime", "pair_daily_rsi"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if not self.dp:
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or "atr_pct" not in dataframe:
            return proposed_stake

        candle_time = pd.Timestamp(current_time)
        if candle_time.tzinfo is not None:
            candle_time = candle_time.tz_convert("UTC").tz_localize(None)

        candle_dates = pd.to_datetime(dataframe["date"])
        if getattr(candle_dates.dt, "tz", None) is not None:
            candle_dates = candle_dates.dt.tz_convert("UTC").dt.tz_localize(None)
        candles = dataframe.loc[candle_dates <= candle_time]
        if candles.empty:
            return proposed_stake

        atr_pct = candles.iloc[-1]["atr_pct"]
        if pd.isna(atr_pct) or atr_pct <= 0:
            return proposed_stake

        fraction = self.target_trade_volatility / (float(atr_pct) * max(1.0, leverage))
        fraction = min(self.max_stake_fraction, max(self.min_stake_fraction, fraction))
        stake = proposed_stake * fraction
        if min_stake:
            stake = max(min_stake, stake)
        return min(max_stake, stake)


class FuturesVolatilityExpansionTrendStrategy(AlternativeFuturesBase):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["volume_mean_48"] = dataframe["volume"].rolling(48, min_periods=48).mean()
        dataframe["range_high"] = dataframe["high"].rolling(30, min_periods=30).max().shift(1)
        dataframe["range_low"] = dataframe["low"].rolling(30, min_periods=30).min().shift(1)
        dataframe["narrow_range"] = (
            dataframe["high"].rolling(18, min_periods=18).max()
            / dataframe["low"].rolling(18, min_periods=18).min()
            - 1.0
        )
        dataframe["narrow_range_mean"] = dataframe["narrow_range"].rolling(90, min_periods=90).mean()
        dataframe["momentum_42"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(12) - 1.0
        dataframe["exit_long_low"] = dataframe["low"].rolling(10, min_periods=10).min().shift(1)
        dataframe["exit_short_high"] = dataframe["high"].rolling(10, min_periods=10).max().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        compression = dataframe["narrow_range"] < dataframe["narrow_range_mean"] * 0.78
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["volume"] > dataframe["volume_mean_48"] * 1.05,
            compression,
            dataframe["close"] > dataframe["range_high"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0,
            dataframe["momentum_42"] > 0.035,
            dataframe["adx"] > 16,
            dataframe["rsi"].between(52, 80),
            dataframe["atr_pct"].between(0.008, 0.13),
            dataframe["btc_long_regime_1d"] == 1,
            dataframe["pair_long_regime_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["volume"] > dataframe["volume_mean_48"] * 1.05,
            compression,
            dataframe["close"] < dataframe["range_low"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_200"],
            dataframe["ema_100_slope"] < 0,
            dataframe["momentum_42"] < -0.035,
            dataframe["adx"] > 16,
            dataframe["rsi"].between(20, 48),
            dataframe["atr_pct"].between(0.008, 0.13),
            dataframe["btc_short_regime_1d"] == 1,
            dataframe["pair_short_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "vol_expansion_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "vol_expansion_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            (dataframe["close"] < dataframe["exit_long_low"])
            | (dataframe["ema_20"] < dataframe["ema_50"])
            | (dataframe["rsi"] < 43)
        )
        short_exit = (
            (dataframe["close"] > dataframe["exit_short_high"])
            | (dataframe["ema_20"] > dataframe["ema_50"])
            | (dataframe["rsi"] > 57)
        )
        dataframe.loc[(dataframe["volume"] > 0) & long_exit, ["exit_long", "exit_tag"]] = (1, "vol_long_exit")
        dataframe.loc[(dataframe["volume"] > 0) & short_exit, ["exit_short", "exit_tag"]] = (1, "vol_short_exit")
        return dataframe


class FuturesVolatilityExpansionTrend3xStrategy(FuturesVolatilityExpansionTrendStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.10
    stoploss = -0.18


class FuturesCrashReversalMeanReversionStrategy(AlternativeFuturesBase):
    minimal_roi = {
        "0": 0.20,
        "36": 0.09,
        "120": 0.03,
        "288": 0,
    }
    stoploss = -0.08
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.075
    target_trade_volatility = 0.055

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["fast_rsi"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24, min_periods=24).mean()
        dataframe["ret_3"] = dataframe["close"] / dataframe["close"].shift(3) - 1.0
        dataframe["ret_6"] = dataframe["close"] / dataframe["close"].shift(6) - 1.0
        dataframe["bb_upper"], dataframe["bb_mid"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"],
            timeperiod=40,
            nbdevup=2.4,
            nbdevdn=2.4,
            matype=0,
        )
        dataframe["mean_revert_high"] = dataframe["high"].rolling(8, min_periods=8).max().shift(1)
        dataframe["mean_revert_low"] = dataframe["low"].rolling(8, min_periods=8).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volatile_enough = dataframe["atr_pct"].between(0.012, 0.16)
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"] * 1.05,
            volatile_enough,
            dataframe["close"] < dataframe["bb_lower"],
            dataframe["fast_rsi"] < 16,
            dataframe["ret_3"] < -0.055,
            dataframe["ret_6"] < -0.075,
            dataframe["close"] > dataframe["ema_200"] * 0.72,
            dataframe["btc_short_regime_1d"] == 0,
            dataframe["pair_daily_rsi_1d"] > 32,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"] * 1.05,
            volatile_enough,
            dataframe["close"] > dataframe["bb_upper"],
            dataframe["fast_rsi"] > 84,
            dataframe["ret_3"] > 0.055,
            dataframe["ret_6"] > 0.075,
            dataframe["close"] < dataframe["ema_200"] * 1.28,
            dataframe["btc_long_regime_1d"] == 0,
            dataframe["pair_daily_rsi_1d"] < 68,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "crash_reversal_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "blowoff_reversal_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            (dataframe["close"] > dataframe["bb_mid"])
            | (dataframe["close"] > dataframe["mean_revert_high"])
            | (dataframe["fast_rsi"] > 68)
        )
        short_exit = (
            (dataframe["close"] < dataframe["bb_mid"])
            | (dataframe["close"] < dataframe["mean_revert_low"])
            | (dataframe["fast_rsi"] < 32)
        )
        dataframe.loc[(dataframe["volume"] > 0) & long_exit, ["exit_long", "exit_tag"]] = (1, "reversal_long_exit")
        dataframe.loc[(dataframe["volume"] > 0) & short_exit, ["exit_short", "exit_tag"]] = (1, "reversal_short_exit")
        return dataframe


class FuturesCrashReversalMeanReversion3xStrategy(FuturesCrashReversalMeanReversionStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.075
    stoploss = -0.10


class FuturesRangeBreakoutAdaptiveStrategy(FuturesVolatilityExpansionTrendStrategy):
    minimal_roi = {
        "0": 0.34,
        "72": 0.14,
        "240": 0.04,
        "720": 0,
    }
    stoploss = -0.12
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.09

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["range_high"],
            dataframe["close"] > dataframe["ema_100"],
            dataframe["ema_20"] > dataframe["ema_50"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["momentum_42"] > 0.02,
            dataframe["adx"] > 13,
            dataframe["rsi"].between(50, 78),
            dataframe["atr_pct"].between(0.007, 0.12),
            dataframe["btc_long_regime_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["range_low"],
            dataframe["close"] < dataframe["ema_100"],
            dataframe["ema_20"] < dataframe["ema_50"],
            dataframe["ema_50"] < dataframe["ema_200"],
            dataframe["momentum_42"] < -0.02,
            dataframe["adx"] > 13,
            dataframe["rsi"].between(22, 50),
            dataframe["atr_pct"].between(0.007, 0.12),
            dataframe["btc_short_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "adaptive_range_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "adaptive_range_short",
        )
        return dataframe


class FuturesExpandedUniverseTrendStrategy(AlternativeFuturesBase):
    minimal_roi = {
        "0": 0.50,
        "144": 0.22,
        "480": 0.08,
        "1200": 0,
    }
    stoploss = -0.16
    target_trade_volatility = 0.075

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_21"] = dataframe["close"] / dataframe["close"].shift(21) - 1.0
        dataframe["momentum_63"] = dataframe["close"] / dataframe["close"].shift(63) - 1.0
        dataframe["volatility_63"] = dataframe["close"].pct_change().rolling(63, min_periods=63).std()
        dataframe["trend_quality"] = dataframe["momentum_63"] / dataframe["volatility_63"].replace(0, pd.NA)
        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(18) - 1.0
        dataframe["breakout_high"] = dataframe["high"].rolling(42, min_periods=42).max().shift(1)
        dataframe["breakout_low"] = dataframe["low"].rolling(42, min_periods=42).min().shift(1)
        dataframe["exit_low"] = dataframe["low"].rolling(14, min_periods=14).min().shift(1)
        dataframe["exit_high"] = dataframe["high"].rolling(14, min_periods=14).max().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["breakout_high"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_20"] > dataframe["ema_50"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0.015,
            dataframe["momentum_21"] > 0.035,
            dataframe["trend_quality"] > 1.0,
            dataframe["adx"] > 18,
            dataframe["rsi"].between(54, 78),
            dataframe["atr_pct"].between(0.010, 0.15),
            dataframe["btc_long_regime_1d"] == 1,
            dataframe["pair_long_regime_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["breakout_low"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_20"] < dataframe["ema_50"],
            dataframe["ema_50"] < dataframe["ema_100"],
            dataframe["ema_100"] < dataframe["ema_200"],
            dataframe["ema_100_slope"] < -0.015,
            dataframe["momentum_21"] < -0.035,
            dataframe["trend_quality"] < -1.0,
            dataframe["adx"] > 18,
            dataframe["rsi"].between(22, 46),
            dataframe["atr_pct"].between(0.010, 0.15),
            dataframe["btc_short_regime_1d"] == 1,
            dataframe["pair_short_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "expanded_trend_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "expanded_trend_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            (dataframe["close"] < dataframe["exit_low"])
            | (dataframe["ema_20"] < dataframe["ema_50"])
            | (dataframe["trend_quality"] < 0)
            | (dataframe["rsi"] < 42)
        )
        short_exit = (
            (dataframe["close"] > dataframe["exit_high"])
            | (dataframe["ema_20"] > dataframe["ema_50"])
            | (dataframe["trend_quality"] > 0)
            | (dataframe["rsi"] > 58)
        )
        dataframe.loc[(dataframe["volume"] > 0) & long_exit, ["exit_long", "exit_tag"]] = (
            1,
            "expanded_trend_long_exit",
        )
        dataframe.loc[(dataframe["volume"] > 0) & short_exit, ["exit_short", "exit_tag"]] = (
            1,
            "expanded_trend_short_exit",
        )
        return dataframe


class FuturesExpandedUniverseTrend3xStrategy(FuturesExpandedUniverseTrendStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.095
    stoploss = -0.20


class FuturesCorePairsDailyTrendStrategy(FuturesExpandedUniverseTrendStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.085
    max_stake_fraction = 0.75
    allowed_pairs = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata["pair"] not in self.allowed_pairs:
            return dataframe
        return super().populate_entry_trend(dataframe, metadata)


class FuturesRiskOffShortOnlyStrategy(FuturesExpandedUniverseTrendStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.075
    stoploss = -0.13
    minimal_roi = {
        "0": 0.32,
        "72": 0.14,
        "240": 0.045,
        "720": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["breakout_low"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_20"] < dataframe["ema_50"],
            dataframe["ema_50"] < dataframe["ema_100"],
            dataframe["ema_100_slope"] < -0.008,
            dataframe["momentum_21"] < -0.025,
            dataframe["trend_quality"] < -0.65,
            dataframe["adx"] > 14,
            dataframe["rsi"].between(18, 48),
            dataframe["atr_pct"].between(0.010, 0.18),
            dataframe["btc_short_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "risk_off_short_only",
        )
        return dataframe


class FuturesFundingBase(AlternativeFuturesBase):
    timeframe = "1h"
    startup_candle_count = 900
    leverage_value = 2.0
    target_trade_volatility = 0.045
    max_stake_fraction = 0.60
    stoploss = -0.065
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.045

    minimal_roi = {
        "0": 0.10,
        "16": 0.045,
        "48": 0.016,
        "144": 0,
    }

    def _merge_carry_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["funding_rate_1h"] = 0.0
            dataframe["funding_mean_72_1h"] = 0.0
            dataframe["funding_rank_168_1h"] = 0.5
            dataframe["basis_pct_1h"] = 0.0
            dataframe["basis_rank_168_1h"] = 0.5
            return dataframe

        try:
            funding = self.dp.get_pair_dataframe(
                pair=metadata["pair"],
                timeframe=self.timeframe,
                candle_type="funding_rate",
            )
            mark = self.dp.get_pair_dataframe(
                pair=metadata["pair"],
                timeframe=self.timeframe,
                candle_type="mark",
            )
        except TypeError:
            funding = pd.DataFrame(columns=["date", "open"])
            mark = pd.DataFrame(columns=["date", "close"])

        carry = dataframe[["date", "close"]].copy()
        carry.rename(columns={"close": "futures_close"}, inplace=True)
        if not funding.empty:
            funding = funding[["date", "open"]].copy()
            funding.rename(columns={"open": "funding_rate"}, inplace=True)
            funding["funding_rate"] = funding["funding_rate"].shift(1)
            funding["funding_mean_72"] = funding["funding_rate"].rolling(72, min_periods=24).mean()
            funding["funding_rank_168"] = funding["funding_rate"].rolling(168, min_periods=48).rank(pct=True)
            carry = carry.merge(
                funding[["date", "funding_rate", "funding_mean_72", "funding_rank_168"]],
                on="date",
                how="left",
            )
        else:
            carry["funding_rate"] = 0.0
            carry["funding_mean_72"] = 0.0
            carry["funding_rank_168"] = 0.5

        if not mark.empty:
            mark = mark[["date", "close"]].copy()
            mark.rename(columns={"close": "mark_close"}, inplace=True)
            carry = carry.merge(mark, on="date", how="left")
            carry["basis_pct"] = ((carry["futures_close"] - carry["mark_close"]) / carry["mark_close"]).shift(1)
            carry["basis_rank_168"] = carry["basis_pct"].rolling(168, min_periods=48).rank(pct=True)
        else:
            carry["basis_pct"] = 0.0
            carry["basis_rank_168"] = 0.5

        informative = carry[
            ["date", "funding_rate", "funding_mean_72", "funding_rank_168", "basis_pct", "basis_rank_168"]
        ]
        return merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.timeframe,
            ffill=True,
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_24"] = ta.EMA(dataframe, timeperiod=24)
        dataframe["ema_72"] = ta.EMA(dataframe, timeperiod=72)
        dataframe["ema_168"] = ta.EMA(dataframe, timeperiod=168)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["fast_rsi"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24, min_periods=24).mean()
        dataframe["momentum_12"] = dataframe["close"] / dataframe["close"].shift(12) - 1.0
        dataframe["momentum_48"] = dataframe["close"] / dataframe["close"].shift(48) - 1.0
        dataframe["range_high_24"] = dataframe["high"].rolling(24, min_periods=24).max().shift(1)
        dataframe["range_low_24"] = dataframe["low"].rolling(24, min_periods=24).min().shift(1)
        dataframe["bb_upper"], dataframe["bb_mid"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"],
            timeperiod=40,
            nbdevup=2.2,
            nbdevdn=2.2,
        )
        dataframe["z_atr"] = (dataframe["close"] - dataframe["bb_mid"]) / dataframe["atr"].replace(0, pd.NA)
        dataframe["exit_low"] = dataframe["low"].rolling(12, min_periods=12).min().shift(1)
        dataframe["exit_high"] = dataframe["high"].rolling(12, min_periods=12).max().shift(1)
        dataframe = self._merge_carry_context(dataframe, metadata)
        return self._merge_daily_context(dataframe, metadata)


class FuturesExtremeFundingReversalStrategy(FuturesFundingBase):
    leverage_value = 3.0
    target_trade_volatility = 0.060
    max_stake_fraction = 0.65
    stoploss = -0.055
    trailing_stop_positive = 0.014
    trailing_stop_positive_offset = 0.034
    minimal_roi = {
        "0": 0.075,
        "8": 0.034,
        "32": 0.012,
        "96": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        funding_rank = dataframe.get("funding_rank_168_1h", pd.Series(0.5, index=dataframe.index)).fillna(0.5)
        basis_rank = dataframe.get("basis_rank_168_1h", pd.Series(0.5, index=dataframe.index)).fillna(0.5)
        funding = dataframe.get("funding_rate_1h", pd.Series(0.0, index=dataframe.index)).fillna(0.0)
        funding_mean = dataframe.get("funding_mean_72_1h", pd.Series(0.0, index=dataframe.index)).fillna(0.0)

        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] > dataframe["ema_168"] * 0.94,
            dataframe["close"] < dataframe["bb_lower"],
            dataframe["z_atr"] < -1.15,
            dataframe["fast_rsi"] < 24,
            funding_rank <= 0.18,
            basis_rank <= 0.35,
            funding < funding_mean,
            dataframe["atr_pct"].between(0.004, 0.080),
            dataframe["btc_short_regime_1d"] == 0,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] < dataframe["ema_168"] * 1.06,
            dataframe["close"] > dataframe["bb_upper"],
            dataframe["z_atr"] > 1.15,
            dataframe["fast_rsi"] > 76,
            funding_rank >= 0.82,
            basis_rank >= 0.65,
            funding > funding_mean,
            dataframe["atr_pct"].between(0.004, 0.080),
            dataframe["btc_long_regime_1d"] == 0,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "extreme_funding_reversal_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "extreme_funding_reversal_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] > dataframe["bb_mid"]) | (dataframe["fast_rsi"] > 68)),
            ["exit_long", "exit_tag"],
        ] = (1, "extreme_funding_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] < dataframe["bb_mid"]) | (dataframe["fast_rsi"] < 32)),
            ["exit_short", "exit_tag"],
        ] = (1, "extreme_funding_short_exit")
        return dataframe


class FuturesCarryTrendRelaxedStrategy(FuturesFundingBase):
    leverage_value = 3.0
    target_trade_volatility = 0.070
    max_stake_fraction = 0.70
    stoploss = -0.085
    trailing_stop_positive = 0.022
    trailing_stop_positive_offset = 0.055
    minimal_roi = {
        "0": 0.130,
        "24": 0.060,
        "72": 0.024,
        "216": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        funding_rank = dataframe.get("funding_rank_168_1h", pd.Series(0.5, index=dataframe.index)).fillna(0.5)
        basis_rank = dataframe.get("basis_rank_168_1h", pd.Series(0.5, index=dataframe.index)).fillna(0.5)

        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] > dataframe["range_high_24"],
            dataframe["close"] > dataframe["ema_168"],
            dataframe["ema_24"] > dataframe["ema_72"],
            dataframe["momentum_12"] > 0.006,
            dataframe["momentum_48"] > 0.015,
            funding_rank >= 0.52,
            basis_rank >= 0.50,
            dataframe["adx"] > 13,
            dataframe["rsi"].between(50, 80),
            dataframe["atr_pct"].between(0.004, 0.085),
            dataframe["btc_long_regime_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] < dataframe["range_low_24"],
            dataframe["close"] < dataframe["ema_168"],
            dataframe["ema_24"] < dataframe["ema_72"],
            dataframe["momentum_12"] < -0.006,
            dataframe["momentum_48"] < -0.015,
            funding_rank <= 0.48,
            basis_rank <= 0.50,
            dataframe["adx"] > 13,
            dataframe["rsi"].between(20, 50),
            dataframe["atr_pct"].between(0.004, 0.085),
            dataframe["btc_short_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "carry_trend_relaxed_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "carry_trend_relaxed_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] < dataframe["exit_low"])
                | (dataframe["ema_24"] < dataframe["ema_72"])
                | (dataframe["rsi"] < 44)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "carry_trend_relaxed_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] > dataframe["exit_high"])
                | (dataframe["ema_24"] > dataframe["ema_72"])
                | (dataframe["rsi"] > 56)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "carry_trend_relaxed_short_exit")
        return dataframe


class FuturesRelativeStrengthLongRotationStrategy(AlternativeFuturesBase):
    can_short = False
    leverage_value = 2.0
    target_trade_volatility = 0.080
    max_stake_fraction = 0.90
    rank_slots = 3.0
    min_rank_pct = 0.12
    min_rs_score = 0.80
    min_market_breadth = 0.28
    exit_rank_pct = 0.40
    stoploss = -0.13
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.11
    minimal_roi = {
        "0": 0.48,
        "120": 0.22,
        "420": 0.08,
        "960": 0,
    }

    def _rank_percentile(self, dataframe: DataFrame, pair: str, column: str) -> pd.Series:
        if not self.dp:
            return pd.Series(0.5, index=dataframe.index)

        ranks = pd.DataFrame(index=dataframe["date"])
        ranks[pair] = dataframe.set_index("date")[column].shift(1)
        for candidate in self.dp.current_whitelist():
            if candidate == pair:
                continue
            candidate_df = self.dp.get_pair_dataframe(candidate, self.timeframe)
            if candidate_df.empty:
                continue
            momentum_fast = candidate_df["close"] / candidate_df["close"].shift(42) - 1.0
            momentum_slow = candidate_df["close"] / candidate_df["close"].shift(126) - 1.0
            volatility = candidate_df["close"].pct_change().rolling(63, min_periods=63).std()
            score = (momentum_fast * 0.65 + momentum_slow * 0.35) / volatility.replace(0, pd.NA)
            ranks[candidate] = score.shift(1).set_axis(candidate_df["date"]).reindex(ranks.index)
        return ranks.rank(axis=1, ascending=False, pct=True, method="first")[pair].to_numpy()

    def _market_breadth(self, dataframe: DataFrame, pair: str) -> pd.Series:
        if not self.dp:
            return pd.Series(1.0, index=dataframe.index)

        trend_flags = pd.DataFrame(index=dataframe["date"])
        current_pair_flag = (dataframe.set_index("date")["close"] > dataframe.set_index("date")["ema_100"]).astype(float)
        trend_flags[pair] = current_pair_flag.shift(1)
        for candidate in self.dp.current_whitelist():
            if candidate == pair:
                continue
            candidate_df = self.dp.get_pair_dataframe(candidate, self.timeframe)
            if candidate_df.empty:
                continue
            ema_100 = ta.EMA(candidate_df, timeperiod=100)
            flag = (candidate_df["close"] > ema_100).astype(float).shift(1)
            trend_flags[candidate] = flag.set_axis(candidate_df["date"]).reindex(trend_flags.index)
        return trend_flags.mean(axis=1, skipna=True).fillna(0.0).to_numpy()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_42"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["momentum_126"] = dataframe["close"] / dataframe["close"].shift(126) - 1.0
        dataframe["volatility_63"] = dataframe["close"].pct_change().rolling(63, min_periods=63).std()
        dataframe["rs_score"] = (
            dataframe["momentum_42"] * 0.65 + dataframe["momentum_126"] * 0.35
        ) / dataframe["volatility_63"].replace(0, pd.NA)
        dataframe["rs_rank_pct"] = self._rank_percentile(dataframe, metadata["pair"], "rs_score")
        dataframe["market_breadth"] = self._market_breadth(dataframe, metadata["pair"])
        dataframe["exit_low"] = dataframe["low"].rolling(12, min_periods=12).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        top_count_pct = max(self.min_rank_pct, self.rank_slots / max(3, len(self.dp.current_whitelist()) if self.dp else 30))
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["rs_rank_pct"] <= top_count_pct,
            dataframe["rs_score"] > self.min_rs_score,
            dataframe["market_breadth"] > self.min_market_breadth,
            dataframe["close"] > dataframe["ema_100"],
            dataframe["ema_20"] > dataframe["ema_50"],
            dataframe["momentum_42"] > 0.04,
            dataframe["momentum_126"] > -0.04,
            dataframe["adx"] > 13,
            dataframe["rsi"].between(50, 82),
            dataframe["atr_pct"].between(0.008, 0.16),
            dataframe["btc_short_regime_1d"] == 0,
            dataframe["pair_long_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "futures_rs_long_rotation",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = (
            (dataframe["rs_rank_pct"] > self.exit_rank_pct)
            | (dataframe["rs_score"] < 0)
            | (dataframe["close"] < dataframe["exit_low"])
            | (dataframe["ema_20"] < dataframe["ema_50"])
            | (dataframe["market_breadth"] < 0.18)
            | (dataframe["rsi"] < 42)
        )
        dataframe.loc[(dataframe["volume"] > 0) & exit_conditions, ["exit_long", "exit_tag"]] = (
            1,
            "futures_rs_rotation_exit",
        )
        return dataframe


class FuturesRelativeStrengthLongRotation3xStrategy(FuturesRelativeStrengthLongRotationStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.105
    stoploss = -0.18
    minimal_roi = {
        "0": 0.72,
        "144": 0.30,
        "480": 0.10,
        "1200": 0,
    }


class FuturesRelativeStrengthLongRotationTop2Strategy(FuturesRelativeStrengthLongRotation3xStrategy):
    rank_slots = 2.0
    min_rank_pct = 0.08
    min_rs_score = 0.95
    min_market_breadth = 0.24
    exit_rank_pct = 0.32
    target_trade_volatility = 0.115
    max_stake_fraction = 0.95
    stoploss = -0.20


class FuturesRelativeStrengthLongRotationTop1Strategy(FuturesRelativeStrengthLongRotation3xStrategy):
    rank_slots = 1.0
    min_rank_pct = 0.04
    min_rs_score = 1.10
    min_market_breadth = 0.22
    exit_rank_pct = 0.25
    target_trade_volatility = 0.125
    max_stake_fraction = 1.0
    stoploss = -0.22
    minimal_roi = {
        "0": 0.90,
        "168": 0.36,
        "540": 0.12,
        "1320": 0,
    }


class FuturesRelativeStrengthLongRotationLooseStrategy(FuturesRelativeStrengthLongRotation3xStrategy):
    rank_slots = 4.0
    min_rank_pct = 0.14
    min_rs_score = 0.55
    min_market_breadth = 0.12
    exit_rank_pct = 0.55
    target_trade_volatility = 0.110
    max_stake_fraction = 1.0
    stoploss = -0.20
    trailing_stop_positive = 0.040
    trailing_stop_positive_offset = 0.14
    minimal_roi = {
        "0": 0.82,
        "192": 0.34,
        "720": 0.12,
        "1440": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        top_count_pct = max(self.min_rank_pct, self.rank_slots / max(4, len(self.dp.current_whitelist()) if self.dp else 30))
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["rs_rank_pct"] <= top_count_pct,
            dataframe["rs_score"] > self.min_rs_score,
            dataframe["market_breadth"] > self.min_market_breadth,
            dataframe["close"] > dataframe["ema_50"],
            dataframe["ema_20"] > dataframe["ema_100"] * 0.98,
            dataframe["momentum_42"] > 0.02,
            dataframe["momentum_126"] > -0.10,
            dataframe["rsi"].between(48, 84),
            dataframe["atr_pct"].between(0.008, 0.18),
            dataframe["btc_short_regime_1d"] == 0,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "futures_rs_loose_rotation",
        )
        return dataframe


class FuturesRelativeStrengthLongRotationLoose2xStrategy(FuturesRelativeStrengthLongRotationLooseStrategy):
    leverage_value = 2.0
    target_trade_volatility = 0.080
    max_stake_fraction = 0.85
    stoploss = -0.14
    trailing_stop_positive = 0.032
    trailing_stop_positive_offset = 0.10
    minimal_roi = {
        "0": 0.46,
        "144": 0.20,
        "540": 0.075,
        "1200": 0,
    }


class FuturesRelativeStrengthLongRotationLooseGuardStrategy(FuturesRelativeStrengthLongRotationLooseStrategy):
    target_trade_volatility = 0.095
    max_stake_fraction = 0.85
    stoploss = -0.16
    trailing_stop_positive = 0.032
    trailing_stop_positive_offset = 0.095
    exit_rank_pct = 0.45
    min_market_breadth = 0.18

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 18,
                "trade_limit": 3,
                "stop_duration_candles": 18,
                "required_profit": 0.0,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 60,
                "trade_limit": 12,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.16,
                "calculation_mode": "equity",
            },
        ]


class FuturesRelativeStrengthLongRotationLooseGuardHighStrategy(FuturesRelativeStrengthLongRotationLooseGuardStrategy):
    target_trade_volatility = 0.115
    max_stake_fraction = 0.95
    stoploss = -0.18
    trailing_stop_positive = 0.038
    trailing_stop_positive_offset = 0.12
    minimal_roi = {
        "0": 0.92,
        "192": 0.38,
        "720": 0.14,
        "1440": 0,
    }


class FuturesPrecomputedRelativeStrengthLooseGuardStrategy(FuturesRelativeStrengthLongRotationLooseGuardStrategy):
    feature_dir = Path(__file__).resolve().parents[1] / "cross_sectional_features"
    universe_size = 30

    @staticmethod
    def _feature_path(pair: str, timeframe: str) -> Path:
        slug = pair.replace("/", "_").replace(":", "_")
        return FuturesPrecomputedRelativeStrengthLooseGuardStrategy.feature_dir / f"{slug}-{timeframe}-cross_sectional.feather"

    def _merge_precomputed_features(self, dataframe: DataFrame, pair: str) -> DataFrame:
        path = self._feature_path(pair, self.timeframe)
        if not path.exists():
            dataframe["rs_rank_pct"] = 1.0
            dataframe["market_breadth"] = 0.0
            return dataframe

        features = pd.read_feather(path)
        features = features[["date", "rs_rank_pct", "market_breadth"]].copy()
        features["date"] = pd.to_datetime(features["date"], utc=True)
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
        return dataframe.merge(features, on="date", how="left")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_42"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["momentum_126"] = dataframe["close"] / dataframe["close"].shift(126) - 1.0
        dataframe["volatility_63"] = dataframe["close"].pct_change().rolling(63, min_periods=63).std()
        dataframe["rs_score"] = (
            dataframe["momentum_42"] * 0.65 + dataframe["momentum_126"] * 0.35
        ) / dataframe["volatility_63"].replace(0, pd.NA)
        dataframe = self._merge_precomputed_features(dataframe, metadata["pair"])
        dataframe["rs_rank_pct"] = dataframe["rs_rank_pct"].fillna(1.0)
        dataframe["market_breadth"] = dataframe["market_breadth"].fillna(0.0)
        dataframe["exit_low"] = dataframe["low"].rolling(12, min_periods=12).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        top_count_pct = max(self.min_rank_pct, self.rank_slots / max(4, self.universe_size))
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["rs_rank_pct"] <= top_count_pct,
            dataframe["rs_score"] > self.min_rs_score,
            dataframe["market_breadth"] > self.min_market_breadth,
            dataframe["close"] > dataframe["ema_50"],
            dataframe["ema_20"] > dataframe["ema_100"] * 0.98,
            dataframe["momentum_42"] > 0.02,
            dataframe["momentum_126"] > -0.10,
            dataframe["rsi"].between(48, 84),
            dataframe["atr_pct"].between(0.008, 0.18),
            dataframe["btc_short_regime_1d"] == 0,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "futures_precomputed_rs_loose_guard",
        )
        return dataframe


class FuturesPrecomputedRelativeStrengthLooseGuardHighStrategy(FuturesPrecomputedRelativeStrengthLooseGuardStrategy):
    target_trade_volatility = 0.115
    max_stake_fraction = 0.95
    stoploss = -0.18
    trailing_stop_positive = 0.038
    trailing_stop_positive_offset = 0.12
    minimal_roi = {
        "0": 0.92,
        "192": 0.38,
        "720": 0.14,
        "1440": 0,
    }


class FuturesPrecomputedRelativeStrengthLooseGuardDefensiveStrategy(FuturesPrecomputedRelativeStrengthLooseGuardStrategy):
    target_trade_volatility = 0.075
    max_stake_fraction = 0.70
    stoploss = -0.12
    min_market_breadth = 0.24
    exit_rank_pct = 0.38
    trailing_stop_positive = 0.028
    trailing_stop_positive_offset = 0.085
    minimal_roi = {
        "0": 0.50,
        "144": 0.22,
        "540": 0.08,
        "1200": 0,
    }


class FuturesPrecomputedRelativeStrengthLooseGuardStrictStrategy(FuturesPrecomputedRelativeStrengthLooseGuardStrategy):
    rank_slots = 3.0
    min_rank_pct = 0.10
    min_rs_score = 0.70
    min_market_breadth = 0.30
    exit_rank_pct = 0.35
    target_trade_volatility = 0.080
    max_stake_fraction = 0.75
    stoploss = -0.13
    trailing_stop_positive = 0.030
    trailing_stop_positive_offset = 0.090


class FuturesSinglePairMomentumGuardStrategy(AlternativeFuturesBase):
    can_short = False
    leverage_value = 3.0
    target_trade_volatility = 0.100
    max_stake_fraction = 0.85
    stoploss = -0.16
    trailing_stop_positive = 0.034
    trailing_stop_positive_offset = 0.11
    minimal_roi = {
        "0": 0.78,
        "168": 0.32,
        "600": 0.11,
        "1320": 0,
    }

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 18,
                "trade_limit": 3,
                "stop_duration_candles": 18,
                "required_profit": 0.0,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 60,
                "trade_limit": 12,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.16,
                "calculation_mode": "equity",
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_21"] = dataframe["close"] / dataframe["close"].shift(21) - 1.0
        dataframe["momentum_42"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["momentum_126"] = dataframe["close"] / dataframe["close"].shift(126) - 1.0
        dataframe["volatility_63"] = dataframe["close"].pct_change().rolling(63, min_periods=63).std()
        dataframe["rs_score"] = (
            dataframe["momentum_42"] * 0.65 + dataframe["momentum_126"] * 0.35
        ) / dataframe["volatility_63"].replace(0, pd.NA)
        dataframe["rs_score_slope"] = dataframe["rs_score"] - dataframe["rs_score"].shift(12)
        dataframe["exit_low"] = dataframe["low"].rolling(12, min_periods=12).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_50"],
            dataframe["ema_20"] > dataframe["ema_100"] * 0.98,
            dataframe["momentum_21"] > 0.015,
            dataframe["momentum_42"] > 0.04,
            dataframe["momentum_126"] > -0.08,
            dataframe["rs_score"] > 0.70,
            dataframe["rs_score_slope"] > -0.50,
            dataframe["rsi"].between(50, 84),
            dataframe["atr_pct"].between(0.008, 0.17),
            dataframe["btc_short_regime_1d"] == 0,
            dataframe["pair_long_regime_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "single_pair_momentum_guard",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = (
            (dataframe["rs_score"] < -0.15)
            | (dataframe["close"] < dataframe["exit_low"])
            | (dataframe["ema_20"] < dataframe["ema_50"])
            | (dataframe["rsi"] < 42)
            | (dataframe["btc_short_regime_1d"] == 1)
        )
        dataframe.loc[(dataframe["volume"] > 0) & exit_conditions, ["exit_long", "exit_tag"]] = (
            1,
            "single_pair_momentum_exit",
        )
        return dataframe


class FuturesTrainWinnersMomentumGuardStrategy(FuturesSinglePairMomentumGuardStrategy):
    allowed_pairs = {
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "AVAX/USDT:USDT",
        "TRX/USDT:USDT",
        "ADA/USDT:USDT",
        "XMR/USDT:USDT",
    }
    target_trade_volatility = 0.110
    max_stake_fraction = 0.90

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata["pair"] not in self.allowed_pairs:
            return dataframe
        return super().populate_entry_trend(dataframe, metadata)


class AlternativeSpotBase(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    informative_timeframe = "1d"
    startup_candle_count = 500
    can_short = False

    minimal_roi = {
        "0": 0.36,
        "120": 0.16,
        "360": 0.05,
        "900": 0,
    }

    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    btc_pair = "BTC/USDT"
    target_trade_volatility = 0.06
    min_stake_fraction = 0.30
    max_stake_fraction = 1.0

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 3,
                "stop_duration_candles": 12,
                "required_profit": 0.0,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 90,
                "trade_limit": 18,
                "stop_duration_candles": 18,
                "max_allowed_drawdown": 0.18,
                "calculation_mode": "equity",
            },
        ]

    def informative_pairs(self):
        pairs = {(self.btc_pair, self.informative_timeframe)}
        if self.dp:
            pairs.update((pair, self.informative_timeframe) for pair in self.dp.current_whitelist())
        return sorted(pairs)

    @staticmethod
    def _daily_context(dataframe: DataFrame, prefix: str) -> DataFrame:
        dataframe[f"{prefix}_ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe[f"{prefix}_ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe[f"{prefix}_ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe[f"{prefix}_rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe[f"{prefix}_momentum_30"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0
        dataframe[f"{prefix}_momentum_90"] = dataframe["close"] / dataframe["close"].shift(90) - 1.0
        dataframe[f"{prefix}_volatility_30"] = dataframe["close"].pct_change().rolling(30, min_periods=30).std()
        return dataframe

    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["btc_risk_on_1d"] = 1
            dataframe["pair_risk_on_1d"] = 1
            dataframe["pair_daily_rsi_1d"] = 50
            return dataframe

        btc_daily = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=self.informative_timeframe)
        btc_daily = self._daily_context(btc_daily, "btc")
        btc_daily["btc_risk_on"] = (
            (btc_daily["close"] > btc_daily["btc_ema_100"])
            & (btc_daily["btc_ema_50"] > btc_daily["btc_ema_200"])
            & (btc_daily["btc_momentum_30"] > -0.02)
            & (btc_daily["btc_volatility_30"] < 0.09)
        ).astype(int)
        dataframe = merge_informative_pair(
            dataframe,
            btc_daily[["date", "btc_risk_on"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        pair_daily = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        pair_daily = self._daily_context(pair_daily, "pair")
        pair_daily["pair_risk_on"] = (
            (pair_daily["close"] > pair_daily["pair_ema_100"])
            & (pair_daily["pair_ema_50"] > pair_daily["pair_ema_200"])
            & (pair_daily["pair_momentum_30"] > -0.04)
        ).astype(int)
        pair_daily["pair_daily_rsi"] = pair_daily["pair_rsi"]
        dataframe = merge_informative_pair(
            dataframe,
            pair_daily[["date", "pair_risk_on", "pair_daily_rsi"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )
        return dataframe

    def _relative_strength_rank(self, dataframe: DataFrame, pair: str, lookback: int) -> pd.Series:
        if not self.dp:
            return pd.Series(1.0, index=dataframe.index)

        scores = pd.DataFrame(index=dataframe["date"])
        scores[pair] = dataframe.set_index("date")["relative_strength_score"]
        for candidate in self.dp.current_whitelist():
            if candidate == pair:
                continue
            candidate_df = self.dp.get_pair_dataframe(candidate, self.timeframe)
            if candidate_df.empty:
                continue
            momentum = candidate_df["close"] / candidate_df["close"].shift(lookback) - 1.0
            volatility = candidate_df["close"].pct_change().rolling(lookback, min_periods=lookback).std()
            candidate_score = momentum / volatility.replace(0, pd.NA)
            scores[candidate] = candidate_score.set_axis(candidate_df["date"]).reindex(scores.index)
        return scores.rank(axis=1, ascending=False, method="first")[pair].to_numpy()

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if not self.dp:
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or "atr_pct" not in dataframe:
            return proposed_stake
        atr_pct = dataframe.iloc[-1]["atr_pct"]
        if pd.isna(atr_pct) or atr_pct <= 0:
            return proposed_stake

        fraction = self.target_trade_volatility / float(atr_pct)
        fraction = min(self.max_stake_fraction, max(self.min_stake_fraction, fraction))
        stake = proposed_stake * fraction
        if min_stake:
            stake = max(min_stake, stake)
        return min(max_stake, stake)


class SpotRelativeStrengthDefensiveRotationStrategy(AlternativeSpotBase):
    rank_top_n = 3
    exit_rank_threshold = 6

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_42"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["momentum_126"] = dataframe["close"] / dataframe["close"].shift(126) - 1.0
        dataframe["volatility_42"] = dataframe["close"].pct_change().rolling(42, min_periods=42).std()
        dataframe["relative_strength_score"] = dataframe["momentum_42"] / dataframe["volatility_42"].replace(0, pd.NA)
        dataframe["relative_strength_rank"] = self._relative_strength_rank(dataframe, metadata["pair"], 42)
        dataframe["exit_low"] = dataframe["low"].rolling(10, min_periods=10).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_100"],
            dataframe["ema_20"] > dataframe["ema_50"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["momentum_42"] > 0.035,
            dataframe["momentum_126"] > 0.02,
            dataframe["relative_strength_rank"] <= self.rank_top_n,
            dataframe["relative_strength_score"] > 0,
            dataframe["rsi"].between(50, 78),
            dataframe["adx"] > 14,
            dataframe["atr_pct"].between(0.008, 0.12),
            dataframe["btc_risk_on_1d"] == 1,
            dataframe["pair_risk_on_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "defensive_rs_rotation",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = (
            (dataframe["relative_strength_rank"] > self.exit_rank_threshold)
            | (dataframe["close"] < dataframe["exit_low"])
            | (dataframe["ema_20"] < dataframe["ema_50"])
            | (dataframe["btc_risk_on_1d"] == 0)
            | (dataframe["rsi"] < 42)
        )
        dataframe.loc[(dataframe["volume"] > 0) & exit_conditions, ["exit_long", "exit_tag"]] = (
            1,
            "defensive_rotation_exit",
        )
        return dataframe


class SpotQualityPullbackRotationStrategy(SpotRelativeStrengthDefensiveRotationStrategy):
    rank_top_n = 4
    minimal_roi = {
        "0": 0.26,
        "96": 0.11,
        "240": 0.04,
        "720": 0,
    }
    stoploss = -0.085
    trailing_stop_positive = 0.028
    trailing_stop_positive_offset = 0.085
    target_trade_volatility = 0.052

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pullback = (dataframe["close"] < dataframe["ema_20"] * 1.015) & (dataframe["close"] > dataframe["ema_50"])
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            pullback,
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["momentum_42"] > 0.015,
            dataframe["momentum_126"] > 0.04,
            dataframe["relative_strength_rank"] <= self.rank_top_n,
            dataframe["rsi"].between(42, 62),
            dataframe["adx"] > 12,
            dataframe["atr_pct"].between(0.007, 0.10),
            dataframe["btc_risk_on_1d"] == 1,
            dataframe["pair_risk_on_1d"] == 1,
            dataframe["pair_daily_rsi_1d"] < 76,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "quality_pullback_rotation",
        )
        return dataframe
