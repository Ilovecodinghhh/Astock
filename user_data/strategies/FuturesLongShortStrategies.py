from functools import reduce
from math import floor

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame


class FuturesLongShortBase(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    informative_timeframe = "1d"
    startup_candle_count = 720
    can_short = True

    minimal_roi = {
        "0": 0.28,
        "72": 0.12,
        "240": 0.04,
        "720": 0,
    }

    stoploss = -0.16
    trailing_stop = True
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    btc_pair = "BTC/USDT:USDT"
    leverage_value = 2.0
    target_trade_volatility = 0.075
    min_stake_fraction = 0.25
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
                "trade_limit": 20,
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
            for pair in self.dp.current_whitelist():
                pairs.add((pair, self.informative_timeframe))
        return sorted(pairs)

    @staticmethod
    def _daily_indicators(dataframe: DataFrame, prefix: str) -> DataFrame:
        dataframe[f"{prefix}_ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe[f"{prefix}_ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe[f"{prefix}_ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe[f"{prefix}_momentum_30"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0
        dataframe[f"{prefix}_momentum_90"] = dataframe["close"] / dataframe["close"].shift(90) - 1.0
        dataframe[f"{prefix}_volatility_30"] = dataframe["close"].pct_change().rolling(30, min_periods=30).std()
        return dataframe

    @staticmethod
    def _daily_long_ok(dataframe: DataFrame, prefix: str) -> pd.Series:
        return (
            (dataframe["close"] > dataframe[f"{prefix}_ema_100"])
            & (dataframe[f"{prefix}_ema_50"] > dataframe[f"{prefix}_ema_200"])
            & (dataframe[f"{prefix}_momentum_30"] > -0.03)
            & (dataframe[f"{prefix}_volatility_30"] < 0.09)
        )

    @staticmethod
    def _daily_short_ok(dataframe: DataFrame, prefix: str) -> pd.Series:
        return (
            (dataframe["close"] < dataframe[f"{prefix}_ema_100"])
            & (dataframe[f"{prefix}_ema_50"] < dataframe[f"{prefix}_ema_200"])
            & (dataframe[f"{prefix}_momentum_30"] < 0.03)
            & (dataframe[f"{prefix}_volatility_30"] < 0.10)
        )

    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["btc_long_ok_1d"] = 1
            dataframe["btc_short_ok_1d"] = 1
            dataframe["pair_long_ok_1d"] = 1
            dataframe["pair_short_ok_1d"] = 1
            return dataframe

        btc_daily = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=self.informative_timeframe)
        btc_daily = self._daily_indicators(btc_daily, "btc")
        btc_daily["btc_long_ok"] = self._daily_long_ok(btc_daily, "btc").astype(int)
        btc_daily["btc_short_ok"] = self._daily_short_ok(btc_daily, "btc").astype(int)
        dataframe = merge_informative_pair(
            dataframe,
            btc_daily[["date", "btc_long_ok", "btc_short_ok"]],
            self.timeframe,
            self.informative_timeframe,
            ffill=True,
        )

        pair_daily = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        pair_daily = self._daily_indicators(pair_daily, "pair")
        pair_daily["pair_long_ok"] = self._daily_long_ok(pair_daily, "pair").astype(int)
        pair_daily["pair_short_ok"] = self._daily_short_ok(pair_daily, "pair").astype(int)
        dataframe = merge_informative_pair(
            dataframe,
            pair_daily[["date", "pair_long_ok", "pair_short_ok"]],
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

        candles = dataframe.copy()
        candle_dates = pd.to_datetime(candles["date"])
        if getattr(candle_dates.dt, "tz", None) is not None:
            candle_dates = candle_dates.dt.tz_convert("UTC").dt.tz_localize(None)
        candles = candles.loc[candle_dates <= candle_time]
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

    def _rank_percentile(self, dataframe: DataFrame, pair: str, column: str, ascending: bool) -> pd.Series:
        if not self.dp:
            return pd.Series(0.5, index=dataframe.index)

        ranks = pd.DataFrame(index=dataframe["date"])
        ranks[pair] = dataframe.set_index("date")[column]
        for candidate in self.dp.current_whitelist():
            if candidate == pair:
                continue
            candidate_df = self.dp.get_pair_dataframe(candidate, self.timeframe)
            if candidate_df.empty:
                continue
            candidate_momentum = candidate_df["close"] / candidate_df["close"].shift(42) - 1.0
            candidate_volatility = candidate_df["close"].pct_change().rolling(42, min_periods=42).std()
            candidate_score = candidate_momentum / candidate_volatility.replace(0, pd.NA)
            ranks[candidate] = candidate_score.set_axis(candidate_df["date"]).reindex(ranks.index)
        return ranks.rank(axis=1, ascending=ascending, pct=True, method="first")[pair].to_numpy()


class FuturesTrendLongShortStrategy(FuturesLongShortBase):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_14"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(12) - 1.0
        dataframe["long_exit_low"] = dataframe["low"].rolling(8, min_periods=8).min().shift(1)
        dataframe["short_exit_high"] = dataframe["high"].rolling(8, min_periods=8).max().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0,
            dataframe["momentum_14"] > 0.04,
            dataframe["adx"] > 18,
            dataframe["rsi"].between(52, 78),
            dataframe["atr_pct"].between(0.008, 0.12),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_100"],
            dataframe["ema_100"] < dataframe["ema_200"],
            dataframe["ema_100_slope"] < 0,
            dataframe["momentum_14"] < -0.04,
            dataframe["adx"] > 18,
            dataframe["rsi"].between(22, 48),
            dataframe["atr_pct"].between(0.008, 0.12),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "futures_trend_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "futures_trend_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit = (
            (dataframe["close"] < dataframe["long_exit_low"])
            | (dataframe["ema_50"] < dataframe["ema_100"])
            | (dataframe["rsi"] < 42)
        )
        short_exit = (
            (dataframe["close"] > dataframe["short_exit_high"])
            | (dataframe["ema_50"] > dataframe["ema_100"])
            | (dataframe["rsi"] > 58)
        )
        dataframe.loc[(dataframe["volume"] > 0) & long_exit, ["exit_long", "exit_tag"]] = (1, "trend_long_exit")
        dataframe.loc[(dataframe["volume"] > 0) & short_exit, ["exit_short", "exit_tag"]] = (1, "trend_short_exit")
        return dataframe


class FuturesTrendLongShort3xStrategy(FuturesTrendLongShortStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.09
    stoploss = -0.20


class FuturesBreakoutLongShortStrategy(FuturesLongShortBase):
    minimal_roi = {
        "0": 0.36,
        "96": 0.16,
        "360": 0.06,
        "900": 0,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["breakout_high"] = dataframe["high"].rolling(24, min_periods=24).max().shift(1)
        dataframe["breakout_low"] = dataframe["low"].rolling(24, min_periods=24).min().shift(1)
        dataframe["exit_low"] = dataframe["low"].rolling(10, min_periods=10).min().shift(1)
        dataframe["exit_high"] = dataframe["high"].rolling(10, min_periods=10).max().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["breakout_high"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["adx"] > 16,
            dataframe["rsi"].between(52, 82),
            dataframe["atr_pct"].between(0.008, 0.14),
            dataframe["btc_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["breakout_low"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_200"],
            dataframe["adx"] > 16,
            dataframe["rsi"].between(18, 48),
            dataframe["atr_pct"].between(0.008, 0.14),
            dataframe["btc_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "futures_breakout_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "futures_breakout_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0) & ((dataframe["close"] < dataframe["exit_low"]) | (dataframe["rsi"] < 42)),
            ["exit_long", "exit_tag"],
        ] = (1, "breakout_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0) & ((dataframe["close"] > dataframe["exit_high"]) | (dataframe["rsi"] > 58)),
            ["exit_short", "exit_tag"],
        ] = (1, "breakout_short_exit")
        return dataframe


class FuturesBreakoutLongShort3xStrategy(FuturesBreakoutLongShortStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.085


class FuturesCrossSectionalLongShortStrategy(FuturesLongShortBase):
    leverage_value = 2.0
    minimal_roi = {
        "0": 0.42,
        "120": 0.18,
        "420": 0.07,
        "960": 0,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_14"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["volatility_14"] = dataframe["close"].pct_change().rolling(42, min_periods=42).std()
        dataframe["risk_adj_momentum"] = dataframe["momentum_14"] / dataframe["volatility_14"].replace(0, pd.NA)
        dataframe["strong_rank_pct"] = self._rank_percentile(dataframe, metadata["pair"], "risk_adj_momentum", False)
        dataframe["weak_rank_pct"] = self._rank_percentile(dataframe, metadata["pair"], "risk_adj_momentum", True)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        top_count_pct = max(0.2, 2.0 / max(2, len(self.dp.current_whitelist()) if self.dp else 10))
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["strong_rank_pct"] <= top_count_pct,
            dataframe["risk_adj_momentum"] > 0,
            dataframe["close"] > dataframe["ema_100"],
            dataframe["rsi"].between(50, 82),
            dataframe["atr_pct"].between(0.008, 0.14),
            dataframe["btc_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["weak_rank_pct"] <= top_count_pct,
            dataframe["risk_adj_momentum"] < 0,
            dataframe["close"] < dataframe["ema_100"],
            dataframe["rsi"].between(18, 50),
            dataframe["atr_pct"].between(0.008, 0.14),
            dataframe["btc_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "cross_sectional_futures_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "cross_sectional_futures_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["strong_rank_pct"] > 0.60) | (dataframe["risk_adj_momentum"] < 0) | (dataframe["rsi"] < 42)),
            ["exit_long", "exit_tag"],
        ] = (1, "cross_sectional_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["weak_rank_pct"] > 0.60) | (dataframe["risk_adj_momentum"] > 0) | (dataframe["rsi"] > 58)),
            ["exit_short", "exit_tag"],
        ] = (1, "cross_sectional_short_exit")
        return dataframe


class FuturesCrossSectionalLongShort3xStrategy(FuturesCrossSectionalLongShortStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.09
    stoploss = -0.20


class FuturesMarketNeutralMomentumStrategy(FuturesCrossSectionalLongShortStrategy):
    leverage_value = 2.0
    target_trade_volatility = 0.065
    max_stake_fraction = 0.70

    minimal_roi = {
        "0": 0.30,
        "96": 0.12,
        "360": 0.04,
        "900": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        top_count_pct = max(0.2, 2.0 / max(2, len(self.dp.current_whitelist()) if self.dp else 10))
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["strong_rank_pct"] <= top_count_pct,
            dataframe["risk_adj_momentum"] > 0.25,
            dataframe["close"] > dataframe["ema_100"],
            dataframe["rsi"].between(52, 84),
            dataframe["atr_pct"].between(0.008, 0.13),
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["weak_rank_pct"] <= top_count_pct,
            dataframe["risk_adj_momentum"] < -0.25,
            dataframe["close"] < dataframe["ema_100"],
            dataframe["rsi"].between(16, 48),
            dataframe["atr_pct"].between(0.008, 0.13),
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "market_neutral_momentum_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "market_neutral_momentum_short",
        )
        return dataframe


class FuturesMarketNeutralMomentum3xStrategy(FuturesMarketNeutralMomentumStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.08
    stoploss = -0.18


class FuturesStrictRegimeTrendStrategy(FuturesTrendLongShortStrategy):
    leverage_value = 2.0
    target_trade_volatility = 0.06
    stoploss = -0.12
    trailing_stop_positive = 0.028
    trailing_stop_positive_offset = 0.08

    minimal_roi = {
        "0": 0.22,
        "72": 0.10,
        "240": 0.035,
        "720": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0.015,
            dataframe["momentum_14"] > 0.06,
            dataframe["adx"] > 22,
            dataframe["rsi"].between(55, 74),
            dataframe["atr_pct"].between(0.010, 0.10),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_100"],
            dataframe["ema_100"] < dataframe["ema_200"],
            dataframe["ema_100_slope"] < -0.015,
            dataframe["momentum_14"] < -0.06,
            dataframe["adx"] > 22,
            dataframe["rsi"].between(26, 45),
            dataframe["atr_pct"].between(0.010, 0.10),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "strict_regime_trend_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "strict_regime_trend_short",
        )
        return dataframe


class FuturesCounterTrendMeanReversionStrategy(FuturesLongShortBase):
    leverage_value = 2.0
    target_trade_volatility = 0.055
    stoploss = -0.10
    trailing_stop_positive = 0.022
    trailing_stop_positive_offset = 0.065

    minimal_roi = {
        "0": 0.18,
        "48": 0.08,
        "144": 0.025,
        "360": 0,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["bb_upper"], dataframe["bb_mid"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"],
            timeperiod=40,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe["z_distance"] = (dataframe["close"] - dataframe["bb_mid"]) / (dataframe["atr"] * 2)
        dataframe["mean_reversion_exit_high"] = dataframe["high"].rolling(10, min_periods=10).max().shift(1)
        dataframe["mean_reversion_exit_low"] = dataframe["low"].rolling(10, min_periods=10).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["close"] < dataframe["bb_lower"],
            dataframe["z_distance"] < -0.85,
            dataframe["rsi"].between(28, 44),
            dataframe["adx"] < 36,
            dataframe["atr_pct"].between(0.008, 0.12),
            dataframe["btc_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_200"],
            dataframe["close"] > dataframe["bb_upper"],
            dataframe["z_distance"] > 0.85,
            dataframe["rsi"].between(56, 72),
            dataframe["adx"] < 36,
            dataframe["atr_pct"].between(0.008, 0.12),
            dataframe["btc_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "countertrend_reversion_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "countertrend_reversion_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] > dataframe["bb_mid"]) | (dataframe["rsi"] > 56)),
            ["exit_long", "exit_tag"],
        ] = (1, "mean_reversion_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] < dataframe["bb_mid"]) | (dataframe["rsi"] < 44)),
            ["exit_short", "exit_tag"],
        ] = (1, "mean_reversion_short_exit")
        return dataframe


class FuturesMegaCapTrend3xStrategy(FuturesTrendLongShortStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.07
    max_stake_fraction = 0.65
    stoploss = -0.16
    allowed_pairs = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
    }

    minimal_roi = {
        "0": 0.34,
        "96": 0.16,
        "360": 0.06,
        "900": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata["pair"] not in self.allowed_pairs:
            return dataframe

        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0.02,
            dataframe["momentum_14"] > 0.08,
            dataframe["adx"] > 24,
            dataframe["rsi"].between(56, 76),
            dataframe["atr_pct"].between(0.010, 0.09),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_100"],
            dataframe["ema_100"] < dataframe["ema_200"],
            dataframe["ema_100_slope"] < -0.02,
            dataframe["momentum_14"] < -0.08,
            dataframe["adx"] > 24,
            dataframe["rsi"].between(24, 44),
            dataframe["atr_pct"].between(0.010, 0.09),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "megacap_trend_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "megacap_trend_short",
        )
        return dataframe


class FuturesBtcEthTrend3xStrategy(FuturesMegaCapTrend3xStrategy):
    allowed_pairs = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    }
    target_trade_volatility = 0.08


class FuturesHighConvictionBtcTrendStrategy(FuturesTrendLongShortStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.06
    max_stake_fraction = 0.70
    stoploss = -0.14
    allowed_pairs = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    }

    minimal_roi = {
        "0": 0.42,
        "144": 0.20,
        "480": 0.08,
        "1200": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata["pair"] not in self.allowed_pairs:
            return dataframe

        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0.03,
            dataframe["momentum_14"] > 0.11,
            dataframe["adx"] > 28,
            dataframe["rsi"].between(58, 74),
            dataframe["atr_pct"].between(0.012, 0.075),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_100"],
            dataframe["ema_100"] < dataframe["ema_200"],
            dataframe["ema_100_slope"] < -0.03,
            dataframe["momentum_14"] < -0.11,
            dataframe["adx"] > 28,
            dataframe["rsi"].between(26, 42),
            dataframe["atr_pct"].between(0.012, 0.075),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "high_conviction_btc_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "high_conviction_btc_short",
        )
        return dataframe


class FuturesRegimeScalpLongShortStrategy(FuturesLongShortBase):
    timeframe = "1h"
    startup_candle_count = 900
    leverage_value = 2.0
    target_trade_volatility = 0.045
    max_stake_fraction = 0.60
    stoploss = -0.065
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.040

    minimal_roi = {
        "0": 0.095,
        "12": 0.045,
        "48": 0.018,
        "144": 0,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_24"] = ta.EMA(dataframe, timeperiod=24)
        dataframe["ema_72"] = ta.EMA(dataframe, timeperiod=72)
        dataframe["ema_168"] = ta.EMA(dataframe, timeperiod=168)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24, min_periods=24).mean()
        dataframe["momentum_12"] = dataframe["close"] / dataframe["close"].shift(12) - 1.0
        dataframe["momentum_48"] = dataframe["close"] / dataframe["close"].shift(48) - 1.0
        dataframe["ema_72_slope"] = dataframe["ema_72"] / dataframe["ema_72"].shift(24) - 1.0
        dataframe["breakout_high"] = dataframe["high"].rolling(36, min_periods=36).max().shift(1)
        dataframe["breakout_low"] = dataframe["low"].rolling(36, min_periods=36).min().shift(1)
        dataframe["exit_low"] = dataframe["low"].rolling(12, min_periods=12).min().shift(1)
        dataframe["exit_high"] = dataframe["high"].rolling(12, min_periods=12).max().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"] * 1.05,
            dataframe["close"] > dataframe["breakout_high"],
            dataframe["close"] > dataframe["ema_168"],
            dataframe["ema_24"] > dataframe["ema_72"],
            dataframe["ema_72"] > dataframe["ema_168"],
            dataframe["ema_72_slope"] > 0.006,
            dataframe["momentum_12"] > 0.012,
            dataframe["momentum_48"] > 0.030,
            dataframe["adx"] > 20,
            dataframe["rsi"].between(54, 78),
            dataframe["atr_pct"].between(0.004, 0.060),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"] * 1.05,
            dataframe["close"] < dataframe["breakout_low"],
            dataframe["close"] < dataframe["ema_168"],
            dataframe["ema_24"] < dataframe["ema_72"],
            dataframe["ema_72"] < dataframe["ema_168"],
            dataframe["ema_72_slope"] < -0.006,
            dataframe["momentum_12"] < -0.012,
            dataframe["momentum_48"] < -0.030,
            dataframe["adx"] > 20,
            dataframe["rsi"].between(22, 46),
            dataframe["atr_pct"].between(0.004, 0.060),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "regime_scalp_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "regime_scalp_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] < dataframe["exit_low"]) | (dataframe["ema_24"] < dataframe["ema_72"])),
            ["exit_long", "exit_tag"],
        ] = (1, "regime_scalp_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] > dataframe["exit_high"]) | (dataframe["ema_24"] > dataframe["ema_72"])),
            ["exit_short", "exit_tag"],
        ] = (1, "regime_scalp_short_exit")
        return dataframe


class FuturesRegimeScalpLongShort3xStrategy(FuturesRegimeScalpLongShortStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.050
    max_stake_fraction = 0.55
    stoploss = -0.080


class FuturesBollingerFundingFadeStrategy(FuturesLongShortBase):
    timeframe = "1h"
    startup_candle_count = 900
    leverage_value = 2.0
    target_trade_volatility = 0.040
    max_stake_fraction = 0.55
    stoploss = -0.055
    trailing_stop_positive = 0.014
    trailing_stop_positive_offset = 0.030

    minimal_roi = {
        "0": 0.070,
        "8": 0.032,
        "36": 0.012,
        "96": 0,
    }

    @staticmethod
    def _empty_funding_frame(dataframe: DataFrame) -> DataFrame:
        return pd.DataFrame(
            {
                "date": dataframe["date"],
                "funding_close": 0.0,
                "funding_mean_24": 0.0,
            }
        )

    def _merge_funding_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["funding_close_1h"] = 0.0
            dataframe["funding_mean_24_1h"] = 0.0
            return dataframe

        try:
            funding = self.dp.get_pair_dataframe(
                pair=metadata["pair"],
                timeframe=self.timeframe,
                candle_type="funding_rate",
            )
        except TypeError:
            funding = self._empty_funding_frame(dataframe)

        if funding.empty:
            funding = self._empty_funding_frame(dataframe)
        else:
            funding = funding[["date", "close"]].copy()
            funding.rename(columns={"close": "funding_close"}, inplace=True)
            funding["funding_close"] = funding["funding_close"].shift(1)
            funding["funding_mean_24"] = funding["funding_close"].rolling(24, min_periods=12).mean()

        return merge_informative_pair(
            dataframe,
            funding[["date", "funding_close", "funding_mean_24"]],
            self.timeframe,
            self.timeframe,
            ffill=True,
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_72"] = ta.EMA(dataframe, timeperiod=72)
        dataframe["ema_168"] = ta.EMA(dataframe, timeperiod=168)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24, min_periods=24).mean()
        dataframe["bb_upper"], dataframe["bb_mid"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"],
            timeperiod=40,
            nbdevup=2.2,
            nbdevdn=2.2,
        )
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        dataframe["z_atr"] = (dataframe["close"] - dataframe["bb_mid"]) / dataframe["atr"].replace(0, pd.NA)
        dataframe = self._merge_funding_context(dataframe, metadata)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        funding = dataframe.get("funding_close_1h", pd.Series(0, index=dataframe.index)).fillna(0)
        funding_mean = dataframe.get("funding_mean_24_1h", pd.Series(0, index=dataframe.index)).fillna(0)

        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] > dataframe["ema_168"],
            dataframe["close"] < dataframe["bb_lower"],
            dataframe["z_atr"] < -1.25,
            dataframe["rsi"].between(24, 42),
            dataframe["adx"] < 34,
            dataframe["bb_width"].between(0.025, 0.24),
            dataframe["atr_pct"].between(0.004, 0.065),
            funding <= funding_mean + 0.00025,
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] < dataframe["ema_168"],
            dataframe["close"] > dataframe["bb_upper"],
            dataframe["z_atr"] > 1.25,
            dataframe["rsi"].between(58, 76),
            dataframe["adx"] < 34,
            dataframe["bb_width"].between(0.025, 0.24),
            dataframe["atr_pct"].between(0.004, 0.065),
            funding >= funding_mean - 0.00025,
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "funding_fade_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "funding_fade_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0) & ((dataframe["close"] > dataframe["bb_mid"]) | (dataframe["rsi"] > 54)),
            ["exit_long", "exit_tag"],
        ] = (1, "funding_fade_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0) & ((dataframe["close"] < dataframe["bb_mid"]) | (dataframe["rsi"] < 46)),
            ["exit_short", "exit_tag"],
        ] = (1, "funding_fade_short_exit")
        return dataframe


class FuturesCarryMomentumStrategy(FuturesLongShortBase):
    timeframe = "1h"
    startup_candle_count = 900
    leverage_value = 2.0
    target_trade_volatility = 0.045
    max_stake_fraction = 0.60
    stoploss = -0.070
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.045

    minimal_roi = {
        "0": 0.090,
        "16": 0.040,
        "48": 0.016,
        "144": 0,
    }

    def _merge_carry_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            dataframe["funding_rate_1h"] = 0.0
            dataframe["funding_rank_72_1h"] = 0.5
            dataframe["basis_pct_1h"] = 0.0
            dataframe["basis_rank_72_1h"] = 0.5
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
            funding["funding_rank_72"] = funding["funding_rate"].rolling(72, min_periods=24).rank(pct=True)
            carry = carry.merge(funding[["date", "funding_rate", "funding_rank_72"]], on="date", how="left")
        else:
            carry["funding_rate"] = 0.0
            carry["funding_rank_72"] = 0.5

        if not mark.empty:
            mark = mark[["date", "close"]].copy()
            mark.rename(columns={"close": "mark_close"}, inplace=True)
            carry = carry.merge(mark, on="date", how="left")
            carry["basis_pct"] = (carry["futures_close"] - carry["mark_close"]) / carry["mark_close"]
            carry["basis_pct"] = carry["basis_pct"].shift(1)
            carry["basis_rank_72"] = carry["basis_pct"].rolling(72, min_periods=24).rank(pct=True)
        else:
            carry["basis_pct"] = 0.0
            carry["basis_rank_72"] = 0.5

        informative = carry[["date", "funding_rate", "funding_rank_72", "basis_pct", "basis_rank_72"]]
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
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24, min_periods=24).mean()
        dataframe["momentum_24"] = dataframe["close"] / dataframe["close"].shift(24) - 1.0
        dataframe["momentum_72"] = dataframe["close"] / dataframe["close"].shift(72) - 1.0
        dataframe["range_high_24"] = dataframe["high"].rolling(24, min_periods=24).max().shift(1)
        dataframe["range_low_24"] = dataframe["low"].rolling(24, min_periods=24).min().shift(1)
        dataframe["exit_low"] = dataframe["low"].rolling(12, min_periods=12).min().shift(1)
        dataframe["exit_high"] = dataframe["high"].rolling(12, min_periods=12).max().shift(1)
        dataframe = self._merge_carry_context(dataframe, metadata)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        funding_rank = dataframe.get("funding_rank_72_1h", pd.Series(0.5, index=dataframe.index)).fillna(0.5)
        basis_rank = dataframe.get("basis_rank_72_1h", pd.Series(0.5, index=dataframe.index)).fillna(0.5)

        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] > dataframe["ema_168"],
            dataframe["ema_24"] > dataframe["ema_72"],
            dataframe["momentum_24"] > 0.010,
            dataframe["momentum_72"] > 0.025,
            dataframe["close"] > dataframe["range_high_24"],
            funding_rank >= 0.55,
            basis_rank >= 0.55,
            dataframe["adx"] > 16,
            dataframe["rsi"].between(52, 78),
            dataframe["atr_pct"].between(0.004, 0.065),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_24"],
            dataframe["close"] < dataframe["ema_168"],
            dataframe["ema_24"] < dataframe["ema_72"],
            dataframe["momentum_24"] < -0.010,
            dataframe["momentum_72"] < -0.025,
            dataframe["close"] < dataframe["range_low_24"],
            funding_rank <= 0.45,
            basis_rank <= 0.45,
            dataframe["adx"] > 16,
            dataframe["rsi"].between(22, 48),
            dataframe["atr_pct"].between(0.004, 0.065),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "carry_momentum_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "carry_momentum_short",
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
        ] = (1, "carry_momentum_long_exit")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] > dataframe["exit_high"])
                | (dataframe["ema_24"] > dataframe["ema_72"])
                | (dataframe["rsi"] > 56)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "carry_momentum_short_exit")
        return dataframe


class FuturesRegimeSwitchCrossSectionalStrategy(FuturesCrossSectionalLongShortStrategy):
    leverage_value = 2.0
    target_trade_volatility = 0.055
    max_stake_fraction = 0.60
    stoploss = -0.12
    trailing_stop_positive = 0.026
    trailing_stop_positive_offset = 0.075

    minimal_roi = {
        "0": 0.24,
        "96": 0.10,
        "360": 0.035,
        "900": 0,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["ema_50_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(18) - 1.0
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        top_count_pct = max(0.2, 2.0 / max(2, len(self.dp.current_whitelist()) if self.dp else 10))
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"] * 1.05,
            dataframe["strong_rank_pct"] <= top_count_pct,
            dataframe["risk_adj_momentum"] > 0.30,
            dataframe["close"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_50_slope"] > 0.006,
            dataframe["adx"] > 18,
            dataframe["rsi"].between(52, 76),
            dataframe["atr_pct"].between(0.008, 0.10),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"] * 1.05,
            dataframe["weak_rank_pct"] <= top_count_pct,
            dataframe["risk_adj_momentum"] < -0.30,
            dataframe["close"] < dataframe["ema_100"],
            dataframe["ema_100"] < dataframe["ema_200"],
            dataframe["ema_50_slope"] < -0.006,
            dataframe["adx"] > 18,
            dataframe["rsi"].between(24, 48),
            dataframe["atr_pct"].between(0.008, 0.10),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "regime_switch_xsec_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "regime_switch_xsec_short",
        )
        return dataframe


class FuturesRegimeSwitchCrossSectional3xStrategy(FuturesRegimeSwitchCrossSectionalStrategy):
    leverage_value = 3.0
    target_trade_volatility = 0.065
    max_stake_fraction = 0.55
    stoploss = -0.15


class FuturesBullBearBreakoutStrategy(FuturesBreakoutLongShortStrategy):
    leverage_value = 2.0
    target_trade_volatility = 0.055
    max_stake_fraction = 0.60
    stoploss = -0.12
    trailing_stop_positive = 0.026
    trailing_stop_positive_offset = 0.075

    minimal_roi = {
        "0": 0.26,
        "96": 0.12,
        "360": 0.04,
        "900": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"] * 1.05,
            dataframe["close"] > dataframe["breakout_high"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["adx"] > 20,
            dataframe["rsi"].between(54, 76),
            dataframe["atr_pct"].between(0.008, 0.10),
            dataframe["btc_long_ok_1d"] == 1,
            dataframe["pair_long_ok_1d"] == 1,
        ]
        short_conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"] * 1.05,
            dataframe["close"] < dataframe["breakout_low"],
            dataframe["close"] < dataframe["ema_200"],
            dataframe["ema_50"] < dataframe["ema_200"],
            dataframe["adx"] > 20,
            dataframe["rsi"].between(24, 46),
            dataframe["atr_pct"].between(0.008, 0.10),
            dataframe["btc_short_ok_1d"] == 1,
            dataframe["pair_short_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "bull_bear_breakout_long",
        )
        dataframe.loc[reduce(lambda left, right: left & right, short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "bull_bear_breakout_short",
        )
        return dataframe


class FuturesBollingerFundingFadeLongOnlyStrategy(FuturesBollingerFundingFadeStrategy):
    leverage_value = 2.0
    target_trade_volatility = 0.035
    max_stake_fraction = 0.50

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        dataframe["enter_short"] = 0
        dataframe.loc[dataframe["enter_tag"] == "funding_fade_short", "enter_tag"] = None
        return dataframe


class FuturesBollingerFundingFadeLongOnlyTightStrategy(FuturesBollingerFundingFadeLongOnlyStrategy):
    target_trade_volatility = 0.030
    max_stake_fraction = 0.42
    minimal_roi = {
        "0": 0.055,
        "6": 0.026,
        "24": 0.010,
        "72": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        funding = dataframe.get("funding_close_1h", pd.Series(0, index=dataframe.index)).fillna(0)
        funding_mean = dataframe.get("funding_mean_24_1h", pd.Series(0, index=dataframe.index)).fillna(0)
        tighten = funding <= funding_mean + 0.00015
        dataframe.loc[~tighten, "enter_long"] = 0
        dataframe.loc[~tighten, "enter_tag"] = None
        return dataframe


class FuturesBollingerFundingFadeLongOnlyFastStrategy(FuturesBollingerFundingFadeLongOnlyStrategy):
    target_trade_volatility = 0.045
    max_stake_fraction = 0.60
    stoploss = -0.045
    trailing_stop_positive = 0.011
    trailing_stop_positive_offset = 0.022
    minimal_roi = {
        "0": 0.040,
        "4": 0.018,
        "16": 0.006,
        "48": 0,
    }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        dataframe["enter_short"] = 0
        dataframe.loc[dataframe["enter_tag"].isna(), "enter_long"] = dataframe.loc[
            dataframe["enter_tag"].isna(), "enter_long"
        ].fillna(0)
        return dataframe
