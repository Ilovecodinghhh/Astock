from functools import reduce

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from pandas import DataFrame


class ScientificSpotBase(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    informative_timeframe = "1d"
    startup_candle_count = 500
    can_short = False

    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    btc_pair = "BTC/USDT"
    target_trade_volatility = 0.055
    min_stake_fraction = 0.35
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
            for pair in self.dp.current_whitelist():
                pairs.add((pair, self.informative_timeframe))
        return sorted(pairs)

    @staticmethod
    def _daily_indicators(dataframe: DataFrame, prefix: str) -> DataFrame:
        dataframe[f"{prefix}_ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe[f"{prefix}_ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe[f"{prefix}_ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe[f"{prefix}_rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe[f"{prefix}_momentum_30"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0
        dataframe[f"{prefix}_momentum_90"] = dataframe["close"] / dataframe["close"].shift(90) - 1.0
        dataframe[f"{prefix}_volatility_30"] = dataframe["close"].pct_change().rolling(30, min_periods=30).std()
        return dataframe

    @staticmethod
    def _trend_ok(dataframe: DataFrame, prefix: str) -> pd.Series:
        return (
            (dataframe["close"] > dataframe[f"{prefix}_ema_100"])
            & (dataframe[f"{prefix}_ema_50"] > dataframe[f"{prefix}_ema_200"])
            & (dataframe[f"{prefix}_momentum_30"] > -0.03)
            & (dataframe[f"{prefix}_volatility_30"] < 0.08)
        )

    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.dp:
            btc_daily = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=self.informative_timeframe)
            btc_daily = self._daily_indicators(btc_daily, "btc")
            btc_daily["btc_trend_ok"] = self._trend_ok(btc_daily, "btc").astype(int)
            dataframe = merge_informative_pair(
                dataframe,
                btc_daily[["date", "btc_trend_ok"]],
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
            )

            pair_daily = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
            pair_daily = self._daily_indicators(pair_daily, "pair")
            pair_daily["pair_trend_ok"] = self._trend_ok(pair_daily, "pair").astype(int)
            pair_daily["pair_daily_rsi"] = pair_daily["pair_rsi"]
            dataframe = merge_informative_pair(
                dataframe,
                pair_daily[["date", "pair_trend_ok", "pair_daily_rsi"]],
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
            )
        else:
            dataframe["btc_trend_ok_1d"] = 1
            dataframe["pair_trend_ok_1d"] = 1
            dataframe["pair_daily_rsi_1d"] = 50
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

        atr_pct = dataframe.iloc[-1]["atr_pct"]
        if pd.isna(atr_pct) or atr_pct <= 0:
            return proposed_stake

        fraction = self.target_trade_volatility / float(atr_pct)
        fraction = min(self.max_stake_fraction, max(self.min_stake_fraction, fraction))
        stake = proposed_stake * fraction
        if min_stake:
            stake = max(min_stake, stake)
        return min(max_stake, stake)


class CrossSectionalMomentumStrategy(ScientificSpotBase):
    minimal_roi = {
        "0": 0.55,
        "120": 0.24,
        "360": 0.10,
        "900": 0,
    }

    rank_top_n = 3
    buy_momentum_min = DecimalParameter(0.02, 0.20, default=0.06, decimals=3, space="buy", optimize=True)
    buy_rsi_max = IntParameter(68, 86, default=80, space="buy", optimize=True)
    exit_rank_threshold = 6

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_30"] = dataframe["close"] / dataframe["close"].shift(180) - 1.0
        dataframe["momentum_14"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["volatility_14"] = dataframe["close"].pct_change().rolling(42, min_periods=42).std()
        dataframe["risk_adj_momentum"] = dataframe["momentum_14"] / dataframe["volatility_14"].replace(0, pd.NA)
        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(12) - 1.0
        dataframe["momentum_rank"] = self._momentum_rank(dataframe, metadata["pair"])
        return self._merge_daily_context(dataframe, metadata)

    def _momentum_rank(self, dataframe: DataFrame, pair: str) -> pd.Series:
        if not self.dp:
            return pd.Series(1.0, index=dataframe.index)

        strength = pd.DataFrame(index=dataframe["date"])
        strength[pair] = dataframe.set_index("date")["risk_adj_momentum"]
        for candidate in self.dp.current_whitelist():
            if candidate == pair:
                continue
            candidate_df = self.dp.get_pair_dataframe(candidate, self.timeframe)
            if candidate_df.empty:
                continue
            candidate_strength = (
                candidate_df["close"] / candidate_df["close"].shift(42) - 1.0
            ) / candidate_df["close"].pct_change().rolling(42, min_periods=42).std().replace(0, pd.NA)
            strength[candidate] = candidate_strength.set_axis(candidate_df["date"]).reindex(strength.index)
        return strength.rank(axis=1, ascending=False, method="first")[pair].to_numpy()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe["ema_100"],
            dataframe["ema_50"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0,
            dataframe["momentum_30"] > self.buy_momentum_min.value,
            dataframe["risk_adj_momentum"] > 0,
            dataframe["momentum_rank"] <= self.rank_top_n,
            dataframe["rsi"] < self.buy_rsi_max.value,
            dataframe["atr_pct"].between(0.012, 0.12),
            dataframe["btc_trend_ok_1d"] == 1,
            dataframe["pair_trend_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "cross_sectional_momentum",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > 0,
            (
                (dataframe["momentum_rank"] > self.exit_rank_threshold)
                | (dataframe["close"] < dataframe["ema_100"])
                | (dataframe["rsi"] < 42)
                | (dataframe["pair_trend_ok_1d"] == 0)
            ),
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["exit_long", "exit_tag"]] = (
            1,
            "momentum_rotation_exit",
        )
        return dataframe


class CrossSectionalMomentumTop2Strategy(CrossSectionalMomentumStrategy):
    rank_top_n = 2
    buy_momentum_min = DecimalParameter(0.04, 0.24, default=0.08, decimals=3, space="buy", optimize=True)


class CrossSectionalMomentumAggressiveStrategy(CrossSectionalMomentumStrategy):
    rank_top_n = 4
    target_trade_volatility = 0.075
    stoploss = -0.14
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.14

    minimal_roi = {
        "0": 0.80,
        "180": 0.34,
        "540": 0.12,
        "1080": 0,
    }


class PullbackTrendStrategy(ScientificSpotBase):
    minimal_roi = {
        "0": 0.30,
        "96": 0.14,
        "240": 0.05,
        "720": 0,
    }

    pullback_rsi_min = IntParameter(34, 48, default=40, space="buy", optimize=True)
    pullback_rsi_max = IntParameter(48, 62, default=55, space="buy", optimize=True)
    trend_adx_min = IntParameter(14, 30, default=18, space="buy", optimize=True)
    exit_rsi = 44

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
        dataframe["bb_upper"], dataframe["bb_mid"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"],
            timeperiod=20,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(12) - 1.0
        dataframe["recent_high"] = dataframe["high"].rolling(18, min_periods=18).max().shift(1)
        dataframe["exit_low"] = dataframe["low"].rolling(8, min_periods=8).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trend = (
            (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["ema_50"] > dataframe["ema_200"])
            & (dataframe["ema_100_slope"] > 0)
            & (dataframe["adx"] > self.trend_adx_min.value)
        )
        pullback = (
            dataframe["rsi"].between(self.pullback_rsi_min.value, self.pullback_rsi_max.value)
            & (dataframe["low"] <= dataframe["ema_50"] * 1.015)
            & (dataframe["close"] > dataframe["bb_mid"])
        )
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            trend,
            pullback,
            dataframe["atr_pct"].between(0.010, 0.10),
            dataframe["btc_trend_ok_1d"] == 1,
            dataframe["pair_trend_ok_1d"] == 1,
            dataframe["pair_daily_rsi_1d"] < 78,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "trend_pullback",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > 0,
            (
                (dataframe["close"] < dataframe["exit_low"])
                | (dataframe["ema_20"] < dataframe["ema_50"])
                | (dataframe["rsi"] < self.exit_rsi)
                | (dataframe["pair_trend_ok_1d"] == 0)
            ),
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["exit_long", "exit_tag"]] = (
            1,
            "pullback_trend_exit",
        )
        return dataframe


class PullbackTrendFastStrategy(PullbackTrendStrategy):
    pullback_rsi_min = IntParameter(38, 52, default=44, space="buy", optimize=True)
    pullback_rsi_max = IntParameter(54, 66, default=60, space="buy", optimize=True)
    exit_rsi = 48


class PullbackTrendDeepStrategy(PullbackTrendStrategy):
    pullback_rsi_min = IntParameter(28, 42, default=34, space="buy", optimize=True)
    pullback_rsi_max = IntParameter(42, 56, default=50, space="buy", optimize=True)
    stoploss = -0.08
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.08


class VolatilityExpansionBreakoutStrategy(ScientificSpotBase):
    timeframe = "1h"
    startup_candle_count = 720

    minimal_roi = {
        "0": 0.18,
        "24": 0.08,
        "72": 0.03,
        "168": 0,
    }

    stoploss = -0.07
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.055
    target_trade_volatility = 0.035

    breakout_window = IntParameter(24, 96, default=48, space="buy", optimize=True)
    compression_window = IntParameter(72, 240, default=120, space="buy", optimize=True)
    volume_mult = DecimalParameter(1.05, 2.50, default=1.30, decimals=2, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_24"] = dataframe["volume"].rolling(24, min_periods=24).mean()
        dataframe["volume_mean_72"] = dataframe["volume"].rolling(72, min_periods=72).mean()
        dataframe["bb_upper"], dataframe["bb_mid"], dataframe["bb_lower"] = ta.BBANDS(
            dataframe["close"],
            timeperiod=40,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]

        for window in self.breakout_window.range:
            dataframe[f"breakout_high_{window}"] = dataframe["high"].rolling(window, min_periods=window).max().shift(1)

        for window in self.compression_window.range:
            dataframe[f"bb_width_floor_{window}"] = (
                dataframe["bb_width"].rolling(window, min_periods=window).quantile(0.35).shift(1)
            )

        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(24) - 1.0
        dataframe["exit_low"] = dataframe["low"].rolling(18, min_periods=18).min().shift(1)
        return self._merge_daily_context(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        breakout_window = self.breakout_window.value
        compression_window = self.compression_window.value
        compression = dataframe["bb_width"].shift(1) <= dataframe[f"bb_width_floor_{compression_window}"]
        expansion = dataframe["close"] > dataframe[f"breakout_high_{breakout_window}"]
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_72"] * self.volume_mult.value,
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0,
            dataframe["adx"] > 16,
            dataframe["rsi"].between(54, 82),
            dataframe["atr_pct"].between(0.004, 0.07),
            compression,
            expansion,
            dataframe["btc_trend_ok_1d"] == 1,
            dataframe["pair_trend_ok_1d"] == 1,
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "volatility_expansion_breakout",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["volume"] > 0,
            (
                (dataframe["close"] < dataframe["exit_low"])
                | (dataframe["ema_50"] < dataframe["ema_100"])
                | (dataframe["rsi"] < 45)
                | (dataframe["pair_trend_ok_1d"] == 0)
            ),
        ]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["exit_long", "exit_tag"]] = (
            1,
            "volatility_breakout_exit",
        )
        return dataframe


class VolatilityExpansionBreakoutLooseStrategy(VolatilityExpansionBreakoutStrategy):
    volume_mult = DecimalParameter(1.00, 2.00, default=1.10, decimals=2, space="buy", optimize=True)
    trailing_stop_positive = 0.014
    trailing_stop_positive_offset = 0.045


class VolatilityExpansionBreakoutAggressiveStrategy(VolatilityExpansionBreakoutStrategy):
    stoploss = -0.10
    target_trade_volatility = 0.05
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.075

    minimal_roi = {
        "0": 0.26,
        "36": 0.12,
        "120": 0.04,
        "240": 0,
    }
