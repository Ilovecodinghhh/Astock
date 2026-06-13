from functools import reduce
from datetime import datetime

import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from pandas import DataFrame


class BreakoutTrendStrategy(IStrategy):
    """
    Long-only trend breakout strategy for liquid Binance spot pairs.

    Signals use shifted rolling highs/lows so the current candle is never part
    of its own breakout threshold.
    """

    INTERFACE_VERSION = 3

    timeframe = "4h"
    startup_candle_count = 500
    can_short = False

    minimal_roi = {
        "0": 0.40,
        "120": 0.20,
        "360": 0.08,
        "720": 0
    }

    stoploss = -0.12
    trailing_stop = True
    trailing_stop_positive = 0.035
    trailing_stop_positive_offset = 0.12
    trailing_only_offset_is_reached = True

    breakout_window = IntParameter(18, 72, default=18, space="buy", optimize=True)
    exit_window = IntParameter(8, 36, default=8, space="sell", optimize=True)
    buy_adx = IntParameter(14, 34, default=20, space="buy", optimize=True)
    buy_atr_min = DecimalParameter(0.008, 0.04, default=0.012, decimals=3, space="buy", optimize=True)
    buy_atr_max = DecimalParameter(0.04, 0.14, default=0.10, decimals=3, space="buy", optimize=True)

    btc_pair = "BTC/USDT"
    eth_pair = "ETH/USDT"
    informative_timeframe = "1d"
    relative_strength_top_n = 8
    target_trade_volatility = 0.065
    min_stake_fraction = 0.50
    max_stake_fraction = 1.0
    use_btc_regime_filter = False
    btc_regime_mode = "strict"
    use_eth_regime_filter = False
    use_pair_daily_filter = True
    pair_daily_mode = "strict"
    use_relative_strength_filter = False
    use_volatility_stake = True
    use_chandelier_exit = False
    chandelier_atr_mult = 3.0
    exit_rsi = 42

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,
            },
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
                "max_allowed_drawdown": 0.16,
                "calculation_mode": "equity",
            },
        ]

    def informative_pairs(self):
        pairs = set()
        if self.use_btc_regime_filter:
            pairs.add((self.btc_pair, self.informative_timeframe))
        if self.use_eth_regime_filter:
            pairs.add((self.eth_pair, self.informative_timeframe))
        if self.use_pair_daily_filter and self.dp:
            for pair in self.dp.current_whitelist():
                pairs.add((pair, self.informative_timeframe))
        return sorted(pairs)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["volume_mean_12"] = dataframe["volume"].rolling(12, min_periods=12).mean()
        dataframe["momentum_14"] = dataframe["close"] / dataframe["close"].shift(42) - 1.0
        dataframe["momentum_30"] = dataframe["close"] / dataframe["close"].shift(180) - 1.0
        dataframe["volatility_14"] = dataframe["close"].pct_change().rolling(42, min_periods=42).std()
        dataframe["risk_adj_momentum"] = dataframe["momentum_14"] / dataframe["volatility_14"].replace(0, pd.NA)
        dataframe["chandelier_long"] = (
            dataframe["high"].rolling(22, min_periods=22).max().shift(1)
            - dataframe["atr"] * self.chandelier_atr_mult
        )

        for window in self.breakout_window.range:
            dataframe[f"breakout_high_{window}"] = dataframe["high"].rolling(window, min_periods=window).max().shift(1)

        for window in self.exit_window.range:
            dataframe[f"exit_low_{window}"] = dataframe["low"].rolling(window, min_periods=window).min().shift(1)

        dataframe["ema_100_slope"] = dataframe["ema_100"] / dataframe["ema_100"].shift(12) - 1.0

        if self.use_btc_regime_filter and self.dp:
            btc_daily = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=self.informative_timeframe)
            btc_daily = self._daily_regime_indicators(btc_daily, "btc")
            btc_daily["btc_regime_ok"] = self._market_regime_ok(btc_daily, "btc", self.btc_regime_mode).astype(int)
            dataframe = merge_informative_pair(
                dataframe,
                btc_daily,
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
            )
        else:
            dataframe["btc_regime_ok_1d"] = 1

        if self.use_eth_regime_filter and self.dp:
            eth_daily = self.dp.get_pair_dataframe(pair=self.eth_pair, timeframe=self.informative_timeframe)
            eth_daily = self._daily_regime_indicators(eth_daily, "eth")
            eth_daily["eth_regime_ok"] = self._market_regime_ok(eth_daily, "eth", "strict").astype(int)
            dataframe = merge_informative_pair(
                dataframe,
                eth_daily,
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
            )
        else:
            dataframe["eth_regime_ok_1d"] = 1

        if self.use_pair_daily_filter and self.dp:
            pair_daily = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
            pair_daily = self._daily_regime_indicators(pair_daily, "pair")
            pair_daily["pair_daily_regime_ok"] = self._pair_daily_regime_ok(
                pair_daily,
                self.pair_daily_mode,
            ).astype(int)
            pair_daily = pair_daily[["date", "pair_daily_regime_ok"]]
            dataframe = merge_informative_pair(
                dataframe,
                pair_daily,
                self.timeframe,
                self.informative_timeframe,
                ffill=True,
            )
        else:
            dataframe["pair_daily_regime_ok_1d"] = 1

        if self.use_relative_strength_filter:
            dataframe["relative_strength_rank"] = self._relative_strength_rank(dataframe, metadata["pair"])
        else:
            dataframe["relative_strength_rank"] = 1.0
        return dataframe

    @staticmethod
    def _daily_regime_indicators(dataframe: DataFrame, prefix: str) -> DataFrame:
        dataframe[f"{prefix}_ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe[f"{prefix}_ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe[f"{prefix}_momentum_30"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0
        dataframe[f"{prefix}_momentum_90"] = dataframe["close"] / dataframe["close"].shift(90) - 1.0
        dataframe[f"{prefix}_volatility_30"] = dataframe["close"].pct_change().rolling(30, min_periods=30).std()
        return dataframe

    @staticmethod
    def _market_regime_ok(dataframe: DataFrame, prefix: str, mode: str) -> pd.Series:
        if mode == "strict":
            return (
                (dataframe["close"] > dataframe[f"{prefix}_ema_200"])
                & (dataframe[f"{prefix}_ema_50"] > dataframe[f"{prefix}_ema_200"])
                & (dataframe[f"{prefix}_momentum_30"] > 0)
                & (dataframe[f"{prefix}_momentum_90"] > -0.05)
                & (dataframe[f"{prefix}_volatility_30"] < 0.055)
            )

        return (
            (
                (dataframe["close"] > dataframe[f"{prefix}_ema_200"])
                | (
                    (dataframe["close"] > dataframe[f"{prefix}_ema_50"])
                    & (dataframe[f"{prefix}_momentum_30"] > -0.02)
                )
                | (dataframe[f"{prefix}_momentum_90"] > 0.08)
            )
            & (dataframe[f"{prefix}_volatility_30"] < 0.09)
        )

    @staticmethod
    def _pair_daily_regime_ok(dataframe: DataFrame, mode: str) -> pd.Series:
        if mode == "strict":
            return (
                (dataframe["close"] > dataframe["pair_ema_50"])
                & (dataframe["close"] > dataframe["pair_ema_200"])
                & (dataframe["pair_ema_50"] > dataframe["pair_ema_200"])
                & (dataframe["pair_momentum_30"] > 0)
            )

        return (
            (dataframe["close"] > dataframe["pair_ema_200"])
            & (dataframe["pair_ema_50"] > dataframe["pair_ema_200"])
            & (dataframe["pair_momentum_30"] > -0.05)
        )

    def _relative_strength_rank(self, dataframe: DataFrame, pair: str) -> pd.Series:
        if not self.dp:
            return pd.Series(1.0, index=dataframe.index)

        try:
            whitelist = [candidate for candidate in self.dp.current_whitelist() if candidate != self.btc_pair]
        except Exception:
            return pd.Series(1.0, index=dataframe.index)

        if pair == self.btc_pair:
            whitelist = [self.btc_pair] + whitelist

        strength = pd.DataFrame(index=dataframe["date"])
        strength[pair] = dataframe.set_index("date")["risk_adj_momentum"]

        for candidate in whitelist:
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
        window = self.breakout_window.value
        conditions = [
            dataframe["volume"] > dataframe["volume_mean_12"],
            dataframe["close"] > dataframe[f"breakout_high_{window}"],
            dataframe["close"] > dataframe["ema_200"],
            dataframe["ema_50"] > dataframe["ema_100"],
            dataframe["ema_100"] > dataframe["ema_200"],
            dataframe["ema_100_slope"] > 0,
            dataframe["adx"] > self.buy_adx.value,
            dataframe["rsi"].between(52, 78),
            dataframe["atr_pct"] > self.buy_atr_min.value,
            dataframe["atr_pct"] < self.buy_atr_max.value,
        ]
        if self.use_btc_regime_filter:
            conditions.append(dataframe["btc_regime_ok_1d"] == 1)
        if self.use_eth_regime_filter:
            conditions.append(dataframe["eth_regime_ok_1d"] == 1)
        if self.use_pair_daily_filter:
            conditions.append(dataframe["pair_daily_regime_ok_1d"] == 1)
        if self.use_relative_strength_filter:
            conditions.append(dataframe["relative_strength_rank"] <= self.relative_strength_top_n)

        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "breakout_trend",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        window = self.exit_window.value
        trend_break = (
            (dataframe["close"] < dataframe[f"exit_low_{window}"])
            | (dataframe["ema_50"] < dataframe["ema_100"])
            | (dataframe["rsi"] < self.exit_rsi)
        )
        if self.use_chandelier_exit:
            trend_break = trend_break | (dataframe["close"] < dataframe["chandelier_long"])

        conditions = [
            dataframe["volume"] > 0,
            trend_break,
        ]

        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["exit_long", "exit_tag"]] = (
            1,
            "trend_break",
        )
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        if not self.use_volatility_stake or not self.dp:
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


class BreakoutBaselineStrategy(BreakoutTrendStrategy):
    use_btc_regime_filter = False
    use_relative_strength_filter = False
    use_volatility_stake = False


class BreakoutBtcRegimeStrategy(BreakoutBaselineStrategy):
    use_btc_regime_filter = True


class BreakoutStrictBtcRegimeStrategy(BreakoutBtcRegimeStrategy):
    btc_regime_mode = "strict"


class BreakoutEthRegimeStrategy(BreakoutBaselineStrategy):
    use_eth_regime_filter = True


class BreakoutBtcEthRegimeStrategy(BreakoutStrictBtcRegimeStrategy):
    use_eth_regime_filter = True


class BreakoutPairDailyTrendStrategy(BreakoutBaselineStrategy):
    use_pair_daily_filter = True


class BreakoutStrictPairDailyTrendStrategy(BreakoutPairDailyTrendStrategy):
    pair_daily_mode = "strict"


class BreakoutBtcPairDailyTrendStrategy(BreakoutStrictBtcRegimeStrategy):
    use_pair_daily_filter = True


class BreakoutBtcStrictPairDailyTrendStrategy(BreakoutBtcPairDailyTrendStrategy):
    pair_daily_mode = "strict"


class BreakoutRelativeStrengthStrategy(BreakoutBaselineStrategy):
    use_relative_strength_filter = True


class BreakoutTop4RelativeStrengthStrategy(BreakoutRelativeStrengthStrategy):
    relative_strength_top_n = 4


class BreakoutVolatilityStakeStrategy(BreakoutBaselineStrategy):
    use_volatility_stake = True


class BreakoutChandelierExitStrategy(BreakoutBaselineStrategy):
    use_chandelier_exit = True


class BreakoutFastRsiExitStrategy(BreakoutBaselineStrategy):
    exit_rsi = 48


class BreakoutStrictDefensiveStrategy(BreakoutTrendStrategy):
    btc_regime_mode = "strict"
    relative_strength_top_n = 4
    use_chandelier_exit = True
