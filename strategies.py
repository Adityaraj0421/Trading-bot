"""
Strategy Library — 6 proven strategies that activate based on market regime.

Each strategy takes OHLCV data with indicators and returns a signal + confidence.
The Strategy Engine selects which strategy to use based on the detected regime.

Strategies:
  1. MomentumStrategy      — Trend following (for TRENDING_UP / TRENDING_DOWN)
  2. MeanReversionStrategy  — Buy dips / sell rallies (for RANGING)
  3. BreakoutStrategy       — Catch new trends early (for RANGING → TRENDING)
  4. GridStrategy           — Grid trading (for RANGING / low volatility)
  5. ScalpingStrategy       — Quick in-and-out on micro moves (for any regime)
  6. SentimentStrategy      — Contrarian + momentum based on Fear/Greed

Inspired by: QuantInsti, Freqtrade, 3Commas, Renaissance Technologies research
"""

import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from regime_detector import MarketRegime
from sentiment import SentimentState

_log = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Trading signal produced by a strategy with confidence and risk parameters."""

    signal: str          # BUY, SELL, HOLD
    confidence: float    # 0.0 to 1.0
    strategy_name: str
    reason: str
    suggested_sl_pct: float = 0.02   # Suggested stop-loss %
    suggested_tp_pct: float = 0.05   # Suggested take-profit %


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    name: str = "base"
    best_regimes: list[MarketRegime] = []

    @abstractmethod
    def generate_signal(
        self, df: pd.DataFrame, sentiment: SentimentState | None = None
    ) -> StrategySignal:
        """Generate a BUY/SELL/HOLD signal from indicator-enriched OHLCV data."""


class MomentumStrategy(BaseStrategy):
    """
    Trend Following / Momentum Strategy.
    Best in: TRENDING_UP, TRENDING_DOWN

    Logic:
    - Buy when price > SMA20 > SMA50, RSI 40-70, MACD bullish crossover
    - Sell when price < SMA20 < SMA50, RSI 30-60, MACD bearish crossover
    - Uses ATR for dynamic stops (wider in trends)

    Source: Top quant fund strategy, 70% of algo trading volume uses momentum.
    """
    name = "Momentum"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.rsi_oversold = p.get("rsi_oversold", 30)
        self.rsi_overbought = p.get("rsi_overbought", 70)
        self.macd_threshold = p.get("macd_threshold", 0.0)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Score MA alignment, MACD crossover, and RSI to produce a momentum signal."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # MA alignment
        ma_bullish = latest["close"] > latest["sma_20"] > latest["sma_50"]
        ma_bearish = latest["close"] < latest["sma_20"] < latest["sma_50"]

        # MACD crossover
        macd_cross_up = latest["macd"] > latest["macd_signal"] and prev["macd"] <= prev["macd_signal"]
        macd_cross_down = latest["macd"] < latest["macd_signal"] and prev["macd"] >= prev["macd_signal"]
        macd_bullish = latest["macd"] > latest["macd_signal"]
        macd_bearish = latest["macd"] < latest["macd_signal"]

        # RSI filter (avoid overbought/oversold in momentum) — evolved params
        rsi = latest["rsi"]
        rsi_ok_buy = self.rsi_oversold < rsi < self.rsi_overbought
        rsi_ok_sell = (self.rsi_oversold - 10) < rsi < (self.rsi_overbought - 10)

        # Volume confirmation
        vol_confirm = latest["volume_ratio"] > 1.0

        # Score
        buy_score = sum([
            ma_bullish * 0.3,
            macd_bullish * 0.2,
            macd_cross_up * 0.2,
            rsi_ok_buy * 0.15,
            vol_confirm * 0.15,
        ])

        sell_score = sum([
            ma_bearish * 0.3,
            macd_bearish * 0.2,
            macd_cross_down * 0.2,
            rsi_ok_sell * 0.15,
            vol_confirm * 0.15,
        ])

        atr_pct = latest.get("atr_pct", 0.02)

        if buy_score > 0.5:
            return StrategySignal(
                signal="BUY", confidence=min(buy_score, 0.95),
                strategy_name=self.name, reason="Momentum bullish alignment",
                suggested_sl_pct=atr_pct * 2, suggested_tp_pct=atr_pct * 3,
            )
        elif sell_score > 0.5:
            return StrategySignal(
                signal="SELL", confidence=min(sell_score, 0.95),
                strategy_name=self.name, reason="Momentum bearish alignment",
                suggested_sl_pct=atr_pct * 2, suggested_tp_pct=atr_pct * 3,
            )
        else:
            return StrategySignal(
                signal="HOLD", confidence=0.5,
                strategy_name=self.name, reason="No clear momentum signal",
            )


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.
    Best in: RANGING

    Logic:
    - Buy when price touches lower Bollinger Band + RSI < 30 (oversold)
    - Sell when price touches upper Bollinger Band + RSI > 70 (overbought)
    - Tighter stops since we're fading the move

    Source: Works best in sideways markets. Bear market short-squeeze rallies.
    """
    name = "MeanReversion"
    best_regimes = [MarketRegime.RANGING]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.rsi_low = p.get("rsi_low", 25)
        self.rsi_high = p.get("rsi_high", 75)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Detect oversold/overbought extremes using Bollinger Bands, RSI, and Stochastic."""
        latest = df.iloc[-1]

        bb_pos = latest.get("bb_position", 0.5)
        rsi = latest["rsi"]
        stoch_k = latest.get("stoch_k", 50)

        # Distance from SMA20 (mean)
        close_to_mean = latest.get("close_to_sma20", 0)

        # Oversold: price near lower BB, RSI low, stoch low — evolved params
        oversold_score = sum([
            (bb_pos < 0.1) * 0.3,
            (bb_pos < 0.2) * 0.1,
            (rsi < self.rsi_low) * 0.25,
            (rsi < self.rsi_low + 10) * 0.1,
            (stoch_k < 20) * 0.15,
            (close_to_mean < -0.02) * 0.1,
        ])

        # Overbought: price near upper BB, RSI high, stoch high — evolved params
        overbought_score = sum([
            (bb_pos > 0.9) * 0.3,
            (bb_pos > 0.8) * 0.1,
            (rsi > self.rsi_high) * 0.25,
            (rsi > self.rsi_high - 10) * 0.1,
            (stoch_k > 80) * 0.15,
            (close_to_mean > 0.02) * 0.1,
        ])

        if oversold_score > 0.4:
            return StrategySignal(
                signal="BUY", confidence=min(oversold_score, 0.9),
                strategy_name=self.name,
                reason=f"Oversold (BB:{bb_pos:.2f}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015, suggested_tp_pct=0.025,
            )
        elif overbought_score > 0.4:
            return StrategySignal(
                signal="SELL", confidence=min(overbought_score, 0.9),
                strategy_name=self.name,
                reason=f"Overbought (BB:{bb_pos:.2f}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015, suggested_tp_pct=0.025,
            )
        else:
            return StrategySignal(
                signal="HOLD", confidence=0.5,
                strategy_name=self.name, reason="Price near mean — no reversion signal",
            )


class BreakoutStrategy(BaseStrategy):
    """
    Breakout / Volatility Expansion Strategy.
    Best in: RANGING (about to break out), HIGH_VOLATILITY

    Logic:
    - Buy when price breaks above upper Bollinger Band with volume surge
    - Sell when price breaks below lower Bollinger Band with volume surge
    - Tight stops: if breakout fails, exit fast

    Source: Capitalizes on volatility expansion after compression (squeeze).
    """
    name = "Breakout"
    best_regimes = [MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.lookback = p.get("lookback", 20)
        self.volume_mult = p.get("volume_mult", 1.5)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Identify Bollinger Band breakouts confirmed by volume surge."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        bb_pos = latest.get("bb_position", 0.5)
        bb_width = latest.get("bb_width", 0)
        vol_ratio = latest.get("volume_ratio", 1.0)

        # Bollinger squeeze: width narrowing then expanding
        bb_widths = df["bb_width"].tail(20) if "bb_width" in df.columns else pd.Series([0])
        was_squeezed = bb_widths.iloc[-5:].mean() < bb_widths.mean() * 0.8
        expanding = bb_width > prev.get("bb_width", 0)

        # Price breaking bands with volume — evolved params
        break_up = bb_pos > 1.0 and vol_ratio > self.volume_mult
        break_down = bb_pos < 0.0 and vol_ratio > self.volume_mult

        # New high/low detection — evolved lookback
        recent_high = df["high"].tail(self.lookback).max()
        recent_low = df["low"].tail(self.lookback).min()
        at_high = latest["close"] >= recent_high * 0.998
        at_low = latest["close"] <= recent_low * 1.002

        if (break_up or at_high) and vol_ratio > 1.2:
            conf = 0.5 + (vol_ratio - 1.0) * 0.2 + was_squeezed * 0.15
            return StrategySignal(
                signal="BUY", confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Breakout UP (vol:{vol_ratio:.1f}x, squeeze:{was_squeezed})",
                suggested_sl_pct=0.01, suggested_tp_pct=0.04,
            )
        elif (break_down or at_low) and vol_ratio > 1.2:
            conf = 0.5 + (vol_ratio - 1.0) * 0.2 + was_squeezed * 0.15
            return StrategySignal(
                signal="SELL", confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Breakout DOWN (vol:{vol_ratio:.1f}x, squeeze:{was_squeezed})",
                suggested_sl_pct=0.01, suggested_tp_pct=0.04,
            )
        else:
            return StrategySignal(
                signal="HOLD", confidence=0.5,
                strategy_name=self.name, reason="No breakout detected",
            )


class GridStrategy(BaseStrategy):
    """
    Grid Trading Strategy.
    Best in: RANGING (low volatility sideways)

    Logic:
    - Places virtual buy/sell levels at intervals around current price
    - Buy at grid support levels, sell at grid resistance levels
    - Profits from oscillation within a range

    Source: Pionex, Bitsgap, 3Commas grid bots.
    """
    name = "Grid"
    best_regimes = [MarketRegime.RANGING]

    def __init__(self, grid_spacing_pct: float = 0.01, grid_levels: int = 5, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.grid_spacing = p.get("grid_size_pct", grid_spacing_pct * 100) / 100.0
        self.grid_levels = p.get("num_levels", grid_levels)
        self.confidence_base = p.get("confidence_base", 0.5)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Signal BUY/SELL when price deviates from grid center by threshold levels."""
        latest = df.iloc[-1]
        close = latest["close"]

        # Calculate grid center (SMA20 as the "fair value")
        sma20 = latest.get("sma_20", close)

        # Where is price relative to grid center?
        deviation = (close - sma20) / sma20

        # Which grid level are we at?
        grid_level = deviation / self.grid_spacing

        # Buy near bottom of grid, sell near top
        if grid_level < -1.5:
            conf = min(0.5 + abs(grid_level) * 0.1, 0.85)
            return StrategySignal(
                signal="BUY", confidence=conf,
                strategy_name=self.name,
                reason=f"Grid level {grid_level:.1f} (below center)",
                suggested_sl_pct=self.grid_spacing * 2,
                suggested_tp_pct=self.grid_spacing * 1.5,
            )
        elif grid_level > 1.5:
            conf = min(0.5 + abs(grid_level) * 0.1, 0.85)
            return StrategySignal(
                signal="SELL", confidence=conf,
                strategy_name=self.name,
                reason=f"Grid level {grid_level:.1f} (above center)",
                suggested_sl_pct=self.grid_spacing * 2,
                suggested_tp_pct=self.grid_spacing * 1.5,
            )
        else:
            return StrategySignal(
                signal="HOLD", confidence=0.5,
                strategy_name=self.name,
                reason=f"Grid level {grid_level:.1f} (near center)",
            )


class ScalpingStrategy(BaseStrategy):
    """
    Scalping / Quick Reversal Strategy.
    Works in: Any regime (but best in high volume)

    Logic:
    - Look for RSI extreme + price rejection candle patterns
    - Very tight stops and targets (0.5-1% range)
    - High win-rate, small gains

    Source: CryptoRobotics, Pionex scalping bots.
    """
    name = "Scalping"
    best_regimes = [MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.volume_spike = p.get("volume_spike", 2.0)
        self.confidence_base = p.get("confidence_base", 0.5)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Detect reversal candle patterns (hammer/shooting star) with volume spike."""
        if len(df) < 5:
            return StrategySignal(
                signal="HOLD", confidence=0.0,
                strategy_name=self.name, reason="Not enough data",
            )

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest["rsi"]
        close = latest["close"]
        open_ = latest["open"]
        high = latest["high"]
        low = latest["low"]

        # Candle body and wick analysis
        body = abs(close - open_)
        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low
        total_range = high - low

        if total_range == 0:
            return StrategySignal(
                signal="HOLD", confidence=0.0,
                strategy_name=self.name, reason="No price movement",
            )

        # Pin bar / hammer detection (reversal candle)
        is_hammer = lower_wick > body * 2 and upper_wick < body * 0.5
        is_shooting_star = upper_wick > body * 2 and lower_wick < body * 0.5

        # Quick reversal signals
        rsi_oversold_bounce = rsi < 25 and latest["close"] > latest["open"]
        rsi_overbought_drop = rsi > 75 and latest["close"] < latest["open"]

        # Volume spike — evolved param
        vol_spike = latest.get("volume_ratio", 1.0) > self.volume_spike

        if (is_hammer or rsi_oversold_bounce) and vol_spike:
            return StrategySignal(
                signal="BUY", confidence=0.7,
                strategy_name=self.name,
                reason=f"Scalp reversal (hammer:{is_hammer}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.005, suggested_tp_pct=0.01,
            )
        elif (is_shooting_star or rsi_overbought_drop) and vol_spike:
            return StrategySignal(
                signal="SELL", confidence=0.7,
                strategy_name=self.name,
                reason=f"Scalp reversal (star:{is_shooting_star}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.005, suggested_tp_pct=0.01,
            )
        else:
            return StrategySignal(
                signal="HOLD", confidence=0.3,
                strategy_name=self.name, reason="No scalp setup",
            )


class SentimentDrivenStrategy(BaseStrategy):
    """
    Sentiment-Driven Strategy.
    Works in: Any regime (modifies other signals)

    Logic:
    - At extreme fear: contrarian BUY (Warren Buffett approach)
    - At extreme greed: contrarian SELL
    - Volume-sentiment divergence = early warning
    - Combines Fear/Greed Index with on-chain momentum

    Source: Alternative.me, CoinGecko sentiment research.
    """
    name = "Sentiment"
    best_regimes = [
        MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
        MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY,
    ]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.fear_threshold = p.get("fear_threshold", 25)
        self.greed_threshold = p.get("greed_threshold", 75)
        self.composite_threshold = p.get("composite_threshold", 0.3)
        self.confidence_base = p.get("confidence_base", 0.5)

    def generate_signal(
        self, df: pd.DataFrame, sentiment: SentimentState = None
    ) -> StrategySignal:
        """Generate contrarian signals at Fear & Greed extremes."""
        if sentiment is None:
            return StrategySignal(
                signal="HOLD", confidence=0.0,
                strategy_name=self.name, reason="No sentiment data",
            )

        fg = sentiment.fear_greed_index
        composite = sentiment.composite_score
        contrarian = sentiment.contrarian_signal

        # Strong contrarian signals at extremes — evolved params
        extreme_fear = self.fear_threshold - 10
        extreme_greed = self.greed_threshold + 10
        if fg <= extreme_fear and composite < -self.composite_threshold * 1.5:
            return StrategySignal(
                signal="BUY", confidence=0.8,
                strategy_name=self.name,
                reason=f"Extreme fear ({fg}) — contrarian buy",
                suggested_sl_pct=0.03, suggested_tp_pct=0.08,
            )
        elif fg >= extreme_greed and composite > self.composite_threshold * 1.5:
            return StrategySignal(
                signal="SELL", confidence=0.8,
                strategy_name=self.name,
                reason=f"Extreme greed ({fg}) — contrarian sell",
                suggested_sl_pct=0.03, suggested_tp_pct=0.08,
            )
        # Moderate signals — evolved params
        elif fg <= self.fear_threshold and composite < -self.composite_threshold:
            return StrategySignal(
                signal="BUY", confidence=0.6,
                strategy_name=self.name,
                reason=f"Fear zone ({fg}) — cautious buy",
                suggested_sl_pct=0.02, suggested_tp_pct=0.05,
            )
        elif fg >= self.greed_threshold and composite > self.composite_threshold:
            return StrategySignal(
                signal="SELL", confidence=0.6,
                strategy_name=self.name,
                reason=f"Greed zone ({fg}) — cautious sell",
                suggested_sl_pct=0.02, suggested_tp_pct=0.05,
            )
        else:
            return StrategySignal(
                signal="HOLD", confidence=0.4,
                strategy_name=self.name,
                reason=f"Neutral sentiment ({fg})",
            )


class VWAPStrategy(BaseStrategy):
    """
    VWAP Reversion Strategy.
    Best in: RANGING, TRENDING_UP, TRENDING_DOWN

    Logic:
    - Price far below VWAP with RSI support = buy (institutional discount)
    - Price far above VWAP with RSI resistance = sell (institutional premium)
    - VWAP acts as institutional fair value — big money trades around it

    Source: Institutional order flow research; VWAP is the #1 institutional benchmark.
    """
    name = "VWAP"
    best_regimes = [MarketRegime.RANGING, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.deviation_threshold = p.get("deviation_threshold", 0.01)  # 1%
        self.confidence_base = p.get("confidence_base", 0.55)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Signal when price deviates significantly from VWAP with RSI confirmation."""
        latest = df.iloc[-1]

        vwap_dev = latest.get("close_to_vwap", 0)
        rsi = latest.get("rsi", 50)
        vol_ratio = latest.get("volume_ratio", 1.0)
        adx = latest.get("adx", 25)

        # Strong deviation from VWAP with confirming RSI
        if vwap_dev < -self.deviation_threshold and rsi < 40:
            strength = min(abs(vwap_dev) / 0.03, 1.0)
            conf = 0.5 + strength * 0.3 + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="BUY", confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Below VWAP ({vwap_dev:.2%}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015, suggested_tp_pct=abs(vwap_dev) * 0.8,
            )
        elif vwap_dev > self.deviation_threshold and rsi > 60:
            strength = min(abs(vwap_dev) / 0.03, 1.0)
            conf = 0.5 + strength * 0.3 + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="SELL", confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Above VWAP ({vwap_dev:+.2%}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015, suggested_tp_pct=abs(vwap_dev) * 0.8,
            )
        return StrategySignal(
            signal="HOLD", confidence=0.4,
            strategy_name=self.name, reason=f"Near VWAP ({vwap_dev:+.2%})",
        )


class OBVDivergenceStrategy(BaseStrategy):
    """
    OBV Divergence Strategy.
    Best in: TRENDING_UP, TRENDING_DOWN (catches reversals)

    Logic:
    - Bullish divergence: price making lower lows but OBV making higher lows
      → Smart money accumulating, reversal likely
    - Bearish divergence: price making higher highs but OBV making lower highs
      → Smart money distributing, reversal likely

    Source: Joe Granville's OBV theory; divergence is a leading indicator.
    """
    name = "OBVDivergence"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.RANGING]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.confidence_base = p.get("confidence_base", 0.55)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Detect bullish or bearish OBV divergence as a leading reversal signal."""
        latest = df.iloc[-1]

        obv_div = latest.get("obv_divergence", 0)
        rsi = latest.get("rsi", 50)
        vol_ratio = latest.get("volume_ratio", 1.0)
        adx = latest.get("adx", 25)

        # Bullish divergence — price down but OBV up (accumulation)
        if obv_div == 1:
            # Stronger signal if RSI is oversold and trend is weakening
            conf = 0.55 + (rsi < 35) * 0.15 + (vol_ratio > 1.0) * 0.1 + (adx < 25) * 0.1
            return StrategySignal(
                signal="BUY", confidence=min(conf, 0.85),
                strategy_name=self.name,
                reason=f"Bullish OBV divergence (RSI:{rsi:.0f}, ADX:{adx:.0f})",
                suggested_sl_pct=0.02, suggested_tp_pct=0.04,
            )

        # Bearish divergence — price up but OBV down (distribution)
        elif obv_div == -1:
            conf = 0.55 + (rsi > 65) * 0.15 + (vol_ratio > 1.0) * 0.1 + (adx < 25) * 0.1
            return StrategySignal(
                signal="SELL", confidence=min(conf, 0.85),
                strategy_name=self.name,
                reason=f"Bearish OBV divergence (RSI:{rsi:.0f}, ADX:{adx:.0f})",
                suggested_sl_pct=0.02, suggested_tp_pct=0.04,
            )

        return StrategySignal(
            signal="HOLD", confidence=0.3,
            strategy_name=self.name, reason="No OBV divergence",
        )


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA 9/21 Crossover with ADX Filter.
    Best in: TRENDING_UP, TRENDING_DOWN

    Logic:
    - Buy on 9 EMA crossing above 21 EMA (golden cross) when ADX > 20
    - Sell on 9 EMA crossing below 21 EMA (death cross) when ADX > 20
    - ADX filter avoids whipsaws in choppy markets

    Source: Classic crossover enhanced with trend strength filter.
    """
    name = "EMACrossover"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        p = params or {}
        self.adx_threshold = p.get("adx_threshold", 20)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Signal on EMA 9/21 crossover events filtered by ADX trend strength."""
        latest = df.iloc[-1]

        ema_cross = latest.get("ema_cross", 0)
        adx = latest.get("adx", 25)
        rsi = latest.get("rsi", 50)
        vol_ratio = latest.get("volume_ratio", 1.0)

        # Need trend strength for crossover to be meaningful
        trend_strong = adx > self.adx_threshold

        if ema_cross == 1 and trend_strong:
            conf = 0.55 + min((adx - 20) / 40, 0.2) + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="BUY", confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"EMA 9/21 golden cross (ADX:{adx:.0f}, vol:{vol_ratio:.1f}x)",
                suggested_sl_pct=0.015, suggested_tp_pct=0.035,
            )
        elif ema_cross == -1 and trend_strong:
            conf = 0.55 + min((adx - 20) / 40, 0.2) + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="SELL", confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"EMA 9/21 death cross (ADX:{adx:.0f}, vol:{vol_ratio:.1f}x)",
                suggested_sl_pct=0.015, suggested_tp_pct=0.035,
            )

        # No crossover this bar, but check trend alignment
        ema_9 = latest.get("ema_9", 0)
        ema_21 = latest.get("ema_21", 0)
        if ema_9 and ema_21 and trend_strong:
            if ema_9 > ema_21 * 1.002 and rsi > 50:
                return StrategySignal(
                    signal="BUY", confidence=0.45,
                    strategy_name=self.name,
                    reason=f"EMA bullish alignment (ADX:{adx:.0f})",
                    suggested_sl_pct=0.015, suggested_tp_pct=0.03,
                )
            elif ema_9 < ema_21 * 0.998 and rsi < 50:
                return StrategySignal(
                    signal="SELL", confidence=0.45,
                    strategy_name=self.name,
                    reason=f"EMA bearish alignment (ADX:{adx:.0f})",
                    suggested_sl_pct=0.015, suggested_tp_pct=0.03,
                )

        return StrategySignal(
            signal="HOLD", confidence=0.3,
            strategy_name=self.name,
            reason=f"No EMA cross (ADX:{adx:.0f})",
        )


# ────────────────────────────────────────────────────────────
#  STRATEGY ENGINE — Adaptive strategy selection
# ────────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Adaptive Strategy Engine — OPTIMIZED v3.
    Only runs strategies relevant to the current regime (short-circuit).
    Caches strategy instances, pre-filters by regime to avoid unnecessary work.
    """

    REGIME_STRATEGY_MAP = {
        MarketRegime.TRENDING_UP: {
            "primary": "Momentum",
            "secondary": ["EMACrossover", "Sentiment", "VWAP"],
            "weights": {"Momentum": 0.35, "EMACrossover": 0.25, "Sentiment": 0.2, "VWAP": 0.2},
        },
        MarketRegime.TRENDING_DOWN: {
            "primary": "Momentum",
            "secondary": ["EMACrossover", "OBVDivergence", "Sentiment"],
            "weights": {"Momentum": 0.35, "EMACrossover": 0.2, "OBVDivergence": 0.25, "Sentiment": 0.2},
        },
        MarketRegime.RANGING: {
            "primary": "MeanReversion",
            "secondary": ["VWAP", "Grid", "OBVDivergence"],
            "weights": {"MeanReversion": 0.35, "VWAP": 0.25, "Grid": 0.2, "OBVDivergence": 0.2},
        },
        MarketRegime.HIGH_VOLATILITY: {
            "primary": "Breakout",
            "secondary": ["Scalping", "EMACrossover", "Sentiment"],
            "weights": {"Breakout": 0.35, "Scalping": 0.25, "EMACrossover": 0.2, "Sentiment": 0.2},
        },
    }

    def __init__(self, evolved_params: dict[str, dict[str, float]] | None = None) -> None:
        """
        Initialize strategies with optional evolved parameters.
        evolved_params: dict mapping strategy name -> param dict from StrategyEvolver.
        """
        ep = evolved_params or {}
        self.strategies = {
            "Momentum": MomentumStrategy(params=ep.get("Momentum")),
            "MeanReversion": MeanReversionStrategy(params=ep.get("MeanReversion")),
            "Breakout": BreakoutStrategy(params=ep.get("Breakout")),
            "Grid": GridStrategy(params=ep.get("Grid")),
            "Scalping": ScalpingStrategy(params=ep.get("Scalping")),
            "Sentiment": SentimentDrivenStrategy(params=ep.get("Sentiment")),
            # v7.0: New strategies
            "VWAP": VWAPStrategy(params=ep.get("VWAP")),
            "OBVDivergence": OBVDivergenceStrategy(params=ep.get("OBVDivergence")),
            "EMACrossover": EMACrossoverStrategy(params=ep.get("EMACrossover")),
        }
        self.last_signals = {}

    # Map strategy names to classes for hot-reload
    _STRATEGY_CLASSES = {
        "Momentum": MomentumStrategy,
        "MeanReversion": MeanReversionStrategy,
        "Breakout": BreakoutStrategy,
        "Grid": GridStrategy,
        "Scalping": ScalpingStrategy,
        "Sentiment": SentimentDrivenStrategy,
        "VWAP": VWAPStrategy,
        "OBVDivergence": OBVDivergenceStrategy,
        "EMACrossover": EMACrossoverStrategy,
    }

    def apply_evolved_params(self, evolved_params: dict[str, dict[str, float]]) -> None:
        """Hot-reload evolved parameters into strategies without restart."""
        for name, params in evolved_params.items():
            if name in self.strategies and params:
                cls = self._STRATEGY_CLASSES.get(name)
                if cls:
                    self.strategies[name] = cls(params=params)

    def run(
        self, df: pd.DataFrame, regime: MarketRegime,
        sentiment: SentimentState | None = None,
    ) -> StrategySignal:
        """
        OPTIMIZED: Run primary first; if high confidence, short-circuit
        and skip secondary strategies to save compute.
        """
        config = self.REGIME_STRATEGY_MAP.get(regime, self.REGIME_STRATEGY_MAP[MarketRegime.RANGING])
        weights = config["weights"]
        signals = {}

        # Run primary strategy first
        primary_name = config["primary"]
        primary_strat = self.strategies[primary_name]
        primary_sig = primary_strat.generate_signal(df, sentiment)
        signals[primary_name] = primary_sig

        # SHORT-CIRCUIT: if primary has strong signal (>0.8), skip secondaries
        if primary_sig.signal != "HOLD" and primary_sig.confidence > 0.8:
            self.last_signals = signals
            return StrategySignal(
                signal=primary_sig.signal,
                confidence=min(primary_sig.confidence * weights[primary_name] / 0.5 + 0.1, 0.95),
                strategy_name=primary_name,
                reason=f"Regime:{regime.value} | {primary_sig.reason} (strong)",
                suggested_sl_pct=primary_sig.suggested_sl_pct,
                suggested_tp_pct=primary_sig.suggested_tp_pct,
            )

        # Run secondary strategies
        for strat_name in config["secondary"]:
            strategy = self.strategies.get(strat_name)
            if strategy:
                signals[strat_name] = strategy.generate_signal(df, sentiment)

        self.last_signals = signals

        # Weighted voting (inlined for speed — no intermediate lists)
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        best_buy_sig = None
        best_buy_weight = 0.0
        best_sell_sig = None
        best_sell_weight = 0.0

        for strat_name, sig in signals.items():
            w = weights.get(strat_name, 0.1)
            weighted = w * sig.confidence
            if sig.signal == "BUY":
                buy_score += weighted
                if weighted > best_buy_weight:
                    best_buy_weight = weighted
                    best_buy_sig = sig
            elif sig.signal == "SELL":
                sell_score += weighted
                if weighted > best_sell_weight:
                    best_sell_weight = weighted
                    best_sell_sig = sig
            else:
                hold_score += weighted

        total = buy_score + sell_score + hold_score
        if total == 0:
            return StrategySignal(
                signal="HOLD", confidence=0.5,
                strategy_name="Ensemble", reason="No signals",
            )

        if buy_score > sell_score and buy_score > hold_score and best_buy_sig:
            return StrategySignal(
                signal="BUY",
                confidence=min(buy_score / total + 0.1, 0.95),
                strategy_name=f"Ensemble({', '.join(n for n, s in signals.items() if s.signal == 'BUY')})",
                reason=f"Regime:{regime.value} | {best_buy_sig.reason}",
                suggested_sl_pct=best_buy_sig.suggested_sl_pct,
                suggested_tp_pct=best_buy_sig.suggested_tp_pct,
            )
        elif sell_score > buy_score and sell_score > hold_score and best_sell_sig:
            return StrategySignal(
                signal="SELL",
                confidence=min(sell_score / total + 0.1, 0.95),
                strategy_name=f"Ensemble({', '.join(n for n, s in signals.items() if s.signal == 'SELL')})",
                reason=f"Regime:{regime.value} | {best_sell_sig.reason}",
                suggested_sl_pct=best_sell_sig.suggested_sl_pct,
                suggested_tp_pct=best_sell_sig.suggested_tp_pct,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=hold_score / total,
                strategy_name="Ensemble",
                reason=f"Regime:{regime.value} | No consensus",
            )
