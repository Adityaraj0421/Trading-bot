"""
Strategy Library — 11 strategies that activate based on market regime.

Each strategy takes OHLCV data with indicators and returns a StrategySignal with
action (BUY/SELL/HOLD) and confidence (0.0–1.0).  The StrategyEngine selects which
strategies to run based on the detected regime and combines their outputs via
weighted voting.

Strategies:
  1. MomentumStrategy       — MA + MACD + RSI trend following (TRENDING_UP/DOWN)
  2. MeanReversionStrategy  — Bollinger Band + RSI oversold/overbought fades (RANGING)
  3. BreakoutStrategy       — Volatility expansion / BB breakout (RANGING, HIGH_VOLATILITY)
  4. GridStrategy           — Price oscillation within SMA20-anchored grid (RANGING)
  5. ScalpingStrategy       — Hammer/shooting-star reversal candles (any regime)
  6. SentimentDrivenStrategy — Contrarian + momentum on Fear/Greed extremes (all regimes)
  7. VWAPStrategy           — VWAP deviation reversion (RANGING, TRENDING_UP/DOWN)
  8. OBVDivergenceStrategy  — Price/OBV divergence leading reversals (TRENDING_UP/DOWN, RANGING)
  9. EMACrossoverStrategy   — EMA 9/21 golden/death cross with ADX filter (TRENDING_UP/DOWN)
 10. IchimokuStrategy       — TK-cross + cloud position + ADX filter (TRENDING_UP/DOWN)
 11. RSIDivergenceStrategy  — RSI divergence + Stochastic crossover (RANGING, TRENDING_UP/DOWN)

Inspired by: QuantInsti, Freqtrade, 3Commas, Renaissance Technologies research
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from regime_detector import MarketRegime
from sentiment import SentimentState

_log = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Trading signal produced by a strategy with confidence and risk parameters."""

    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    reason: str
    suggested_sl_pct: float = 0.02  # Suggested stop-loss %
    suggested_tp_pct: float = 0.05  # Suggested take-profit %


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must set the class-level ``name`` and ``best_regimes`` attributes and
    implement ``generate_signal``.
    """

    name: str = "base"
    best_regimes: list[MarketRegime] = []

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Generate a BUY/SELL/HOLD signal from indicator-enriched OHLCV data.

        Args:
            df: DataFrame with OHLCV columns plus computed indicator columns.
            sentiment: Optional current market sentiment state.

        Returns:
            A StrategySignal with signal, confidence, reason, and SL/TP hints.
        """


class MomentumStrategy(BaseStrategy):
    """Trend-following / momentum strategy for trending markets.

    Generates BUY signals when price > SMA20 > SMA50, MACD is bullish, RSI is in
    the healthy trend range (rsi_oversold–rsi_overbought, default 30–70), and volume confirms. Generates SELL signals on
    the mirror-image bearish alignment. Stop-loss and take-profit are ATR-adaptive
    (2× ATR and 3× ATR respectively).

    Best regimes: TRENDING_UP, TRENDING_DOWN.
    """

    name = "Momentum"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``rsi_oversold`` (default 30), ``rsi_overbought`` (default 70),
                ``macd_threshold`` (default 0.0), ``confidence_base`` (default 0.6).
        """
        p = params or {}
        self.rsi_oversold = p.get("rsi_oversold", 30)
        self.rsi_overbought = p.get("rsi_overbought", 70)
        self.macd_threshold = p.get("macd_threshold", 0.0)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Score MA alignment, MACD crossover, and RSI to produce a momentum signal.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns: ``close``,
                ``sma_20``, ``sma_50``, ``macd``, ``macd_signal``, ``rsi``,
                ``volume_ratio``, ``atr_pct``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. HOLD confidence is 0.5.
            BUY/SELL confidence is capped at 0.95 and scaled by component scores.
        """
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
        buy_score = sum(
            [
                ma_bullish * 0.3,
                macd_bullish * 0.2,
                macd_cross_up * 0.2,
                rsi_ok_buy * 0.15,
                vol_confirm * 0.15,
            ]
        )

        sell_score = sum(
            [
                ma_bearish * 0.3,
                macd_bearish * 0.2,
                macd_cross_down * 0.2,
                rsi_ok_sell * 0.15,
                vol_confirm * 0.15,
            ]
        )

        atr_pct = latest.get("atr_pct", 0.02)

        if buy_score > 0.5:
            return StrategySignal(
                signal="BUY",
                confidence=min(buy_score, 0.95),
                strategy_name=self.name,
                reason="Momentum bullish alignment",
                suggested_sl_pct=atr_pct * 2,
                suggested_tp_pct=atr_pct * 3,
            )
        elif sell_score > 0.5:
            return StrategySignal(
                signal="SELL",
                confidence=min(sell_score, 0.95),
                strategy_name=self.name,
                reason="Momentum bearish alignment",
                suggested_sl_pct=atr_pct * 2,
                suggested_tp_pct=atr_pct * 3,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=0.5,
                strategy_name=self.name,
                reason="No clear momentum signal",
            )


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy for ranging, low-trend markets.

    Generates BUY signals when price touches the lower Bollinger Band (bb_position <
    0.1–0.2), RSI is oversold, and Stochastic %K is depressed. Generates SELL signals
    on the mirror-image overbought conditions. Uses tighter fixed stops (1.5%/2.5%)
    since the trade fades the current move.

    Best regimes: RANGING.
    """

    name = "MeanReversion"
    best_regimes = [MarketRegime.RANGING]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``rsi_low`` (default 25), ``rsi_high`` (default 75),
                ``confidence_base`` (default 0.6).
        """
        p = params or {}
        self.rsi_low = p.get("rsi_low", 25)
        self.rsi_high = p.get("rsi_high", 75)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Detect oversold/overbought extremes using Bollinger Bands, RSI, and Stochastic.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns: ``bb_position``,
                ``rsi``, ``stoch_k``, ``close_to_sma20``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. HOLD confidence is 0.5.
            BUY/SELL confidence is capped at 0.9, derived from component scores.
        """
        latest = df.iloc[-1]

        bb_pos = latest.get("bb_position", 0.5)
        rsi = latest["rsi"]
        stoch_k = latest.get("stoch_k", 50)

        # Distance from SMA20 (mean)
        close_to_mean = latest.get("close_to_sma20", 0)

        # Oversold: price near lower BB, RSI low, stoch low — evolved params
        oversold_score = sum(
            [
                (bb_pos < 0.1) * 0.3,
                (bb_pos < 0.2) * 0.1,
                (rsi < self.rsi_low) * 0.25,
                (rsi < self.rsi_low + 10) * 0.1,
                (stoch_k < 20) * 0.15,
                (close_to_mean < -0.02) * 0.1,
            ]
        )

        # Overbought: price near upper BB, RSI high, stoch high — evolved params
        overbought_score = sum(
            [
                (bb_pos > 0.9) * 0.3,
                (bb_pos > 0.8) * 0.1,
                (rsi > self.rsi_high) * 0.25,
                (rsi > self.rsi_high - 10) * 0.1,
                (stoch_k > 80) * 0.15,
                (close_to_mean > 0.02) * 0.1,
            ]
        )

        if oversold_score > 0.4:
            return StrategySignal(
                signal="BUY",
                confidence=min(oversold_score, 0.9),
                strategy_name=self.name,
                reason=f"Oversold (BB:{bb_pos:.2f}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015,
                suggested_tp_pct=0.025,
            )
        elif overbought_score > 0.4:
            return StrategySignal(
                signal="SELL",
                confidence=min(overbought_score, 0.9),
                strategy_name=self.name,
                reason=f"Overbought (BB:{bb_pos:.2f}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015,
                suggested_tp_pct=0.025,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=0.5,
                strategy_name=self.name,
                reason="Price near mean — no reversion signal",
            )


class BreakoutStrategy(BaseStrategy):
    """Volatility-expansion / breakout strategy.

    Generates BUY signals when price breaks above the upper Bollinger Band (bb_position
    > 1.0) or hits a new N-bar high with volume surge. Generates SELL signals on
    mirror-image downside breaks. A prior Bollinger squeeze increases confidence.
    Tight stops (1%) allow fast exit if the breakout fails.

    Best regimes: RANGING (about to break out), HIGH_VOLATILITY.
    """

    name = "Breakout"
    best_regimes = [MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``lookback`` (default 20), ``volume_mult`` (default 1.5),
                ``confidence_base`` (default 0.6), ``atr_mult`` (default 0.8).
        """
        p = params or {}
        self.lookback = p.get("lookback", 20)
        self.volume_mult = p.get("volume_mult", 1.5)
        self.confidence_base = p.get("confidence_base", 0.6)
        self.atr_mult = p.get("atr_mult", 0.8)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Identify Bollinger Band breakouts confirmed by volume surge and ATR expansion.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns: ``bb_position``,
                ``volume_ratio``, ``bb_width``, ``high``, ``low``, ``close``, ``atr``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. HOLD confidence is 0.3.
            BUY/SELL confidence scales with volume ratio and squeeze detection.

        Note:
            Three confirmation filters required for a signal:
            1. Price break (BB band or N-bar high/low exceeded, not merely touched).
            2. Volume surge >= ``volume_mult`` × average (same threshold for both paths).
            3. Bar range >= ``atr_mult`` × ATR — rejects wicks and noise spikes.
        """
        latest = df.iloc[-1]

        bb_pos = latest.get("bb_position", 0.5)
        vol_ratio = latest.get("volume_ratio", 1.0)
        atr = latest.get("atr", 0.0)

        # Bollinger squeeze: width narrowing then expanding
        bb_widths = df["bb_width"].tail(20) if "bb_width" in df.columns else pd.Series([0])
        was_squeezed = bb_widths.iloc[-5:].mean() < bb_widths.mean() * 0.8

        # ATR expansion guard — bar range must show real expansion, not a noise wick
        bar_range = latest["high"] - latest["low"]
        atr_confirmed = atr <= 0 or bar_range >= atr * self.atr_mult

        # Price breaking bands with volume — evolved params
        break_up = bb_pos > 1.0 and vol_ratio > self.volume_mult
        break_down = bb_pos < 0.0 and vol_ratio > self.volume_mult

        # New high/low detection — price must *exceed* the lookback extreme (not just touch it)
        recent_high = df["high"].tail(self.lookback).max()
        recent_low = df["low"].tail(self.lookback).min()
        at_high = latest["close"] > recent_high  # strict: must exceed prior high
        at_low = latest["close"] < recent_low   # strict: must undercut prior low

        if (break_up or at_high) and vol_ratio >= self.volume_mult and atr_confirmed:
            conf = 0.5 + (vol_ratio - 1.0) * 0.2 + was_squeezed * 0.15
            return StrategySignal(
                signal="BUY",
                confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Breakout UP (vol:{vol_ratio:.1f}x, atr_ok:{atr_confirmed}, squeeze:{was_squeezed})",
                suggested_sl_pct=0.01,
                suggested_tp_pct=0.04,
            )
        elif (break_down or at_low) and vol_ratio >= self.volume_mult and atr_confirmed:
            conf = 0.5 + (vol_ratio - 1.0) * 0.2 + was_squeezed * 0.15
            return StrategySignal(
                signal="SELL",
                confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Breakout DOWN (vol:{vol_ratio:.1f}x, atr_ok:{atr_confirmed}, squeeze:{was_squeezed})",
                suggested_sl_pct=0.01,
                suggested_tp_pct=0.04,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=0.3,
                strategy_name=self.name,
                reason="No confirmed breakout",
            )


class GridStrategy(BaseStrategy):
    """Virtual grid trading strategy for low-volatility sideways markets.

    Uses SMA20 as the grid centre. Generates BUY signals when price falls more than
    1.5 grid levels below centre and SELL signals when it rises more than 1.5 grid
    levels above centre.  Profits from oscillation within the range; stop-loss and
    take-profit are multiples of the grid spacing.

    Best regimes: RANGING.
    """

    name = "Grid"
    best_regimes = [MarketRegime.RANGING]

    def __init__(
        self, grid_spacing_pct: float = 0.01, grid_levels: int = 5, params: dict[str, float] | None = None
    ) -> None:
        """Initialise with optional evolved parameters.

        Args:
            grid_spacing_pct: Default grid spacing as a fraction of price (e.g. 0.01
                for 1%). Overridden by ``params["grid_size_pct"]`` if present.
            grid_levels: Number of grid levels (unused in signal logic, stored for
                reference). Overridden by ``params["num_levels"]`` if present.
            params: Dict of evolved hyperparameters. Recognised keys:
                ``grid_size_pct`` (percentage, default ``grid_spacing_pct * 100``),
                ``num_levels`` (default ``grid_levels``),
                ``confidence_base`` (default 0.5).
        """
        p = params or {}
        self.grid_spacing = p.get("grid_size_pct", grid_spacing_pct * 100) / 100.0
        self.grid_levels = p.get("num_levels", grid_levels)
        self.confidence_base = p.get("confidence_base", 0.5)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Signal BUY/SELL when price deviates from grid center by threshold levels.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns: ``close``,
                ``sma_20``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. HOLD confidence is 0.5.
            BUY/SELL confidence grows with distance from grid centre, capped at 0.85.
        """
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
                signal="BUY",
                confidence=conf,
                strategy_name=self.name,
                reason=f"Grid level {grid_level:.1f} (below center)",
                suggested_sl_pct=self.grid_spacing * 2,
                suggested_tp_pct=self.grid_spacing * 1.5,
            )
        elif grid_level > 1.5:
            conf = min(0.5 + abs(grid_level) * 0.1, 0.85)
            return StrategySignal(
                signal="SELL",
                confidence=conf,
                strategy_name=self.name,
                reason=f"Grid level {grid_level:.1f} (above center)",
                suggested_sl_pct=self.grid_spacing * 2,
                suggested_tp_pct=self.grid_spacing * 1.5,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=0.5,
                strategy_name=self.name,
                reason=f"Grid level {grid_level:.1f} (near center)",
            )


class ScalpingStrategy(BaseStrategy):
    """High-frequency scalping strategy based on reversal candle patterns.

    Generates BUY signals on hammer candles (long lower wick, short body) or RSI
    oversold bounce with a volume spike. Generates SELL signals on shooting-star
    candles or RSI overbought drops with a volume spike. Targets are tight (0.5–1%)
    for high win-rate, small-profit trades.

    Returns ``confidence=0.0`` (data-guard HOLD) when fewer than 5 bars are present
    or when the candle has zero range.

    Works in: Any regime (best in high-volume conditions).
    """

    name = "Scalping"
    best_regimes = [MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``volume_spike`` (default 2.0), ``confidence_base`` (default 0.5).
        """
        p = params or {}
        self.volume_spike = p.get("volume_spike", 2.0)
        self.confidence_base = p.get("confidence_base", 0.5)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Detect reversal candle patterns (hammer/shooting star) with volume spike.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns: ``rsi``,
                ``close``, ``open``, ``high``, ``low``, ``volume_ratio``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. BUY/SELL confidence is fixed at 0.7.
            HOLD confidence is 0.3 (normal) or 0.0 (data-guard: < 5 bars or zero range).
        """
        if len(df) < 5:
            return StrategySignal(
                signal="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason="Not enough data",
            )

        latest = df.iloc[-1]

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
                signal="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason="No price movement",
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
                signal="BUY",
                confidence=0.7,
                strategy_name=self.name,
                reason=f"Scalp reversal (hammer:{is_hammer}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.005,
                suggested_tp_pct=0.01,
            )
        elif (is_shooting_star or rsi_overbought_drop) and vol_spike:
            return StrategySignal(
                signal="SELL",
                confidence=0.7,
                strategy_name=self.name,
                reason=f"Scalp reversal (star:{is_shooting_star}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.005,
                suggested_tp_pct=0.01,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=0.3,
                strategy_name=self.name,
                reason="No scalp setup",
            )


class SentimentDrivenStrategy(BaseStrategy):
    """Contrarian sentiment strategy based on the Fear & Greed Index.

    Generates BUY signals at extreme fear (fg <= fear_threshold - 10) when composite
    sentiment is strongly negative, and SELL signals at extreme greed (fg >=
    greed_threshold + 10) when composite sentiment is strongly positive. Moderate
    fear/greed zones produce lower-confidence signals. Returns HOLD with confidence
    0.0 when no sentiment data is available.

    Works in: All regimes.
    """

    name = "Sentiment"
    best_regimes = [
        MarketRegime.TRENDING_UP,
        MarketRegime.TRENDING_DOWN,
        MarketRegime.RANGING,
        MarketRegime.HIGH_VOLATILITY,
    ]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``fear_threshold`` (default 25), ``greed_threshold`` (default 75),
                ``composite_threshold`` (default 0.3), ``confidence_base`` (default 0.5).
        """
        p = params or {}
        self.fear_threshold = p.get("fear_threshold", 25)
        self.greed_threshold = p.get("greed_threshold", 75)
        self.composite_threshold = p.get("composite_threshold", 0.3)
        self.confidence_base = p.get("confidence_base", 0.5)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Generate contrarian signals at Fear & Greed extremes.

        Args:
            df: Indicator-enriched OHLCV DataFrame (not directly used; required by
                the base class contract).
            sentiment: Current market sentiment. Returns HOLD (confidence=0.0) if
                ``None``.

        Returns:
            StrategySignal with BUY/SELL/HOLD. Extreme signals have confidence 0.8;
            moderate zone signals have confidence 0.6; neutral HOLD has confidence 0.4.
        """
        if sentiment is None:
            return StrategySignal(
                signal="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason="No sentiment data",
            )

        fg = sentiment.fear_greed_index
        composite = sentiment.composite_score

        # Strong contrarian signals at extremes — evolved params
        extreme_fear = self.fear_threshold - 10
        extreme_greed = self.greed_threshold + 10
        if fg <= extreme_fear and composite < -self.composite_threshold * 1.5:
            return StrategySignal(
                signal="BUY",
                confidence=0.8,
                strategy_name=self.name,
                reason=f"Extreme fear ({fg}) — contrarian buy",
                suggested_sl_pct=0.03,
                suggested_tp_pct=0.08,
            )
        elif fg >= extreme_greed and composite > self.composite_threshold * 1.5:
            return StrategySignal(
                signal="SELL",
                confidence=0.8,
                strategy_name=self.name,
                reason=f"Extreme greed ({fg}) — contrarian sell",
                suggested_sl_pct=0.03,
                suggested_tp_pct=0.08,
            )
        # Moderate signals — evolved params
        elif fg <= self.fear_threshold and composite < -self.composite_threshold:
            return StrategySignal(
                signal="BUY",
                confidence=0.6,
                strategy_name=self.name,
                reason=f"Fear zone ({fg}) — cautious buy",
                suggested_sl_pct=0.02,
                suggested_tp_pct=0.05,
            )
        elif fg >= self.greed_threshold and composite > self.composite_threshold:
            return StrategySignal(
                signal="SELL",
                confidence=0.6,
                strategy_name=self.name,
                reason=f"Greed zone ({fg}) — cautious sell",
                suggested_sl_pct=0.02,
                suggested_tp_pct=0.05,
            )
        else:
            return StrategySignal(
                signal="HOLD",
                confidence=0.4,
                strategy_name=self.name,
                reason=f"Neutral sentiment ({fg})",
            )


class VWAPStrategy(BaseStrategy):
    """VWAP deviation reversion strategy.

    Generates BUY signals when price is below VWAP by more than ``deviation_threshold``
    and RSI < 40 (institutional discount zone). Generates SELL signals when price is
    above VWAP by more than the threshold and RSI > 60 (institutional premium zone).
    Volume confirmation increases confidence.

    Best regimes: RANGING, TRENDING_UP, TRENDING_DOWN.
    """

    name = "VWAP"
    best_regimes = [MarketRegime.RANGING, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``deviation_threshold`` (default 0.01 = 1%),
                ``confidence_base`` (default 0.55).
        """
        p = params or {}
        self.deviation_threshold = p.get("deviation_threshold", 0.01)  # 1%
        self.confidence_base = p.get("confidence_base", 0.55)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Signal when price deviates significantly from VWAP with RSI confirmation.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns:
                ``close_to_vwap``, ``rsi``, ``volume_ratio``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. HOLD confidence is 0.4.
            BUY/SELL confidence scales with VWAP deviation magnitude, capped at 0.9.
        """
        latest = df.iloc[-1]

        vwap_dev = latest.get("close_to_vwap", 0)
        rsi = latest.get("rsi", 50)
        vol_ratio = latest.get("volume_ratio", 1.0)

        # Strong deviation from VWAP with confirming RSI
        if vwap_dev < -self.deviation_threshold and rsi < 40:
            strength = min(abs(vwap_dev) / 0.03, 1.0)
            conf = 0.5 + strength * 0.3 + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="BUY",
                confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Below VWAP ({vwap_dev:.2%}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015,
                suggested_tp_pct=abs(vwap_dev) * 0.8,
            )
        elif vwap_dev > self.deviation_threshold and rsi > 60:
            strength = min(abs(vwap_dev) / 0.03, 1.0)
            conf = 0.5 + strength * 0.3 + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="SELL",
                confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"Above VWAP ({vwap_dev:+.2%}, RSI:{rsi:.0f})",
                suggested_sl_pct=0.015,
                suggested_tp_pct=abs(vwap_dev) * 0.8,
            )
        return StrategySignal(
            signal="HOLD",
            confidence=0.4,
            strategy_name=self.name,
            reason=f"Near VWAP ({vwap_dev:+.2%})",
        )


class OBVDivergenceStrategy(BaseStrategy):
    """OBV divergence strategy for detecting smart-money reversals.

    Generates BUY signals on bullish divergence (price making lower lows while OBV
    makes higher lows — accumulation), and SELL signals on bearish divergence (price
    making higher highs while OBV makes lower highs — distribution). Signal strength
    is enhanced when RSI is oversold/overbought and ADX is weakening.

    Best regimes: TRENDING_UP, TRENDING_DOWN (catching reversals), RANGING.
    """

    name = "OBVDivergence"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.RANGING]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``confidence_base`` (default 0.55).
        """
        p = params or {}
        self.confidence_base = p.get("confidence_base", 0.55)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Detect bullish or bearish OBV divergence as a leading reversal signal.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns:
                ``obv_divergence`` (1 = bullish, -1 = bearish, 0 = none),
                ``rsi``, ``volume_ratio``, ``adx``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. HOLD confidence is 0.3.
            BUY/SELL base confidence is 0.55, boosted by RSI and ADX conditions,
            capped at 0.85.
        """
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
                signal="BUY",
                confidence=min(conf, 0.85),
                strategy_name=self.name,
                reason=f"Bullish OBV divergence (RSI:{rsi:.0f}, ADX:{adx:.0f})",
                suggested_sl_pct=0.02,
                suggested_tp_pct=0.04,
            )

        # Bearish divergence — price up but OBV down (distribution)
        elif obv_div == -1:
            conf = 0.55 + (rsi > 65) * 0.15 + (vol_ratio > 1.0) * 0.1 + (adx < 25) * 0.1
            return StrategySignal(
                signal="SELL",
                confidence=min(conf, 0.85),
                strategy_name=self.name,
                reason=f"Bearish OBV divergence (RSI:{rsi:.0f}, ADX:{adx:.0f})",
                suggested_sl_pct=0.02,
                suggested_tp_pct=0.04,
            )

        return StrategySignal(
            signal="HOLD",
            confidence=0.3,
            strategy_name=self.name,
            reason="No OBV divergence",
        )


class EMACrossoverStrategy(BaseStrategy):
    """EMA 9/21 crossover strategy with ADX trend-strength filter.

    Generates BUY signals when EMA 9 crosses above EMA 21 (golden cross) and ADX >
    threshold, and SELL signals when EMA 9 crosses below EMA 21 (death cross) with
    the same ADX filter. When no fresh cross is detected but trend is strong, a
    lower-confidence directional alignment signal is returned.

    Best regimes: TRENDING_UP, TRENDING_DOWN.
    """

    name = "EMACrossover"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``adx_threshold`` (default 20), ``confidence_base`` (default 0.6).
        """
        p = params or {}
        self.adx_threshold = p.get("adx_threshold", 20)
        self.confidence_base = p.get("confidence_base", 0.6)

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Signal on EMA 9/21 crossover events filtered by ADX trend strength.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns:
                ``ema_cross`` (1 = golden cross, -1 = death cross, 0 = none),
                ``adx``, ``rsi``, ``volume_ratio``, ``ema_9``, ``ema_21``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. Cross-event confidence ranges 0.55–0.85;
            alignment-only confidence is fixed at 0.45. HOLD confidence is 0.3.
        """
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
                signal="BUY",
                confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"EMA 9/21 golden cross (ADX:{adx:.0f}, vol:{vol_ratio:.1f}x)",
                suggested_sl_pct=0.015,
                suggested_tp_pct=0.035,
            )
        elif ema_cross == -1 and trend_strong:
            conf = 0.55 + min((adx - 20) / 40, 0.2) + (vol_ratio > 1.2) * 0.1
            return StrategySignal(
                signal="SELL",
                confidence=min(conf, 0.9),
                strategy_name=self.name,
                reason=f"EMA 9/21 death cross (ADX:{adx:.0f}, vol:{vol_ratio:.1f}x)",
                suggested_sl_pct=0.015,
                suggested_tp_pct=0.035,
            )

        # No crossover this bar, but check trend alignment
        ema_9 = latest.get("ema_9", 0)
        ema_21 = latest.get("ema_21", 0)
        if ema_9 and ema_21 and trend_strong:
            if ema_9 > ema_21 * 1.002 and rsi > 50:
                return StrategySignal(
                    signal="BUY",
                    confidence=0.45,
                    strategy_name=self.name,
                    reason=f"EMA bullish alignment (ADX:{adx:.0f})",
                    suggested_sl_pct=0.015,
                    suggested_tp_pct=0.03,
                )
            elif ema_9 < ema_21 * 0.998 and rsi < 50:
                return StrategySignal(
                    signal="SELL",
                    confidence=0.45,
                    strategy_name=self.name,
                    reason=f"EMA bearish alignment (ADX:{adx:.0f})",
                    suggested_sl_pct=0.015,
                    suggested_tp_pct=0.03,
                )

        return StrategySignal(
            signal="HOLD",
            confidence=0.3,
            strategy_name=self.name,
            reason=f"No EMA cross (ADX:{adx:.0f})",
        )


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud strategy using TK-cross, cloud position, and ADX filter.

    Generates BUY signals when Tenkan/Kijun cross is bullish (tk_cross=1), price is
    above the cloud, and ADX >= threshold. Generates SELL signals on the mirror-image
    bearish conditions. When no fresh cross is present but trend alignment persists, a
    lower-confidence directional signal is emitted. Returns HOLD (confidence=0.0) when
    fewer than 3 bars are available (data-guard).

    Confidence builds from ``confidence_base`` (0.60) + volume bonus (0.10) + ADX
    bonus (up to 0.15), capped at 0.90 (actual max 0.85 with current bonus structure). SL = 2× ATR, TP = 4× ATR (fresh-cross) or 3× ATR (alignment-only).

    Best regimes: TRENDING_UP, TRENDING_DOWN.

    Source: Goichi Hosoda (1969). Widely used in Japanese & Asian institutional trading.
    """

    name = "Ichimoku"
    best_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters. Recognised keys:
                ``adx_threshold`` (default 20), ``confidence_base`` (default 0.6).
        """
        p = params or {}
        self.adx_threshold = float(p.get("adx_threshold", 20))
        self.confidence_base = float(p.get("confidence_base", 0.6))

    def generate_signal(self, df: pd.DataFrame, sentiment: SentimentState | None = None) -> StrategySignal:
        """Generate BUY/SELL/HOLD from Ichimoku TK-cross + cloud position + ADX filter.

        Args:
            df: Indicator-enriched OHLCV DataFrame. Required columns:
                ``ichimoku_tk_cross`` (1/-1/0), ``ichimoku_above_cloud``,
                ``ichimoku_below_cloud``, ``ichimoku_tenkan``, ``ichimoku_kijun``,
                ``adx``, ``volume_ratio``, ``atr_pct``.
            sentiment: Ignored by this strategy.

        Returns:
            StrategySignal with BUY/SELL/HOLD. Fresh-cross signals have confidence
            0.60–0.85; alignment-only signals are fixed at 0.45.
            HOLD confidence is 0.3 (normal) or 0.0 (data-guard: < 3 bars).
        """
        if len(df) < 3:
            return StrategySignal(
                signal="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason="Not enough data",
            )

        latest = df.iloc[-1]
        tk_cross = latest.get("ichimoku_tk_cross", 0)
        above_cloud = latest.get("ichimoku_above_cloud", 0)
        below_cloud = latest.get("ichimoku_below_cloud", 0)
        tenkan = latest.get("ichimoku_tenkan", 0.0)
        kijun = latest.get("ichimoku_kijun", 0.0)
        adx = latest.get("adx", 0.0)
        vol_ratio = latest.get("volume_ratio", 1.0)
        atr_pct = latest.get("atr_pct", 0.02)

        trend_strong = adx >= self.adx_threshold

        if tk_cross == 1 and above_cloud and trend_strong:
            conf = self.confidence_base
            conf += 0.1 if vol_ratio > 1.2 else 0.0
            conf += min((adx - self.adx_threshold) / 30.0, 0.15)
            return StrategySignal(
                signal="BUY",
                confidence=min(conf, 0.90),
                strategy_name=self.name,
                reason=f"Ichimoku TK bullish cross above cloud (ADX:{adx:.0f})",
                suggested_sl_pct=atr_pct * 2.0,
                suggested_tp_pct=atr_pct * 4.0,
            )

        if tk_cross == -1 and below_cloud and trend_strong:
            conf = self.confidence_base
            conf += 0.1 if vol_ratio > 1.2 else 0.0
            conf += min((adx - self.adx_threshold) / 30.0, 0.15)
            return StrategySignal(
                signal="SELL",
                confidence=min(conf, 0.90),
                strategy_name=self.name,
                reason=f"Ichimoku TK bearish cross below cloud (ADX:{adx:.0f})",
                suggested_sl_pct=atr_pct * 2.0,
                suggested_tp_pct=atr_pct * 4.0,
            )

        # Sustained trend alignment (no fresh cross — lower confidence)
        if above_cloud and trend_strong and tenkan > kijun:
            return StrategySignal(
                signal="BUY",
                confidence=0.45,
                strategy_name=self.name,
                reason=f"Ichimoku bullish alignment above cloud (ADX:{adx:.0f})",
                suggested_sl_pct=atr_pct * 2.0,
                suggested_tp_pct=atr_pct * 3.0,
            )
        if below_cloud and trend_strong and tenkan < kijun:
            return StrategySignal(
                signal="SELL",
                confidence=0.45,
                strategy_name=self.name,
                reason=f"Ichimoku bearish alignment below cloud (ADX:{adx:.0f})",
                suggested_sl_pct=atr_pct * 2.0,
                suggested_tp_pct=atr_pct * 3.0,
            )

        return StrategySignal(
            signal="HOLD",
            confidence=0.3,
            strategy_name=self.name,
            reason="No Ichimoku signal",
        )



class RSIDivergenceStrategy(BaseStrategy):
    """RSI divergence + Stochastic crossover strategy (v10).

    Detects hidden momentum by comparing price direction against RSI direction
    over a lookback window. Requires a Stochastic crossover in the appropriate
    zone to confirm entry.

    Signal logic:

    - Bullish divergence: price makes a lower low but RSI makes a higher low
      over the last ``lookback`` bars, confirmed when ``stoch_k`` crosses
      above ``stoch_d`` in the oversold zone (``stoch_k < stoch_threshold``).
    - Bearish divergence: price makes a higher high but RSI makes a lower high
      over the last ``lookback`` bars, confirmed when ``stoch_k`` crosses
      below ``stoch_d`` in the overbought zone (``stoch_k > 100 - stoch_threshold``).

    Confidence is 0.65 for strong divergence (RSI delta > 10 points) or 0.55
    for weak divergence.  HOLD uses 0.3 (data-guard returns 0.0).
    """

    name = "RSIDivergence"
    best_regimes = [MarketRegime.RANGING, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Initialise with optional evolved parameters.

        Args:
            params: Dict of evolved hyperparameters.  Recognised keys:
                ``lookback`` (default 20, range 15–30) and
                ``stoch_threshold`` (default 35.0, range 25–40).
        """
        p = params or {}
        self.lookback: int = int(p.get("lookback", 20))
        self.stoch_threshold: float = float(p.get("stoch_threshold", 35.0))

    def generate_signal(
        self,
        df: pd.DataFrame,
        sentiment: SentimentState | None = None,
    ) -> StrategySignal:
        """Detect RSI divergence confirmed by Stochastic crossover.

        Args:
            df: Indicator-enriched OHLCV DataFrame.  Required columns:
                ``close``, ``rsi``, ``stoch_k``, ``stoch_d``, ``atr_pct``.
            sentiment: Unused by this strategy.

        Returns:
            StrategySignal with BUY, SELL, or HOLD.  Confidence is 0.65 for
            strong divergence (RSI delta > 10) or 0.55 for weak.  HOLD
            returns confidence=0.3; data-guard returns confidence=0.0.
        """
        if len(df) < self.lookback + 1:
            return StrategySignal(
                signal="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reason="Not enough data for divergence detection",
            )

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        stoch_k = float(latest.get("stoch_k", 50))
        stoch_d = float(latest.get("stoch_d", 50))
        prev_stoch_k = float(prev.get("stoch_k", 50))
        prev_stoch_d = float(prev.get("stoch_d", 50))

        bull_div, bull_rsi_delta = self._detect_bullish_divergence(df)
        bear_div, bear_rsi_delta = self._detect_bearish_divergence(df)

        stoch_cross_up = (prev_stoch_k <= prev_stoch_d) and (stoch_k > stoch_d)
        stoch_cross_down = (prev_stoch_k >= prev_stoch_d) and (stoch_k < stoch_d)
        stoch_oversold = stoch_k < self.stoch_threshold
        stoch_overbought = stoch_k > (100.0 - self.stoch_threshold)

        atr_pct = float(latest.get("atr_pct", 0.02))

        if bull_div and stoch_cross_up and stoch_oversold:
            conf = 0.65 if bull_rsi_delta > 10.0 else 0.55
            return StrategySignal(
                signal="BUY",
                confidence=conf,
                strategy_name=self.name,
                reason=f"Bullish RSI div + stoch cross up (rsi_delta:{bull_rsi_delta:.1f}, stoch_k:{stoch_k:.1f})",
                suggested_sl_pct=atr_pct * 2.0,
                suggested_tp_pct=atr_pct * 3.5,
            )

        if bear_div and stoch_cross_down and stoch_overbought:
            conf = 0.65 if bear_rsi_delta > 10.0 else 0.55
            return StrategySignal(
                signal="SELL",
                confidence=conf,
                strategy_name=self.name,
                reason=f"Bearish RSI div + stoch cross down (rsi_delta:{bear_rsi_delta:.1f}, stoch_k:{stoch_k:.1f})",
                suggested_sl_pct=atr_pct * 2.0,
                suggested_tp_pct=atr_pct * 3.5,
            )

        return StrategySignal(
            signal="HOLD",
            confidence=0.3,
            strategy_name=self.name,
            reason="No RSI divergence or stochastic confirmation",
        )

    def _detect_bullish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Detect bullish RSI divergence over the lookback window.

        Bullish divergence: price forms a lower low while RSI forms a higher
        low — suggesting weakening downward momentum.

        Args:
            df: Full indicator DataFrame (at least ``lookback + 1`` rows).

        Returns:
            Tuple of (divergence_detected, rsi_delta) where
            ``divergence_detected`` is True when current close is below the
            lookback window's minimum close but current RSI is above the RSI
            at that price low, and ``rsi_delta`` is the absolute RSI point
            difference used for confidence scoring.
        """
        window = df.iloc[-(self.lookback + 1) : -1]
        current_close = float(df["close"].iloc[-1])
        current_rsi = float(df["rsi"].iloc[-1])

        window_price_low = float(window["close"].min())
        low_idx = window["close"].idxmin()
        window_rsi_at_low = float(df.loc[low_idx, "rsi"])

        price_lower_low = current_close < window_price_low
        rsi_higher_low = current_rsi > window_rsi_at_low
        rsi_delta = abs(current_rsi - window_rsi_at_low)

        return (price_lower_low and rsi_higher_low), rsi_delta

    def _detect_bearish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """Detect bearish RSI divergence over the lookback window.

        Bearish divergence: price forms a higher high while RSI forms a lower
        high — suggesting weakening upward momentum.

        Args:
            df: Full indicator DataFrame (at least ``lookback + 1`` rows).

        Returns:
            Tuple of (divergence_detected, rsi_delta) where
            ``divergence_detected`` is True when current close is above the
            lookback window's maximum close but current RSI is below the RSI
            at that price high, and ``rsi_delta`` is the absolute RSI point
            difference.
        """
        window = df.iloc[-(self.lookback + 1) : -1]
        current_close = float(df["close"].iloc[-1])
        current_rsi = float(df["rsi"].iloc[-1])

        window_price_high = float(window["close"].max())
        high_idx = window["close"].idxmax()
        window_rsi_at_high = float(df.loc[high_idx, "rsi"])

        price_higher_high = current_close > window_price_high
        rsi_lower_high = current_rsi < window_rsi_at_high
        rsi_delta = abs(current_rsi - window_rsi_at_high)

        return (price_higher_high and rsi_lower_high), rsi_delta

# ────────────────────────────────────────────────────────────
#  STRATEGY ENGINE — Adaptive strategy selection
# ────────────────────────────────────────────────────────────


class StrategyEngine:
    """Adaptive Strategy Engine (OPTIMIZED v3).

    Selects and runs only strategies relevant to the current market regime.
    Runs the primary strategy first and short-circuits when its confidence
    exceeds 0.8, skipping secondary strategies to save compute.  Otherwise
    combines all active strategies via weighted voting.

    Attributes:
        strategies: Dict mapping strategy name to instantiated strategy object.
        last_signals: Dict of the most recent signals returned per strategy name.
        REGIME_STRATEGY_MAP: Class-level mapping of regime -> primary/secondary/weights.
    """

    REGIME_STRATEGY_MAP = {
        MarketRegime.TRENDING_UP: {
            "primary": "Momentum",
            "secondary": ["Ichimoku", "EMACrossover", "OBVDivergence", "RSIDivergence"],
            "weights": {"Momentum": 0.35, "Ichimoku": 0.25, "EMACrossover": 0.20, "OBVDivergence": 0.12, "RSIDivergence": 0.08},
        },
        MarketRegime.TRENDING_DOWN: {
            "primary": "Momentum",
            "secondary": ["EMACrossover", "Ichimoku", "OBVDivergence", "RSIDivergence"],
            "weights": {"Momentum": 0.35, "EMACrossover": 0.22, "Ichimoku": 0.20, "OBVDivergence": 0.15, "RSIDivergence": 0.08},
        },
        MarketRegime.RANGING: {
            # Backtesting (2021-2026) showed MeanReversion/VWAP/Grid destroyed capital
            # in crypto's persistent trends. OBV+RSI divergence performs across all regimes.
            # Phase 6 Cycle 1: Removed Momentum from secondaries (trend-follower in ranging).
            # Phase 6 Cycle 3: Removed EMACrossover from secondaries — OBV+EMA ensemble was
            # the new dominant loser (119 BTC trades, −$84) after Momentum removal. Only
            # RSIDivergence remains as secondary — both OBV and RSI are divergence-based
            # and appropriate for sideways price action.
            "primary": "OBVDivergence",
            "secondary": ["RSIDivergence"],
            "weights": {"OBVDivergence": 0.60, "RSIDivergence": 0.40},
        },
        MarketRegime.HIGH_VOLATILITY: {
            "primary": "Breakout",
            "secondary": ["EMACrossover", "Momentum", "Scalping"],
            "weights": {"Breakout": 0.40, "EMACrossover": 0.25, "Momentum": 0.20, "Scalping": 0.15},
        },
    }

    def __init__(self, evolved_params: dict[str, dict[str, float]] | None = None) -> None:
        """Initialise all strategy instances with optional evolved parameters.

        Args:
            evolved_params: Dict mapping strategy name to a hyperparameter dict as
                produced by StrategyEvolver. Strategies not present in the dict are
                initialised with default parameters.
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
            # v9.1: Ichimoku Cloud strategy
            "Ichimoku": IchimokuStrategy(params=ep.get("Ichimoku")),
            # v10: RSI Divergence + Stochastic strategy
            "RSIDivergence": RSIDivergenceStrategy(params=ep.get("RSIDivergence")),
        }
        self.last_signals: dict[str, StrategySignal] = {}

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
        "Ichimoku": IchimokuStrategy,
        "RSIDivergence": RSIDivergenceStrategy,
    }

    def apply_evolved_params(self, evolved_params: dict[str, dict[str, float]]) -> None:
        """Hot-reload evolved parameters into strategies without a full restart.

        Reinstantiates only the strategies that appear in ``evolved_params`` and have
        a matching entry in ``_STRATEGY_CLASSES``.

        Args:
            evolved_params: Dict mapping strategy name to evolved hyperparameter dict.
        """
        for name, params in evolved_params.items():
            if name in self.strategies and params:
                cls = self._STRATEGY_CLASSES.get(name)
                if cls:
                    self.strategies[name] = cls(params=params)

    def run(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentState | None = None,
    ) -> StrategySignal:
        """Run the regime-appropriate strategies and return a combined signal.

        Runs the primary strategy first.  If it produces a non-HOLD signal with
        confidence > 0.8, secondary strategies are skipped (short-circuit).
        Otherwise, secondary strategies are also run and their signals are combined
        via weighted voting.

        Args:
            df: Indicator-enriched OHLCV DataFrame passed to each strategy.
            regime: Current detected market regime used to select strategies.
            sentiment: Optional sentiment state forwarded to sentiment-aware strategies.

        Returns:
            A StrategySignal representing the ensemble consensus. BUY/SELL confidence
            is capped at 0.95. HOLD confidence reflects the weighted hold score ratio.
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
                signal="HOLD",
                confidence=0.5,
                strategy_name="Ensemble",
                reason="No signals",
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
