"""
Technical indicators module — OPTIMIZED v2.
Single-pass vectorized computation with result caching.
Eliminates redundant recomputation across modules.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Feature columns used by the ML model (v9.1: expanded to 24 — added Williams %R and CCI)
FEATURE_COLUMNS = [
    # Core momentum & oscillators
    "rsi",
    "macd",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    # Bollinger Band positioning
    "bb_position",
    "bb_width",
    # Volatility
    "atr_pct",
    "high_low_range",
    # Volume analysis
    "volume_ratio",
    "obv_divergence",
    # Trend strength
    "close_to_sma20",
    "close_to_sma50",
    "close_to_vwap",
    "ema_cross",
    "adx",
    # Returns at multiple horizons
    "returns_1",
    "returns_5",
    "returns_10",
    # Directional movement
    "plus_di",
    "minus_di",
    # Volatility regime
    "rolling_vol_10",
    # v9.1: New oscillators for richer ML signal
    "williams_r",
    "cci",
]


class Indicators:
    """
    Compute all indicators in a single vectorized pass.
    Caches results — only recomputes when data actually changes.
    """

    _cache_key: tuple[int, float, str] | None = None
    _cache_result: pd.DataFrame | None = None

    @classmethod
    def add_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the OHLCV DataFrame.

        Returns the cached result when the data is unchanged (cache key
        is ``(len, last_close, last_timestamp)``).  Computes all indicators
        in a single vectorised pass otherwise.

        Indicators added (columns):
            Moving averages: ``sma_10``, ``sma_20``, ``sma_50``,
            ``ema_12``, ``ema_26``, ``ema_9``, ``ema_21``.
            MACD: ``macd``, ``macd_signal``, ``macd_hist``.
            RSI: ``rsi``.
            Stochastic: ``stoch_k``, ``stoch_d``.
            Bollinger Bands: ``bb_upper``, ``bb_lower``,
            ``bb_width``, ``bb_position``.
            ATR: ``atr``, ``atr_pct``.
            Volume: ``volume_sma_20``, ``volume_ratio``.
            Price action: ``returns_1``, ``returns_5``, ``returns_10``,
            ``high_low_range``, ``close_to_sma20``, ``close_to_sma50``.
            Log returns / vol: ``log_returns``, ``rolling_vol_10``.
            VWAP: ``vwap``, ``close_to_vwap``.
            OBV: ``obv``, ``obv_sma_20``, ``obv_divergence``.
            EMA cross: ``ema_cross``.
            ADX/DI: ``adx``, ``plus_di``, ``minus_di``.
            Williams %R: ``williams_r``.
            CCI: ``cci``.
            Ichimoku: ``ichimoku_tenkan``, ``ichimoku_kijun``,
            ``ichimoku_span_a``, ``ichimoku_span_b``,
            ``ichimoku_above_cloud``, ``ichimoku_below_cloud``,
            ``ichimoku_tk_cross``.
            Target: ``future_return``.

        Args:
            df: Raw OHLCV DataFrame with columns
                ``open``, ``high``, ``low``, ``close``, ``volume``,
                indexed by timestamp.

        Returns:
            Copy of ``df`` with all indicator columns appended. Rows with
            NaN values (warm-up period) are dropped.
        """
        key = (len(df), float(df["close"].iloc[-1]), str(df.index[-1]))
        if cls._cache_key == key and cls._cache_result is not None:
            return cls._cache_result

        out = df.copy()
        close = out["close"]
        high = out["high"]
        low = out["low"]
        volume = out["volume"]

        # --- Moving Averages (computed once, reused below) ---
        out["sma_10"] = close.rolling(10).mean()
        out["sma_20"] = close.rolling(20).mean()
        out["sma_50"] = close.rolling(50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["ema_12"] = ema_12
        out["ema_26"] = ema_26

        # --- MACD ---
        out["macd"] = ema_12 - ema_26
        out["macd_signal"] = out["macd"].ewm(span=9).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        # --- RSI ---
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi"] = 100 - (100 / (1 + rs))

        # --- Stochastic ---
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        hl_range = high_14 - low_14
        out["stoch_k"] = 100 * (close - low_14) / hl_range.replace(0, np.nan)
        out["stoch_d"] = out["stoch_k"].rolling(3).mean()

        # --- Bollinger Bands (reuse sma_20) ---
        std_20 = close.rolling(20).std()
        out["bb_upper"] = out["sma_20"] + 2 * std_20
        out["bb_lower"] = out["sma_20"] - 2 * std_20
        bb_range = out["bb_upper"] - out["bb_lower"]
        out["bb_width"] = bb_range / out["sma_20"].replace(0, np.nan)
        out["bb_position"] = (close - out["bb_lower"]) / bb_range.replace(0, np.nan)

        # --- ATR (single computation, reused by regime_detector + strategies) ---
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(14).mean()
        out["atr_pct"] = out["atr"] / close

        # --- Volume ---
        vol_sma = volume.rolling(20).mean()
        out["volume_sma_20"] = vol_sma
        out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

        # --- Price action ---
        out["returns_1"] = close.pct_change(1)
        out["returns_5"] = close.pct_change(5)
        out["returns_10"] = close.pct_change(10)
        out["high_low_range"] = high_low / close
        out["close_to_sma20"] = (close - out["sma_20"]) / out["sma_20"].replace(0, np.nan)
        out["close_to_sma50"] = (close - out["sma_50"]) / out["sma_50"].replace(0, np.nan)

        # --- Log returns + rolling vol (for regime detector — compute once here) ---
        out["log_returns"] = np.log(close / close.shift(1))
        out["rolling_vol_10"] = out["log_returns"].rolling(10).std()

        # --- v7.0: VWAP (Volume Weighted Average Price) ---
        typical_price = (high + low + close) / 3
        cumvol = volume.cumsum()
        out["vwap"] = (typical_price * volume).cumsum() / cumvol.replace(0, np.nan)
        out["close_to_vwap"] = (close - out["vwap"]) / out["vwap"].replace(0, np.nan)

        # --- v7.0: OBV (On-Balance Volume) ---
        obv_direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        out["obv"] = (volume * obv_direction).cumsum()
        out["obv_sma_20"] = out["obv"].rolling(20).mean()
        out["obv_divergence"] = np.where(
            (close > close.shift(5)) & (out["obv"] < out["obv"].shift(5)),
            -1,  # bearish divergence
            np.where(
                (close < close.shift(5)) & (out["obv"] > out["obv"].shift(5)),
                1,  # bullish divergence
                0,
            ),
        )

        # --- v7.0: EMA crossover signals ---
        out["ema_9"] = close.ewm(span=9, adjust=False).mean()
        out["ema_21"] = close.ewm(span=21, adjust=False).mean()
        out["ema_cross"] = np.where(
            (out["ema_9"] > out["ema_21"]) & (out["ema_9"].shift(1) <= out["ema_21"].shift(1)),
            1,
            np.where((out["ema_9"] < out["ema_21"]) & (out["ema_9"].shift(1) >= out["ema_21"].shift(1)), -1, 0),
        )

        # --- v7.0: ADX (Average Directional Index) for trend strength ---
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm_arr = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        atr_14 = true_range.rolling(14).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_14.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm_arr, index=df.index).rolling(14).mean() / atr_14.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        out["adx"] = dx.rolling(14).mean()
        out["plus_di"] = plus_di
        out["minus_di"] = minus_di

        # --- v9.1: Williams %R (14-period): -100 (oversold) to 0 (overbought) ---
        hh14 = high.rolling(14).max()
        ll14 = low.rolling(14).min()
        out["williams_r"] = -100 * (hh14 - close) / (hh14 - ll14).replace(0, np.nan)

        # --- v9.1: CCI (20-period Commodity Channel Index) ---
        # Reuses typical_price computed above for VWAP
        mean_dev = typical_price.rolling(20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        out["cci"] = (typical_price - typical_price.rolling(20).mean()) / (
            0.015 * mean_dev.replace(0, np.nan)
        )

        # --- v9.1: Ichimoku Cloud (structural trend tool — not in FEATURE_COLUMNS) ---
        # No Chikou span: shift(-26) would consume 26 tail rows from every feed.
        tenkan_high = high.rolling(9).max()
        tenkan_low = low.rolling(9).min()
        out["ichimoku_tenkan"] = (tenkan_high + tenkan_low) / 2

        kijun_high = high.rolling(26).max()
        kijun_low = low.rolling(26).min()
        out["ichimoku_kijun"] = (kijun_high + kijun_low) / 2

        out["ichimoku_span_a"] = ((out["ichimoku_tenkan"] + out["ichimoku_kijun"]) / 2).shift(26)
        span_b_high = high.rolling(52).max()
        span_b_low = low.rolling(52).min()
        out["ichimoku_span_b"] = ((span_b_high + span_b_low) / 2).shift(26)

        # Derived helper flags consumed by IchimokuStrategy
        out["ichimoku_above_cloud"] = (
            (close > out["ichimoku_span_a"]) & (close > out["ichimoku_span_b"])
        ).astype(int)
        out["ichimoku_below_cloud"] = (
            (close < out["ichimoku_span_a"]) & (close < out["ichimoku_span_b"])
        ).astype(int)
        out["ichimoku_tk_cross"] = np.where(
            (out["ichimoku_tenkan"] > out["ichimoku_kijun"])
            & (out["ichimoku_tenkan"].shift(1) <= out["ichimoku_kijun"].shift(1)),
            1,
            np.where(
                (out["ichimoku_tenkan"] < out["ichimoku_kijun"])
                & (out["ichimoku_tenkan"].shift(1) >= out["ichimoku_kijun"].shift(1)),
                -1,
                0,
            ),
        )

        # --- Target for training ---
        out["future_return"] = close.shift(-1) / close - 1

        out.dropna(inplace=True)

        cls._cache_key = key
        cls._cache_result = out
        return out

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return the list of feature column names used by the ML model.

        Returns:
            The module-level ``FEATURE_COLUMNS`` list (24 entries for v9.1).
        """
        return FEATURE_COLUMNS

    @classmethod
    def invalidate_cache(cls) -> None:
        """Clear the cached indicator DataFrame, forcing recomputation on next call."""
        cls._cache_key = None
        cls._cache_result = None
