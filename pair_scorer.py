"""
Pair Scorer — Dynamic pair selection for multi-pair trading (v9.1).

Ranks candidate pairs from PAIR_POOL by profit potential (ATR% × volume_ratio)
and returns the top PAIR_SELECTOR_TOP_N. Called from the agent main loop every
PAIR_SCORER_INTERVAL_CYCLES cycles.

ATR% and volume_ratio are computed inline from raw OHLCV to avoid polluting the
Indicators class-level cache (which is keyed per DataFrame identity).
"""

import logging
from dataclasses import dataclass

import pandas as pd

from config import Config

_log = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Scored pair entry with its component metrics."""

    symbol: str
    score: float  # atr_pct × volume_ratio
    atr_pct: float
    volume_ratio: float


class PairScorer:
    """
    Score pairs in PAIR_POOL and select the top N by profit potential.

    Scoring formula:
        score = atr_pct_14bar × volume_ratio_20bar

    - atr_pct: normalised volatility (higher → more price movement opportunity)
    - volume_ratio: current volume / 20-bar average (higher → more liquidity)

    The inline ATR computation avoids importing Indicators, preventing eviction
    of the main-pair indicators from the class-level cache.
    """

    def __init__(self) -> None:
        self._last_scores: list[PairScore] = []

    def score_pairs(self, price_data: dict[str, pd.DataFrame]) -> list[PairScore]:
        """
        Score all pairs present in price_data and return sorted list (highest first).

        Args:
            price_data: dict mapping symbol -> raw OHLCV DataFrame.
                        Must contain columns: open, high, low, close, volume.

        Returns:
            List of PairScore sorted descending by score.
        """
        scores: list[PairScore] = []
        for symbol, df in price_data.items():
            if df is None or df.empty or len(df) < 20:
                _log.debug("PairScorer: skipping %s — insufficient data (%d bars)", symbol, len(df) if df is not None else 0)
                continue
            try:
                close = df["close"]
                high = df["high"]
                low = df["low"]
                vol = df["volume"]

                # ATR% — inline computation (no Indicators dependency)
                prev_close = close.shift(1)
                tr = pd.concat(
                    [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                    axis=1,
                ).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                last_close = float(close.iloc[-1])
                atr_pct = float(atr / last_close) if last_close > 0 else 0.0

                # Volume ratio
                vol_sma = vol.rolling(20).mean().iloc[-1]
                volume_ratio = float(vol.iloc[-1] / vol_sma) if vol_sma > 0 else 1.0

                raw_score = atr_pct * volume_ratio
                scores.append(
                    PairScore(
                        symbol=symbol,
                        score=raw_score,
                        atr_pct=atr_pct,
                        volume_ratio=volume_ratio,
                    )
                )
            except Exception:
                _log.warning("PairScorer: error scoring %s", symbol, exc_info=True)

        scores.sort(key=lambda x: x.score, reverse=True)
        self._last_scores = scores
        return scores

    def select_top_pairs(self, price_data: dict[str, pd.DataFrame]) -> list[str]:
        """
        Return the top PAIR_SELECTOR_TOP_N symbols from the pool.
        Falls back to Config.TRADING_PAIRS if scoring fails or yields no results.

        Args:
            price_data: dict mapping symbol -> raw OHLCV DataFrame.

        Returns:
            List of symbol strings to actively trade this scoring period.
        """
        top_n = Config.PAIR_SELECTOR_TOP_N
        try:
            scored = self.score_pairs(price_data)
            if not scored:
                _log.warning("PairScorer: no pairs scored — using default TRADING_PAIRS")
                return list(Config.TRADING_PAIRS)
            top = [s.symbol for s in scored[:top_n]]
            _log.info(
                "PairScorer: top %d pairs selected: %s",
                top_n,
                ", ".join(f"{s.symbol}(score={s.score:.5f})" for s in scored[:top_n]),
            )
            return top
        except Exception:
            _log.warning("PairScorer: selection failed — using default TRADING_PAIRS", exc_info=True)
            return list(Config.TRADING_PAIRS)

    def get_last_scores(self) -> list[PairScore]:
        """Return the most recently computed pair scores (empty before first call)."""
        return self._last_scores
