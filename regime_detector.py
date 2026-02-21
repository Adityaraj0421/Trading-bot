"""
Market Regime Detector — OPTIMIZED v2.
Reuses pre-computed indicators instead of recalculating.
Bounded history, cached HMM results.
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from collections import deque


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    volatility: float
    trend_strength: float
    regime_duration: int


class RegimeDetector:
    """
    Multi-method regime detection.
    OPTIMIZED: Accepts pre-computed df_ind to avoid redundant indicator math.
    Uses bounded deque for history (max 500 entries).
    """

    MAX_HISTORY = 500

    def __init__(self):
        self.hmm_model = None
        self.has_hmm = False
        self.current_regime = MarketRegime.RANGING
        self.regime_history = deque(maxlen=self.MAX_HISTORY)
        self.regime_duration = 0
        self._hmm_cache_key = None
        self._hmm_cache_result = None
        self._try_init_hmm()

    def _try_init_hmm(self):
        try:
            import warnings
            warnings.filterwarnings("ignore", module="hmmlearn")
            from hmmlearn.hmm import GaussianHMM
            self.hmm_model = GaussianHMM(
                n_components=3, covariance_type="full",
                n_iter=200, random_state=42, init_params="",
            )
            self.has_hmm = True
        except ImportError:
            self.has_hmm = False

    def detect(self, df: pd.DataFrame, df_ind: pd.DataFrame = None) -> RegimeState:
        """
        Detect market regime. Accepts pre-computed indicators (df_ind)
        to reuse ATR, BBs, MAs already computed by Indicators.add_all().
        """
        if len(df) < 50:
            return RegimeState(MarketRegime.RANGING, 0.5, 0.0, 0.0, 0)

        vol_regime = self._volatility_regime(df, df_ind)
        trend_regime = self._trend_regime(df, df_ind)
        hmm_regime = self._hmm_regime(df, df_ind) if self.has_hmm else None

        regime, confidence = self._combine_regimes(vol_regime, trend_regime, hmm_regime)

        if regime == self.current_regime:
            self.regime_duration += 1
        else:
            self.regime_duration = 1
            self.current_regime = regime

        state = RegimeState(
            regime=regime, confidence=confidence,
            volatility=vol_regime["volatility_pct"],
            trend_strength=trend_regime["strength"],
            regime_duration=self.regime_duration,
        )
        self.regime_history.append(state)
        return state

    def _volatility_regime(self, df: pd.DataFrame, df_ind: pd.DataFrame = None) -> dict:
        """Reuses pre-computed ATR and BB width from indicators."""
        if df_ind is not None and "atr_pct" in df_ind.columns:
            atr_pct = df_ind["atr_pct"]
            current_vol = float(atr_pct.iloc[-1])
            vol_percentile = float((atr_pct < current_vol).mean())

            if "bb_width" in df_ind.columns:
                bb_width = df_ind["bb_width"]
                bb_percentile = float((bb_width < bb_width.iloc[-1]).mean())
            else:
                bb_percentile = 0.5
        else:
            # Fallback: compute from raw data
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_pct = (tr.rolling(14).mean() / df["close"]).dropna()
            if len(atr_pct) < 20:
                return {"regime": MarketRegime.RANGING, "volatility_pct": 0.0, "confidence": 0.5}
            current_vol = float(atr_pct.iloc[-1])
            vol_percentile = float((atr_pct < current_vol).mean())
            bb_percentile = 0.5

        avg_pct = (vol_percentile + bb_percentile) / 2

        if avg_pct > 0.8:
            return {"regime": MarketRegime.HIGH_VOLATILITY, "volatility_pct": current_vol, "confidence": min(avg_pct, 0.95)}
        elif avg_pct < 0.3:
            return {"regime": MarketRegime.RANGING, "volatility_pct": current_vol, "confidence": float(1 - avg_pct)}
        else:
            return {"regime": MarketRegime.RANGING, "volatility_pct": current_vol, "confidence": 0.5}

    def _trend_regime(self, df: pd.DataFrame, df_ind: pd.DataFrame = None) -> dict:
        """Reuses pre-computed MAs from indicators."""
        close = df["close"]
        latest_close = float(close.iloc[-1])

        # Reuse pre-computed MAs if available
        if df_ind is not None and "sma_10" in df_ind.columns:
            latest_10 = float(df_ind["sma_10"].iloc[-1])
            latest_20 = float(df_ind["sma_20"].iloc[-1])
            latest_50 = float(df_ind["sma_50"].iloc[-1])
        else:
            latest_10 = float(close.rolling(10).mean().iloc[-1])
            latest_20 = float(close.rolling(20).mean().iloc[-1])
            latest_50 = float(close.rolling(50).mean().iloc[-1])

        # MA alignment score
        alignment = 0
        alignment += 1 if latest_close > latest_10 else -1
        alignment += 1 if latest_10 > latest_20 else -1
        alignment += 1 if latest_20 > latest_50 else -1

        # ADX proxy
        returns = close.pct_change().dropna()
        pos_moves = returns.where(returns > 0, 0).rolling(14).mean()
        neg_moves = (-returns.where(returns < 0, 0)).rolling(14).mean()
        total = pos_moves + neg_moves
        dx = ((pos_moves - neg_moves).abs() / total.replace(0, np.nan)).dropna()
        adx_proxy = dx.rolling(14).mean()
        trend_strength = float(adx_proxy.iloc[-1]) if len(adx_proxy) > 0 else 0.0

        if alignment >= 2 and trend_strength > 0.3:
            regime = MarketRegime.TRENDING_UP
            conf = min(0.6 + trend_strength * 0.3, 0.95)
        elif alignment <= -2 and trend_strength > 0.3:
            regime = MarketRegime.TRENDING_DOWN
            conf = min(0.6 + trend_strength * 0.3, 0.95)
        else:
            regime = MarketRegime.RANGING
            conf = 0.5 + (1 - trend_strength) * 0.3

        return {"regime": regime, "strength": trend_strength, "alignment": alignment, "confidence": float(conf)}

    def _hmm_regime(self, df: pd.DataFrame, df_ind: pd.DataFrame = None) -> dict | None:
        """Reuses pre-computed log returns and rolling vol."""
        try:
            # Reuse from indicators if available
            if df_ind is not None and "log_returns" in df_ind.columns and "rolling_vol_10" in df_ind.columns:
                log_ret = df_ind["log_returns"].dropna()
                vol = df_ind["rolling_vol_10"].dropna()
                common = log_ret.index.intersection(vol.index)
                features = np.column_stack([log_ret.loc[common].values, vol.loc[common].values])
            else:
                returns = np.log(df["close"] / df["close"].shift(1)).dropna()
                vol = returns.rolling(10).std().dropna()
                common = returns.index.intersection(vol.index)
                features = np.column_stack([returns.loc[common].values, vol.loc[common].values])

            if len(features) < 50:
                return None

            # Cache: refit only if data content changed (hash last 20 rows + length)
            tail_hash = hash(features[-20:].tobytes()) if len(features) >= 20 else hash(features.tobytes())
            cache_key = (len(features), tail_hash)
            if self._hmm_cache_key != cache_key:
                self.hmm_model.fit(features)
                self._hmm_cache_key = cache_key

            states = self.hmm_model.predict(features)
            proba = self.hmm_model.predict_proba(features)

            # Characterize states by mean return using groupby-like approach
            returns_arr = features[:, 0]
            state_means = {s: returns_arr[states == s].mean() if (states == s).sum() > 0 else 0 for s in range(3)}

            sorted_states = sorted(state_means.items(), key=lambda x: x[1])
            bear_state, range_state, bull_state = sorted_states[0][0], sorted_states[1][0], sorted_states[2][0]

            current = states[-1]
            if current == bull_state:
                regime = MarketRegime.TRENDING_UP
            elif current == bear_state:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING

            return {"regime": regime, "confidence": float(max(proba[-1]))}
        except Exception as e:
            logging.getLogger(__name__).debug("HMM regime detection failed: %s", e)
            return None

    def _combine_regimes(self, vol_result, trend_result, hmm_result):
        votes = {}

        if vol_result["regime"] == MarketRegime.HIGH_VOLATILITY and vol_result["confidence"] > 0.8:
            return MarketRegime.HIGH_VOLATILITY, vol_result["confidence"]

        trend_r = trend_result["regime"]
        votes[trend_r] = votes.get(trend_r, 0) + 0.4 * trend_result["confidence"]

        vol_r = vol_result["regime"]
        votes[vol_r] = votes.get(vol_r, 0) + 0.2 * vol_result["confidence"]

        if hmm_result:
            hmm_r = hmm_result["regime"]
            votes[hmm_r] = votes.get(hmm_r, 0) + 0.4 * hmm_result["confidence"]
        else:
            votes[trend_r] = votes.get(trend_r, 0) + 0.2 * trend_result["confidence"]

        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[winner] / total if total > 0 else 0.5
        return winner, confidence
