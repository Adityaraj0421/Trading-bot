"""
Multi-Timeframe Confirmation Module v2.0
=========================================
Confirms signals across multiple timeframes to filter false signals.

v2.0 enhancements:
  - Multi-timeframe regime confirmation (15m, 1h, 4h, 1D)
  - Requires 2+ timeframes to agree before trading
  - Higher TF as trend bias, lower TF for entry timing
  - Regime transition probability tracking
  - Original single-TF confirmation preserved for backward compatibility

Logic:
  - Primary timeframe: generates signals
  - Multiple confirmation timeframes: validate direction
  - Regime consensus scoring across all timeframes
  - Transition probability matrix from historical regime sequences
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass, field

import pandas as pd

from indicators import Indicators

_log = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class TimeframeRegime:
    """Regime state for a single timeframe."""

    timeframe: str
    bias: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    trend_alignment: int  # MA alignment score (-5 to +5)
    rsi: float
    macd_direction: str  # "bullish", "bearish"
    momentum: float  # Recent % return
    details: dict = field(default_factory=dict)


@dataclass
class MultiTFConsensus:
    """Consensus result across all timeframes."""

    overall_bias: str  # "bullish", "bearish", "neutral", "conflicted"
    consensus_strength: float  # 0.0 to 1.0
    agreement_count: int  # How many TFs agree with majority
    total_timeframes: int
    regime_by_tf: dict  # {tf: TimeframeRegime}
    transition_probs: dict  # Next regime probabilities
    confidence_multiplier: float  # 0.5 to 1.5 applied to signal confidence


# ── Multi-Timeframe Confirmer ─────────────────────────────────────


class MultiTimeframeConfirmer:
    """
    v2.0: Confirms trading signals using multiple timeframes with
    regime detection, consensus scoring, and transition probability tracking.
    """

    # Timeframe hierarchy: higher = more weight in consensus
    TIMEFRAME_WEIGHTS = {
        "15m": 0.10,
        "1h": 0.25,
        "4h": 0.35,
        "1d": 0.30,
    }

    # Minimum agreement for trading
    MIN_AGREEMENT_RATIO = 0.5  # At least 50% of TFs must agree

    # Transition tracking
    MAX_TRANSITION_HISTORY = 200

    def __init__(self, timeframes: list[str] = None):
        self.timeframes = timeframes or ["15m", "1h", "4h", "1d"]
        self._htf_cache: dict = {}  # {tf: (cache_key, result)}

        # Regime transition tracking: {(from_regime, to_regime): count}
        self._transition_counts: Counter = Counter()
        self._regime_sequence: deque = deque(maxlen=self.MAX_TRANSITION_HISTORY)
        self._last_regime: str | None = None

    # ── Public Interface ──────────────────────────────────────────

    def get_multi_tf_consensus(self, df: pd.DataFrame) -> MultiTFConsensus:
        """
        Analyze regime across all configured timeframes and compute consensus.

        Args:
            df: OHLCV DataFrame at the primary (lowest) timeframe

        Returns:
            MultiTFConsensus with overall direction and confidence multiplier
        """
        regime_by_tf = {}

        for tf in self.timeframes:
            try:
                tf_regime = self._analyze_timeframe(df, tf)
                regime_by_tf[tf] = tf_regime
            except Exception as e:
                _log.debug("TF %s analysis failed: %s", tf, e)

        if not regime_by_tf:
            return MultiTFConsensus(
                overall_bias="neutral",
                consensus_strength=0.0,
                agreement_count=0,
                total_timeframes=0,
                regime_by_tf={},
                transition_probs={},
                confidence_multiplier=1.0,
            )

        # Compute weighted consensus
        overall_bias, consensus_strength, agreement = self._compute_consensus(regime_by_tf)

        # Track regime transition
        transition_probs = self._update_transitions(overall_bias)

        # Compute confidence multiplier based on agreement
        confidence_mult = self._consensus_to_multiplier(agreement, len(regime_by_tf), consensus_strength)

        return MultiTFConsensus(
            overall_bias=overall_bias,
            consensus_strength=round(consensus_strength, 3),
            agreement_count=agreement,
            total_timeframes=len(regime_by_tf),
            regime_by_tf=regime_by_tf,
            transition_probs=transition_probs,
            confidence_multiplier=round(confidence_mult, 3),
        )

    def get_htf_bias(self, df: pd.DataFrame, target_tf: str = "4h") -> dict:
        """
        Backward-compatible: single-timeframe bias check.
        """
        key = (len(df), float(df["close"].iloc[-1]), target_tf)
        cached = self._htf_cache.get(target_tf)
        if cached and cached[0] == key:
            return cached[1]

        htf = self.resample_to_higher_tf(df, target_tf=target_tf)
        if len(htf) < 20:
            result = {"bias": "neutral", "strength": 0.0, "details": {}}
            self._htf_cache[target_tf] = (key, result)
            return result

        Indicators.invalidate_cache()
        htf_ind = Indicators.add_all(htf)
        Indicators.invalidate_cache()

        if len(htf_ind) < 5:
            result = {"bias": "neutral", "strength": 0.0, "details": {}}
            self._htf_cache[target_tf] = (key, result)
            return result

        latest = htf_ind.iloc[-1]
        signals = self._compute_directional_signals(htf_ind, latest)

        total = sum(signals.values())
        max_possible = len(signals)

        if total >= 3:
            bias, strength = "bullish", min(total / max_possible, 1.0)
        elif total <= -3:
            bias, strength = "bearish", min(abs(total) / max_possible, 1.0)
        else:
            bias, strength = "neutral", 0.3

        result = {"bias": bias, "strength": strength, "details": signals}
        self._htf_cache[target_tf] = (key, result)
        return result

    def confirm_signal(self, signal: str, confidence: float, htf_bias: dict) -> tuple[str, float]:
        """Backward-compatible: single-TF confirmation."""
        bias = htf_bias["bias"]
        strength = htf_bias["strength"]

        if signal == "HOLD":
            return signal, confidence

        if signal == "BUY":
            if bias == "bullish":
                return "BUY", min(confidence + 0.15 * strength, 0.95)
            elif bias == "bearish":
                adjusted = confidence * (1 - 0.25 * strength)
                return ("HOLD", adjusted) if adjusted < 0.3 else ("BUY", adjusted)
            return "BUY", confidence

        if signal == "SELL":
            if bias == "bearish":
                return "SELL", min(confidence + 0.15 * strength, 0.95)
            elif bias == "bullish":
                adjusted = confidence * (1 - 0.25 * strength)
                return ("HOLD", adjusted) if adjusted < 0.3 else ("SELL", adjusted)
            return "SELL", confidence

        return signal, confidence

    def confirm_signal_multi_tf(self, signal: str, confidence: float, consensus: MultiTFConsensus) -> tuple[str, float]:
        """
        NEW v2.0: Confirm signal using full multi-timeframe consensus.

        Stronger than single-TF: uses weighted agreement across all TFs
        and regime transition probabilities.
        """
        if signal == "HOLD":
            return signal, confidence

        bias = consensus.overall_bias
        mult = consensus.confidence_multiplier

        # Apply consensus multiplier
        adjusted_conf = confidence * mult

        # Direction alignment check
        if signal == "BUY":
            if bias == "bullish":
                adjusted_conf = min(adjusted_conf + 0.1, 0.95)
            elif bias == "bearish":
                adjusted_conf *= 0.6  # Heavy penalty for counter-trend
                if adjusted_conf < 0.25:
                    return "HOLD", adjusted_conf
            elif bias == "conflicted":
                adjusted_conf *= 0.8

        elif signal == "SELL":
            if bias == "bearish":
                adjusted_conf = min(adjusted_conf + 0.1, 0.95)
            elif bias == "bullish":
                adjusted_conf *= 0.6
                if adjusted_conf < 0.25:
                    return "HOLD", adjusted_conf
            elif bias == "conflicted":
                adjusted_conf *= 0.8

        # Transition probability boost
        # If transition probs suggest regime continuation, boost further
        if consensus.transition_probs:
            continuation_prob = consensus.transition_probs.get(bias, 0)
            if continuation_prob > 0.6:
                adjusted_conf *= 1.05  # Slight boost for likely continuation
            elif continuation_prob < 0.3:
                adjusted_conf *= 0.95  # Slight penalty for likely reversal

        return signal, min(round(adjusted_conf, 4), 0.95)

    # ── Resampling ────────────────────────────────────────────────

    def resample_to_higher_tf(self, df: pd.DataFrame, source_tf: str = "1h", target_tf: str = "4h") -> pd.DataFrame:
        """Resample OHLCV to higher timeframe with proper aggregation."""
        tf_map = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"}
        resample_rule = tf_map.get(target_tf, "4h")

        htf = (
            df.resample(resample_rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        return htf

    # ── Internal: Per-Timeframe Analysis ──────────────────────────

    def _analyze_timeframe(self, df: pd.DataFrame, target_tf: str) -> TimeframeRegime:
        """Analyze regime for a single timeframe."""
        # The primary timeframe doesn't need resampling
        if target_tf == self.timeframes[0]:
            htf = df
        else:
            htf = self.resample_to_higher_tf(df, target_tf=target_tf)

        if len(htf) < 20:
            return TimeframeRegime(
                timeframe=target_tf,
                bias="neutral",
                strength=0.0,
                trend_alignment=0,
                rsi=50.0,
                macd_direction="neutral",
                momentum=0.0,
            )

        Indicators.invalidate_cache()
        htf_ind = Indicators.add_all(htf)
        Indicators.invalidate_cache()

        if len(htf_ind) < 5:
            return TimeframeRegime(
                timeframe=target_tf,
                bias="neutral",
                strength=0.0,
                trend_alignment=0,
                rsi=50.0,
                macd_direction="neutral",
                momentum=0.0,
            )

        latest = htf_ind.iloc[-1]
        signals = self._compute_directional_signals(htf_ind, latest)

        total = sum(signals.values())
        max_possible = max(len(signals), 1)

        if total >= 2:
            bias = "bullish"
            strength = min(total / max_possible, 1.0)
        elif total <= -2:
            bias = "bearish"
            strength = min(abs(total) / max_possible, 1.0)
        else:
            bias = "neutral"
            strength = 0.2

        # Extract individual metrics
        rsi = float(latest.get("rsi", 50))
        macd_dir = "bullish" if latest.get("macd", 0) > latest.get("macd_signal", 0) else "bearish"

        # Momentum: recent bars return
        lookback = min(5, len(htf_ind) - 1)
        momentum = (
            (float(htf_ind["close"].iloc[-1]) / float(htf_ind["close"].iloc[-lookback - 1]) - 1)
            if lookback > 0
            else 0.0
        )

        return TimeframeRegime(
            timeframe=target_tf,
            bias=bias,
            strength=round(strength, 3),
            trend_alignment=total,
            rsi=round(rsi, 1),
            macd_direction=macd_dir,
            momentum=round(momentum, 4),
            details=signals,
        )

    def _compute_directional_signals(self, htf_ind: pd.DataFrame, latest: pd.Series) -> dict:
        """Compute directional signals for a timeframe."""
        signals = {}

        # MA trend
        sma20 = latest.get("sma_20", latest["close"])
        signals["ma_trend"] = 1 if latest["close"] > sma20 else -1

        # MACD direction
        signals["macd"] = 1 if latest.get("macd", 0) > latest.get("macd_signal", 0) else -1

        # RSI zone
        rsi = latest.get("rsi", 50)
        if rsi > 55:
            signals["rsi"] = 1
        elif rsi < 45:
            signals["rsi"] = -1
        else:
            signals["rsi"] = 0

        # Price vs SMA50
        if "sma_50" in htf_ind.columns and pd.notna(latest.get("sma_50")):
            signals["sma50"] = 1 if latest["close"] > latest["sma_50"] else -1
        else:
            signals["sma50"] = 0

        # Recent momentum
        lookback = min(5, len(htf_ind) - 1)
        if lookback > 0:
            recent_return = htf_ind["close"].iloc[-1] / htf_ind["close"].iloc[-lookback - 1] - 1
            signals["momentum"] = 1 if recent_return > 0.01 else (-1 if recent_return < -0.01 else 0)
        else:
            signals["momentum"] = 0

        # BB position (new for v2.0)
        if "bb_upper" in htf_ind.columns and pd.notna(latest.get("bb_upper")):
            bb_position = (latest["close"] - latest["bb_lower"]) / (latest["bb_upper"] - latest["bb_lower"] + 1e-10)
            if bb_position > 0.7:
                signals["bb_position"] = 1
            elif bb_position < 0.3:
                signals["bb_position"] = -1
            else:
                signals["bb_position"] = 0

        return signals

    # ── Internal: Consensus Computation ───────────────────────────

    def _compute_consensus(self, regime_by_tf: dict) -> tuple[str, float, int]:
        """
        Compute weighted consensus across timeframes.

        Returns:
            (overall_bias, consensus_strength, agreement_count)
        """
        weighted_score = 0.0
        total_weight = 0.0
        biases = []

        for tf, regime in regime_by_tf.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.2)
            direction = 1 if regime.bias == "bullish" else (-1 if regime.bias == "bearish" else 0)
            weighted_score += direction * regime.strength * weight
            total_weight += weight
            biases.append(regime.bias)

        if total_weight <= 0:
            return "neutral", 0.0, 0

        normalized_score = weighted_score / total_weight

        # Count agreement
        bias_counts = Counter(biases)
        majority_bias, majority_count = bias_counts.most_common(1)[0]

        # Determine overall bias
        if abs(normalized_score) > 0.15:
            overall_bias = "bullish" if normalized_score > 0 else "bearish"
        elif majority_count >= 2 and majority_bias != "neutral":
            overall_bias = majority_bias
        elif len(set(biases) - {"neutral"}) > 1:
            overall_bias = "conflicted"
        else:
            overall_bias = "neutral"

        consensus_strength = abs(normalized_score)

        # Count how many TFs agree with the overall direction
        agreement = sum(1 for b in biases if b == overall_bias or (overall_bias == "conflicted" and b != "neutral"))

        return overall_bias, consensus_strength, agreement

    def _consensus_to_multiplier(self, agreement: int, total: int, strength: float) -> float:
        """
        Convert consensus metrics to a confidence multiplier.

        Strong agreement → boost (up to 1.4x)
        No agreement → reduce (down to 0.6x)
        """
        if total == 0:
            return 1.0

        ratio = agreement / total

        if ratio >= 0.75:
            # Strong consensus: boost 10-40% based on strength
            return 1.0 + 0.4 * strength
        elif ratio >= self.MIN_AGREEMENT_RATIO:
            # Moderate consensus: slight boost
            return 1.0 + 0.1 * strength
        elif ratio >= 0.25:
            # Weak consensus: slight reduction
            return 0.85
        else:
            # No consensus: strong reduction
            return 0.6

    # ── Internal: Transition Probability Tracking ─────────────────

    def _update_transitions(self, current_regime: str) -> dict:
        """
        Track regime transitions and compute forward probabilities.

        Builds a Markov-like transition matrix from observed regime sequences.
        Returns P(next_regime | current_regime).
        """
        if self._last_regime is not None and current_regime != self._last_regime:
            transition = (self._last_regime, current_regime)
            self._transition_counts[transition] += 1

        self._last_regime = current_regime
        self._regime_sequence.append(current_regime)

        # Compute P(next | current)
        transitions_from_current = {k: v for k, v in self._transition_counts.items() if k[0] == current_regime}
        total = sum(transitions_from_current.values())

        if total < 3:
            return {}

        probs = {}
        for (_, to_regime), count in transitions_from_current.items():
            probs[to_regime] = round(count / total, 3)

        # Add continuation probability (staying in same regime)
        if current_regime not in probs:
            # Estimate from how often we see same regime consecutively
            consecutive = 0
            for r in reversed(list(self._regime_sequence)):
                if r == current_regime:
                    consecutive += 1
                else:
                    break
            avg_duration = max(consecutive, 1)
            # Longer current streak → higher continuation probability
            probs[current_regime] = round(min(0.5 + avg_duration * 0.05, 0.9), 3)

        return probs

    # ── Serialization ─────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current multi-TF state for dashboards."""
        return {
            "timeframes": self.timeframes,
            "transition_counts": dict({f"{k[0]}->{k[1]}": v for k, v in self._transition_counts.items()}),
            "regime_history_length": len(self._regime_sequence),
            "last_regime": self._last_regime,
        }
