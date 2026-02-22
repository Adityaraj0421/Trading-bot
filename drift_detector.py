"""
Model Drift Detection v2.0 — Predictive Drift Detection
=========================================================
Detects model drift BEFORE performance degrades, not after.

v2.0 enhancements:
  - Feature distribution monitoring (KS test per feature vs training)
  - Confidence decay RATE tracking (gradient, not just level)
  - Regime-model mismatch detection
  - Predictive drift score (0-100, triggers at 70+)
  - Gradual model blending weights for safe transitions
  - All original methods preserved for backward compatibility

Original methods:
  1. Rolling accuracy tracker
  2. Feature distribution shift (now using proper KS test)
  3. Prediction confidence decay monitoring
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Config

_log = logging.getLogger(__name__)

try:
    from scipy.stats import ks_2samp

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class DriftReport:
    """Comprehensive drift analysis report."""

    drift_detected: bool
    predictive_drift_score: float  # 0-100, higher = more likely to drift
    current_accuracy: float
    accuracy_drop: float
    confidence_decay: float
    confidence_slope: float  # Gradient of confidence over time
    feature_drift_pct: float  # % of features with significant distribution shift
    regime_mismatch: bool  # Model trained on wrong regime
    reason: str
    resolved_samples: int
    drift_count: int
    blend_weight: float  # Suggested weight for new model (0-1)
    feature_alerts: list = field(default_factory=list)


# ── Predictive Drift Detector ─────────────────────────────────────


class DriftDetector:
    """
    v2.0: Predictive drift detection that triggers retraining
    BEFORE accuracy drops, based on feature distribution shifts,
    confidence decay gradients, and regime-model mismatch.
    """

    # Predictive drift thresholds
    PREDICTIVE_DRIFT_THRESHOLD = 70  # Score 0-100
    KS_SIGNIFICANCE = 0.05  # p-value threshold for KS test
    CONFIDENCE_SLOPE_ALERT = -0.005  # Slope below this = decaying fast
    REGIME_MISMATCH_THRESHOLD = 0.3  # 30% mismatch between train/live regimes

    # Model blending
    BLEND_INITIAL = 0.2  # New model starts at 20% weight
    BLEND_INCREMENT = 0.1  # Increase by 10% each validation cycle
    BLEND_MAX = 1.0

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.predictions: deque[str] = deque(maxlen=window_size)
        self.actuals: deque[str | None] = deque(maxlen=window_size)
        self.confidences: deque[float] = deque(maxlen=window_size * 2)
        self.baseline_accuracy: float = 0.0
        self.baseline_confidence: float = 0.0
        self.drift_count: int = 0

        # v2.0: Feature distribution tracking
        self._training_features: np.ndarray | None = None  # Shape: (n_samples, n_features)
        self._feature_names: list[str] = []
        self._recent_features: deque = deque(maxlen=200)

        # v2.0: Confidence slope tracking
        self._confidence_timestamps: deque = deque(maxlen=100)
        self._confidence_index: int = 0

        # v2.0: Regime tracking
        self._training_regime_dist: dict = {}  # {regime: fraction}
        self._live_regime_counts: dict = {}

        # v2.0: Model blending state
        self._blend_weight: float = 1.0  # 1.0 = fully trust current model
        self._new_model_validated: bool = False

    # ── Backward-Compatible Interface ─────────────────────────────

    def record_prediction(self, predicted: str, confidence: float) -> None:
        """Record a new prediction (actual outcome filled in later)."""
        self.predictions.append(predicted)
        self.confidences.append(confidence)
        self.actuals.append(None)

        # Track confidence timeline
        self._confidence_index += 1
        self._confidence_timestamps.append((self._confidence_index, confidence))

    def record_outcome(self, actual_return: float, threshold: float = 0.01) -> None:
        """Fill in the actual outcome for the most recent unresolved prediction."""
        if actual_return > threshold:
            actual = "BUY"
        elif actual_return < -threshold:
            actual = "SELL"
        else:
            actual = "HOLD"

        for i in range(len(self.actuals) - 1, -1, -1):
            if self.actuals[i] is None:
                self.actuals[i] = actual
                break

    def set_baseline(self, accuracy: float, avg_confidence: float) -> None:
        """Set the baseline accuracy from initial training."""
        self.baseline_accuracy = accuracy
        self.baseline_confidence = avg_confidence

    def check_drift(self) -> dict[str, Any]:
        """
        Backward-compatible: Check for model drift.
        Returns dict with original fields + new v2.0 fields.
        """
        report = self.check_drift_predictive()
        return {
            "drift_detected": report.drift_detected,
            "current_accuracy": report.current_accuracy,
            "accuracy_drop": report.accuracy_drop,
            "confidence_decay": report.confidence_decay,
            "reason": report.reason,
            "resolved_samples": report.resolved_samples,
            "drift_count": report.drift_count,
            # New v2.0 fields
            "predictive_drift_score": report.predictive_drift_score,
            "confidence_slope": report.confidence_slope,
            "feature_drift_pct": report.feature_drift_pct,
            "regime_mismatch": report.regime_mismatch,
            "blend_weight": report.blend_weight,
            "feature_alerts": report.feature_alerts,
        }

    def reset(self) -> None:
        """Reset after retraining."""
        self.predictions.clear()
        self.actuals.clear()
        self.confidences.clear()
        self._confidence_timestamps.clear()
        self._confidence_index = 0
        self._recent_features.clear()
        self._live_regime_counts.clear()

    # ── v2.0: Enhanced Interface ──────────────────────────────────

    def set_training_features(self, features: np.ndarray, feature_names: list[str]) -> None:
        """
        Store training feature distributions for KS testing.
        Call this after model training with the scaled feature matrix.
        """
        self._training_features = features.copy()
        self._feature_names = feature_names

    def set_training_regime_dist(self, regime_distribution: dict[str, float]) -> None:
        """
        Store the regime distribution during training.
        e.g., {"trending_up": 0.3, "ranging": 0.5, "trending_down": 0.2}
        """
        self._training_regime_dist = regime_distribution

    def record_features(self, feature_vector: np.ndarray) -> None:
        """Record a single feature vector for distribution monitoring."""
        self._recent_features.append(feature_vector)

    def record_live_regime(self, regime: str) -> None:
        """Track what regime the market is in during live trading."""
        self._live_regime_counts[regime] = self._live_regime_counts.get(regime, 0) + 1

    def check_drift_predictive(self) -> DriftReport:
        """
        v2.0: Full predictive drift analysis.

        Computes a drift score (0-100) from multiple signals:
        - Accuracy degradation (reactive, 25 points)
        - Confidence decay rate (early warning, 25 points)
        - Feature distribution shift (predictive, 30 points)
        - Regime mismatch (predictive, 20 points)
        """
        # ── Component 1: Accuracy (reactive) ─────────────────────
        resolved = [(p, a) for p, a in zip(self.predictions, self.actuals) if a is not None]
        n_resolved = len(resolved)

        if n_resolved >= self.window_size // 2:
            correct = sum(1 for p, a in resolved if p == a)
            current_acc = correct / n_resolved
            acc_drop = self.baseline_accuracy - current_acc
        else:
            current_acc = self.baseline_accuracy
            acc_drop = 0.0

        # Accuracy score: 0-25 points
        if acc_drop > Config.DRIFT_THRESHOLD:
            acc_score = 25
        elif acc_drop > Config.DRIFT_THRESHOLD * 0.5:
            acc_score = 15
        elif current_acc < 0.33:
            acc_score = 25
        else:
            acc_score = max(0, acc_drop / Config.DRIFT_THRESHOLD * 15)

        # ── Component 2: Confidence slope (early warning) ─────────
        conf_slope = self._compute_confidence_slope()
        recent_conf = list(self.confidences)[-20:]
        avg_recent_conf = np.mean(recent_conf) if recent_conf else self.baseline_confidence
        conf_decay = self.baseline_confidence - avg_recent_conf

        # Slope score: 0-25 points
        if conf_slope < self.CONFIDENCE_SLOPE_ALERT:
            slope_score = 25
        elif conf_slope < self.CONFIDENCE_SLOPE_ALERT * 0.5:
            slope_score = 15
        elif conf_decay > 0.15:
            slope_score = 20
        else:
            slope_score = max(0, -conf_slope / abs(self.CONFIDENCE_SLOPE_ALERT) * 10)

        # ── Component 3: Feature distribution shift (predictive) ──
        feature_drift_pct, feature_alerts = self._check_feature_drift()

        # Feature score: 0-30 points
        if feature_drift_pct > 0.5:
            feature_score = 30
        elif feature_drift_pct > 0.3:
            feature_score = 20
        elif feature_drift_pct > 0.1:
            feature_score = 10
        else:
            feature_score = feature_drift_pct * 20

        # ── Component 4: Regime mismatch (predictive) ─────────────
        regime_mismatch = self._check_regime_mismatch()

        # Regime score: 0-20 points
        regime_score = 20 if regime_mismatch else 0

        # ── Aggregate drift score ─────────────────────────────────
        drift_score = acc_score + slope_score + feature_score + regime_score
        drift_score = min(drift_score, 100)

        # Drift decision
        drift_detected = drift_score >= self.PREDICTIVE_DRIFT_THRESHOLD
        if drift_detected:
            self.drift_count += 1

        # Determine reason
        reasons = []
        if acc_score >= 15:
            reasons.append(f"accuracy_drop({current_acc:.2%} vs {self.baseline_accuracy:.2%})")
        if slope_score >= 15:
            reasons.append(f"confidence_decaying(slope={conf_slope:.4f})")
        if feature_score >= 15:
            reasons.append(f"feature_shift({feature_drift_pct:.0%} drifted)")
        if regime_mismatch:
            reasons.append("regime_mismatch")
        reason = " + ".join(reasons) if reasons else "ok"

        # Blend weight recommendation
        blend_weight = self._recommend_blend_weight(drift_score)

        return DriftReport(
            drift_detected=drift_detected,
            predictive_drift_score=round(drift_score, 1),
            current_accuracy=round(current_acc, 4),
            accuracy_drop=round(acc_drop, 4),
            confidence_decay=round(conf_decay, 4),
            confidence_slope=round(conf_slope, 6),
            feature_drift_pct=round(feature_drift_pct, 3),
            regime_mismatch=regime_mismatch,
            reason=reason,
            resolved_samples=n_resolved,
            drift_count=self.drift_count,
            blend_weight=round(blend_weight, 2),
            feature_alerts=feature_alerts,
        )

    def get_blend_weight(self) -> float:
        """
        Return current blend weight for model transitions.

        1.0 = fully trust current model
        0.0 = fully trust new model
        Use: final_prediction = blend * old_pred + (1-blend) * new_pred
        """
        return self._blend_weight

    def advance_blend(self, new_model_performing: bool) -> None:
        """
        Advance model blending after a validation cycle.

        If the new model is outperforming, increase its weight.
        Otherwise, slow down or revert.
        """
        if new_model_performing:
            self._blend_weight = max(self._blend_weight - self.BLEND_INCREMENT, 0.0)
            if self._blend_weight <= 0.0:
                self._new_model_validated = True
        else:
            # Pull back toward old model
            self._blend_weight = min(self._blend_weight + self.BLEND_INCREMENT * 0.5, 1.0)

    # ── Internal: Confidence Slope ────────────────────────────────

    def _compute_confidence_slope(self) -> float:
        """
        Compute the slope of confidence over recent predictions.

        Uses simple linear regression on (index, confidence) pairs.
        Negative slope = confidence is decaying over time.
        """
        timestamps = list(self._confidence_timestamps)
        if len(timestamps) < 10:
            return 0.0

        # Use last 50 data points
        recent = timestamps[-50:]
        x = np.array([t[0] for t in recent], dtype=float)
        y = np.array([t[1] for t in recent], dtype=float)

        # Normalize x to prevent numerical issues
        x_norm = x - x.mean()
        x_var = np.sum(x_norm**2)

        if x_var < 1e-10:
            return 0.0

        slope = np.sum(x_norm * (y - y.mean())) / x_var
        return float(slope)

    # ── Internal: Feature Distribution Monitoring ─────────────────

    def _check_feature_drift(self) -> tuple[float, list[dict]]:
        """
        Check each feature's distribution vs training distribution
        using the Kolmogorov-Smirnov test.

        Returns:
            (fraction of features drifted, list of alerts)
        """
        if self._training_features is None or len(self._recent_features) < 30:
            return 0.0, []

        if not _HAS_SCIPY:
            return self._check_feature_drift_simple()

        recent_matrix = np.array(list(self._recent_features))
        n_features = min(self._training_features.shape[1], recent_matrix.shape[1])

        drifted = 0
        alerts = []

        for i in range(n_features):
            train_col = self._training_features[:, i]
            recent_col = recent_matrix[:, i]

            # Filter out NaN/Inf
            train_valid = train_col[np.isfinite(train_col)]
            recent_valid = recent_col[np.isfinite(recent_col)]

            if len(train_valid) < 10 or len(recent_valid) < 10:
                continue

            stat, p_value = ks_2samp(train_valid, recent_valid)

            if p_value < self.KS_SIGNIFICANCE:
                drifted += 1
                fname = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                alerts.append(
                    {
                        "feature": fname,
                        "ks_statistic": round(float(stat), 4),
                        "p_value": round(float(p_value), 6),
                        "train_mean": round(float(np.mean(train_valid)), 4),
                        "recent_mean": round(float(np.mean(recent_valid)), 4),
                    }
                )

        fraction = drifted / n_features if n_features > 0 else 0.0
        # Sort by severity
        alerts.sort(key=lambda a: a["ks_statistic"], reverse=True)
        return fraction, alerts[:10]

    def _check_feature_drift_simple(self) -> tuple[float, list[dict]]:
        """
        Fallback drift check without scipy — uses mean/std comparison.
        """
        recent_matrix = np.array(list(self._recent_features))
        n_features = min(self._training_features.shape[1], recent_matrix.shape[1])

        drifted = 0
        alerts = []

        for i in range(n_features):
            train_col = self._training_features[:, i]
            recent_col = recent_matrix[:, i]

            train_valid = train_col[np.isfinite(train_col)]
            recent_valid = recent_col[np.isfinite(recent_col)]

            if len(train_valid) < 10 or len(recent_valid) < 10:
                continue

            train_mean, train_std = np.mean(train_valid), np.std(train_valid)
            recent_mean = np.mean(recent_valid)

            # Z-test for mean shift
            if train_std > 1e-10:
                z_score = abs(recent_mean - train_mean) / (train_std / np.sqrt(len(recent_valid)))
                if z_score > 3.0:
                    drifted += 1
                    fname = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                    alerts.append(
                        {
                            "feature": fname,
                            "z_score": round(float(z_score), 2),
                            "train_mean": round(float(train_mean), 4),
                            "recent_mean": round(float(recent_mean), 4),
                        }
                    )

        fraction = drifted / n_features if n_features > 0 else 0.0
        alerts.sort(key=lambda a: a.get("z_score", 0), reverse=True)
        return fraction, alerts[:10]

    # ── Internal: Regime Mismatch ─────────────────────────────────

    def _check_regime_mismatch(self) -> bool:
        """
        Check if the market regime during live trading significantly
        differs from the regime distribution during training.

        Example: Model trained on 70% ranging data but market is now
        80% trending → mismatch detected.
        """
        if not self._training_regime_dist or not self._live_regime_counts:
            return False

        total_live = sum(self._live_regime_counts.values())
        if total_live < 20:
            return False

        # Compute live distribution
        live_dist = {k: v / total_live for k, v in self._live_regime_counts.items()}

        # Compare using total variation distance (L1/2)
        all_regimes = set(list(self._training_regime_dist.keys()) + list(live_dist.keys()))
        tv_distance = 0.5 * sum(abs(self._training_regime_dist.get(r, 0) - live_dist.get(r, 0)) for r in all_regimes)

        return tv_distance > self.REGIME_MISMATCH_THRESHOLD

    # ── Internal: Blend Weight ────────────────────────────────────

    def _recommend_blend_weight(self, drift_score: float) -> float:
        """
        Recommend model blend weight based on drift severity.

        Low drift → keep current model (blend=1.0)
        High drift → prepare for new model (blend=0.5-0.2)
        """
        if drift_score < 30:
            return 1.0
        elif drift_score < 50:
            return 0.8
        elif drift_score < 70:
            return 0.5
        elif drift_score < 90:
            return 0.3
        else:
            return self.BLEND_INITIAL  # 0.2 — almost fully switch
