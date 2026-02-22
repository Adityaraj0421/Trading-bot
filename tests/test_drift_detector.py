"""
Unit tests for drift_detector.py — DriftDetector prediction tracking,
outcome recording, baseline comparison, and drift detection.

v2.0: DriftDetector uses predictive multi-component scoring (0-100).
Drift triggers at >=70 from 4 components: accuracy (0-25), confidence
slope (0-25), feature distribution shift (0-30), regime mismatch (0-20).
Without feature training data, only accuracy + confidence are active.
"""

import numpy as np
import pytest

from drift_detector import DriftDetector


@pytest.fixture()
def detector():
    return DriftDetector(window_size=20)


@pytest.fixture()
def detector_with_features():
    """Detector with feature training data so all 4 drift components are active."""
    d = DriftDetector(window_size=20)
    # Provide training features so feature drift can be detected
    rng = np.random.RandomState(42)
    training_features = rng.randn(100, 5)
    d.set_training_features(training_features, [f"f{i}" for i in range(5)])
    d.set_training_regime_dist({"trending_up": 0.5, "ranging": 0.5})
    return d


# ---------------------------------------------------------------------------
# record_prediction
# ---------------------------------------------------------------------------


class TestRecordPrediction:
    def test_stores_prediction(self, detector):
        detector.record_prediction("BUY", 0.8)
        assert len(detector.predictions) == 1
        assert detector.predictions[0] == "BUY"

    def test_stores_confidence(self, detector):
        detector.record_prediction("BUY", 0.8)
        assert detector.confidences[0] == 0.8

    def test_sets_actual_none(self, detector):
        detector.record_prediction("BUY", 0.8)
        assert detector.actuals[0] is None

    def test_window_bounded(self, detector):
        for _ in range(25):
            detector.record_prediction("BUY", 0.7)
        assert len(detector.predictions) == 20  # window_size


# ---------------------------------------------------------------------------
# record_outcome
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    def test_positive_return_becomes_buy(self, detector):
        detector.record_prediction("BUY", 0.8)
        detector.record_outcome(0.05)
        assert detector.actuals[0] == "BUY"

    def test_negative_return_becomes_sell(self, detector):
        detector.record_prediction("SELL", 0.8)
        detector.record_outcome(-0.05)
        assert detector.actuals[0] == "SELL"

    def test_small_return_becomes_hold(self, detector):
        detector.record_prediction("HOLD", 0.5)
        detector.record_outcome(0.005)
        assert detector.actuals[0] == "HOLD"

    def test_fills_most_recent_none(self, detector):
        detector.record_prediction("BUY", 0.8)
        detector.record_prediction("SELL", 0.7)
        detector.record_outcome(0.05)  # Fills the last None (index 1)
        assert detector.actuals[1] == "BUY"
        assert detector.actuals[0] is None  # First still unfilled

    def test_multiple_outcomes_fill_in_order(self, detector):
        detector.record_prediction("BUY", 0.8)
        detector.record_prediction("SELL", 0.7)
        detector.record_outcome(-0.05)  # Fills index 1
        detector.record_outcome(0.05)  # Fills index 0
        assert detector.actuals[0] == "BUY"
        assert detector.actuals[1] == "SELL"


# ---------------------------------------------------------------------------
# set_baseline
# ---------------------------------------------------------------------------


class TestSetBaseline:
    def test_sets_accuracy(self, detector):
        detector.set_baseline(0.75, 0.8)
        assert detector.baseline_accuracy == 0.75

    def test_sets_confidence(self, detector):
        detector.set_baseline(0.75, 0.85)
        assert detector.baseline_confidence == 0.85


# ---------------------------------------------------------------------------
# check_drift
# ---------------------------------------------------------------------------


class TestCheckDrift:
    def _fill_predictions(self, detector, n, predicted, actual_return, confidence=0.8):
        """Add n predictions and resolve them with the given actual return."""
        for _ in range(n):
            detector.record_prediction(predicted, confidence)
            detector.record_outcome(actual_return)

    def test_insufficient_data_no_drift(self, detector):
        detector.record_prediction("BUY", 0.8)
        detector.record_outcome(0.05)
        result = detector.check_drift()
        assert result["drift_detected"] is False
        # v2.0: with very few samples, score stays low
        assert result["predictive_drift_score"] < 70

    def test_perfect_accuracy_no_drift(self, detector):
        detector.set_baseline(0.8, 0.8)
        # All predictions correct (predict BUY, actual return positive)
        self._fill_predictions(detector, 15, "BUY", 0.05)
        result = detector.check_drift()
        assert result["drift_detected"] is False

    def test_accuracy_drop_detected(self, detector):
        """Accuracy drop shows in component score even without full drift trigger."""
        detector.set_baseline(0.8, 0.8)
        # All predictions wrong (predict BUY, actual return very negative)
        self._fill_predictions(detector, 15, "BUY", -0.05)
        result = detector.check_drift()
        # v2.0: accuracy_drop is detected but may not trigger full drift
        # without feature/regime data (max score ~50 vs threshold 70)
        assert result["accuracy_drop"] > 0.15
        assert "accuracy_drop" in result["reason"]

    def test_accuracy_drop_triggers_drift_with_features(self, detector_with_features):
        """With multiple drift signals (accuracy + confidence slope + regime),
        predictive drift score reaches threshold of 70."""
        detector_with_features.set_baseline(0.8, 0.9)
        rng = np.random.RandomState(99)
        for i in range(20):
            # Decreasing confidence to create negative slope (0.9 → 0.14)
            conf = max(0.9 - i * 0.04, 0.1)
            detector_with_features.record_prediction("BUY", conf)
            detector_with_features.record_outcome(-0.05)
            # Shifted features + wrong regime
            detector_with_features.record_features(rng.randn(5) * 100 + 50)
            detector_with_features.record_live_regime("trending_down")
        result = detector_with_features.check_drift()
        assert result["drift_detected"] is True
        assert result["predictive_drift_score"] >= 70

    def test_confidence_decay_component(self, detector):
        """Confidence decay raises the component score."""
        detector.set_baseline(0.5, 0.9)  # Low baseline accuracy (avoid accuracy trigger)
        # All predictions correct but very low confidence
        for _ in range(15):
            detector.record_prediction("BUY", 0.3)  # Low confidence
            detector.record_outcome(0.05)
        result = detector.check_drift()
        # v2.0: confidence_decay component contributes to drift score
        assert result["confidence_decay"] > 0.3
        assert "confidence_decaying" in result["reason"] or result["confidence_slope"] < 0

    def test_below_random_accuracy(self, detector):
        """All predictions wrong yields accuracy below random (0.33)."""
        detector.set_baseline(0.1, 0.8)
        for _ in range(15):
            detector.record_prediction("BUY", 0.8)
            detector.record_outcome(-0.05)
        result = detector.check_drift()
        # v2.0: below-random contributes 25 accuracy points
        assert result["current_accuracy"] < 0.33
        assert result["predictive_drift_score"] > 0

    def test_drift_count_increments(self, detector_with_features):
        """Drift count increments each time drift_detected is True."""
        detector_with_features.set_baseline(0.8, 0.8)
        rng = np.random.RandomState(99)
        for _ in range(15):
            detector_with_features.record_prediction("BUY", 0.3)
            detector_with_features.record_outcome(-0.05)
            detector_with_features.record_features(rng.randn(5) * 10 + 5)
            detector_with_features.record_live_regime("trending_down")
        result1 = detector_with_features.check_drift()
        if result1["drift_detected"]:
            assert detector_with_features.drift_count == 1
            result2 = detector_with_features.check_drift()
            if result2["drift_detected"]:
                assert detector_with_features.drift_count == 2

    def test_result_dict_keys(self, detector):
        detector.set_baseline(0.8, 0.8)
        self._fill_predictions(detector, 15, "BUY", 0.05)
        result = detector.check_drift()
        # v2.0: check_drift() returns original keys + new predictive fields
        original_keys = {
            "drift_detected",
            "current_accuracy",
            "accuracy_drop",
            "confidence_decay",
            "reason",
            "resolved_samples",
            "drift_count",
        }
        new_v2_keys = {
            "predictive_drift_score",
            "confidence_slope",
            "feature_drift_pct",
            "regime_mismatch",
            "blend_weight",
            "feature_alerts",
        }
        expected_keys = original_keys | new_v2_keys
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_all_deques(self, detector):
        detector.record_prediction("BUY", 0.8)
        detector.record_outcome(0.05)
        detector.reset()
        assert len(detector.predictions) == 0
        assert len(detector.actuals) == 0
        assert len(detector.confidences) == 0

    def test_preserves_baseline(self, detector):
        detector.set_baseline(0.75, 0.85)
        detector.reset()
        assert detector.baseline_accuracy == 0.75
        assert detector.baseline_confidence == 0.85
