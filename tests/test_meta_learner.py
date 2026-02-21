"""
Unit tests for meta_learner.py — MetaConfig, TradeObservation, MetaLearner.

Tests online learning of signal weights, position sizing (Kelly criterion),
retraining frequency, regime adjustments, confidence thresholds, and
serialization.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from meta_learner import MetaConfig, TradeObservation, MetaLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_observations(ml: MetaLearner, n: int, *,
                      pnl: float = 10.0,
                      strat_signal: str = "BUY",
                      ml_signal: str = "BUY",
                      final_signal: str = "BUY",
                      strat_conf: float = 0.7,
                      ml_conf: float = 0.7,
                      regime: str = "trending_up"):
    """Inject n identical observations into the MetaLearner."""
    for _ in range(n):
        ml.observe_trade(
            pnl=pnl,
            strategy_signal=strat_signal,
            ml_signal=ml_signal,
            final_signal=final_signal,
            strategy_confidence=strat_conf,
            ml_confidence=ml_conf,
            regime=regime,
        )


# ---------------------------------------------------------------------------
# MetaConfig
# ---------------------------------------------------------------------------

class TestMetaConfig:
    def test_defaults(self):
        cfg = MetaConfig()
        assert cfg.strategy_weight == 0.6
        assert cfg.ml_weight == 0.4
        assert cfg.position_size_method == "fixed"

    def test_to_dict_shape(self):
        cfg = MetaConfig()
        d = cfg.to_dict()
        assert d["strategy_weight"] == 0.6
        assert d["ml_weight"] == 0.4
        assert "kelly_fraction" in d
        assert "regime_adjustments" in d


# ---------------------------------------------------------------------------
# TradeObservation
# ---------------------------------------------------------------------------

class TestTradeObservation:
    def test_to_dict(self):
        obs = TradeObservation(
            pnl=50.0, signal_source="both",
            strategy_signal="BUY", ml_signal="BUY",
            final_signal="BUY", strategy_confidence=0.8,
            ml_confidence=0.7, regime="trending_up",
        )
        d = obs.to_dict()
        assert d["pnl"] == 50.0
        assert d["signal_source"] == "both"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# MetaLearner — observe_trade
# ---------------------------------------------------------------------------

class TestObserveTrade:
    def test_observation_appended(self):
        ml = MetaLearner()
        ml.observe_trade(10, "BUY", "BUY", "BUY", 0.7, 0.7, "trending_up")
        assert len(ml.observations) == 1

    def test_signal_source_both(self):
        ml = MetaLearner()
        ml.observe_trade(10, "BUY", "BUY", "BUY", 0.7, 0.7, "trending_up")
        assert ml.observations[0].signal_source == "both"

    def test_signal_source_strategy(self):
        ml = MetaLearner()
        ml.observe_trade(10, "BUY", "SELL", "BUY", 0.7, 0.5, "ranging")
        assert ml.observations[0].signal_source == "strategy"

    def test_signal_source_ml(self):
        ml = MetaLearner()
        ml.observe_trade(10, "BUY", "SELL", "SELL", 0.5, 0.8, "ranging")
        assert ml.observations[0].signal_source == "ml"

    def test_regime_observations_tracked(self):
        ml = MetaLearner()
        _add_observations(ml, 5, regime="trending_up")
        _add_observations(ml, 3, regime="ranging")
        assert len(ml._regime_observations["trending_up"]) == 5
        assert len(ml._regime_observations["ranging"]) == 3

    def test_observation_window_bounded(self):
        ml = MetaLearner(window_size=20)
        _add_observations(ml, 30)
        assert len(ml.observations) == 20


# ---------------------------------------------------------------------------
# MetaLearner — learn (top-level)
# ---------------------------------------------------------------------------

class TestLearn:
    def test_insufficient_data(self):
        ml = MetaLearner()
        _add_observations(ml, 5)
        result = ml.learn()
        assert result["status"] == "insufficient_data"

    def test_increments_learning_count(self):
        ml = MetaLearner()
        _add_observations(ml, 15, strat_signal="BUY", ml_signal="SELL",
                          final_signal="BUY", pnl=10.0)
        ml.learn()
        assert ml.learning_count == 1
        assert ml.last_learn_time is not None

    def test_returns_learned_status(self):
        ml = MetaLearner()
        _add_observations(ml, 15, strat_signal="BUY", ml_signal="SELL",
                          final_signal="BUY", pnl=10.0)
        # v2.0: learn() starts A/B experiment, use learn_immediate() for v1.0 compat
        result = ml.learn_immediate()
        assert result["status"] == "learned"
        assert "round" in result


# ---------------------------------------------------------------------------
# _learn_signal_weights
# ---------------------------------------------------------------------------

class TestLearnSignalWeights:
    def test_strategy_wins_get_higher_weight(self):
        ml = MetaLearner()
        # Strategy-sourced wins
        _add_observations(ml, 10, strat_signal="BUY", ml_signal="SELL",
                          final_signal="BUY", pnl=50.0)
        # ML-sourced losses
        _add_observations(ml, 10, strat_signal="SELL", ml_signal="BUY",
                          final_signal="BUY", pnl=-30.0)
        initial_sw = ml.config.strategy_weight
        ml._learn_signal_weights_proposed(ml.config)
        assert ml.config.strategy_weight > initial_sw

    def test_weights_sum_to_one(self):
        ml = MetaLearner()
        _add_observations(ml, 15, strat_signal="BUY", ml_signal="SELL",
                          final_signal="BUY", pnl=10.0)
        ml._learn_signal_weights_proposed(ml.config)
        assert ml.config.strategy_weight + ml.config.ml_weight == pytest.approx(1.0)

    def test_weights_bounded(self):
        ml = MetaLearner()
        _add_observations(ml, 15, strat_signal="BUY", ml_signal="SELL",
                          final_signal="BUY", pnl=100.0)
        ml._learn_signal_weights_proposed(ml.config)
        assert 0.2 <= ml.config.strategy_weight <= 0.8
        assert 0.2 <= ml.config.ml_weight <= 0.8

    def test_agreement_bonus_increases(self):
        ml = MetaLearner()
        # "both" trades very profitable (must be > max of strat_avg, ml_avg)
        _add_observations(ml, 10, strat_signal="BUY", ml_signal="BUY",
                          final_signal="BUY", pnl=100.0)
        # Strategy-only trades less profitable
        _add_observations(ml, 5, strat_signal="BUY", ml_signal="SELL",
                          final_signal="BUY", pnl=10.0)
        # ML-only trades less profitable
        _add_observations(ml, 5, strat_signal="SELL", ml_signal="BUY",
                          final_signal="BUY", pnl=10.0)
        initial_bonus = ml.config.agreement_bonus
        ml._learn_signal_weights_proposed(ml.config)
        assert ml.config.agreement_bonus >= initial_bonus

    def test_insufficient_data_returns_none(self):
        ml = MetaLearner()
        # Only "both" observations (neither strat-only nor ml-only have >= 3)
        _add_observations(ml, 5, strat_signal="BUY", ml_signal="BUY",
                          final_signal="BUY", pnl=10.0)
        result = ml._learn_signal_weights_proposed(ml.config)
        assert result is None


# ---------------------------------------------------------------------------
# _learn_position_sizing
# ---------------------------------------------------------------------------

class TestLearnPositionSizing:
    def test_insufficient_data(self):
        ml = MetaLearner()
        _add_observations(ml, 10)
        result = ml._learn_position_sizing_proposed(ml.config)
        assert result is None

    def test_switches_to_kelly_with_edge(self):
        ml = MetaLearner()
        # 80% win rate, 2:1 reward/risk → strong Kelly edge
        _add_observations(ml, 16, pnl=20.0)   # wins
        _add_observations(ml, 4, pnl=-10.0)    # losses
        result = ml._learn_position_sizing_proposed(ml.config)
        assert result is not None
        assert result["method"] == "kelly"
        assert ml.config.kelly_fraction > 0

    def test_stays_fixed_without_edge(self):
        ml = MetaLearner()
        # 50% win rate, 1:1 reward/risk → no edge
        _add_observations(ml, 10, pnl=10.0)
        _add_observations(ml, 10, pnl=-10.0)
        result = ml._learn_position_sizing_proposed(ml.config)
        assert result["method"] == "fixed"

    def test_kelly_fraction_capped(self):
        ml = MetaLearner()
        # Extreme edge
        _add_observations(ml, 20, pnl=100.0)
        ml._learn_position_sizing_proposed(ml.config)
        # Half-Kelly caps at 0.5 * 0.5 = 0.25
        assert ml.config.kelly_fraction <= 0.25


# ---------------------------------------------------------------------------
# _learn_retraining_frequency
# ---------------------------------------------------------------------------

class TestLearnRetrainingFrequency:
    def test_insufficient_drift_events(self):
        ml = MetaLearner()
        result = ml._learn_retraining_frequency()
        assert result is None

    def test_adjusts_retrain_hours(self):
        ml = MetaLearner()
        # Simulate drift events 4 hours apart
        now = datetime.now()
        ml._drift_events.append({"timestamp": now - timedelta(hours=8)})
        ml._drift_events.append({"timestamp": now - timedelta(hours=4)})
        ml._drift_events.append({"timestamp": now})
        result = ml._learn_retraining_frequency()
        assert result is not None
        assert "retrain_hours" in result
        assert 1 <= ml.config.retrain_hours <= 24

    def test_record_drift_event(self):
        ml = MetaLearner()
        ml.record_drift_event()
        assert len(ml._drift_events) == 1


# ---------------------------------------------------------------------------
# _learn_regime_adjustments
# ---------------------------------------------------------------------------

class TestLearnRegimeAdjustments:
    def test_insufficient_regime_data(self):
        ml = MetaLearner()
        _add_observations(ml, 3, regime="trending_up")
        result = ml._learn_regime_adjustments_proposed(ml.config)
        assert result is None

    def test_learns_strategy_bias(self):
        ml = MetaLearner()
        # Strategy outperforms ML in this regime
        for _ in range(5):
            ml.observe_trade(pnl=50.0, strategy_signal="BUY", ml_signal="SELL",
                             final_signal="BUY", strategy_confidence=0.8,
                             ml_confidence=0.5, regime="trending_up")
        for _ in range(5):
            ml.observe_trade(pnl=-20.0, strategy_signal="SELL", ml_signal="BUY",
                             final_signal="BUY", strategy_confidence=0.5,
                             ml_confidence=0.8, regime="trending_up")
        result = ml._learn_regime_adjustments_proposed(ml.config)
        assert result is not None
        assert "trending_up" in result
        assert result["trending_up"]["bias"] == "strategy"


# ---------------------------------------------------------------------------
# _learn_confidence_threshold
# ---------------------------------------------------------------------------

class TestLearnConfidenceThreshold:
    def test_insufficient_data(self):
        ml = MetaLearner()
        _add_observations(ml, 10)
        result = ml._learn_confidence_threshold_proposed(ml.config)
        assert result is None

    def test_raises_threshold_when_low_conf_loses(self):
        ml = MetaLearner()
        # Low confidence trades lose money
        _add_observations(ml, 10, pnl=-20.0, strat_conf=0.3, ml_conf=0.3)
        # High confidence trades make money
        _add_observations(ml, 10, pnl=50.0, strat_conf=0.8, ml_conf=0.8)
        old_threshold = ml.config.min_confidence
        ml._learn_confidence_threshold_proposed(ml.config)
        # Threshold should move toward the profitable bucket (0.8)
        assert ml.config.min_confidence >= old_threshold

    def test_threshold_bounded(self):
        ml = MetaLearner()
        _add_observations(ml, 20, pnl=10.0, strat_conf=0.9, ml_conf=0.9)
        ml._learn_confidence_threshold_proposed(ml.config)
        assert 0.3 <= ml.config.min_confidence <= 0.8


# ---------------------------------------------------------------------------
# get_signal_weights
# ---------------------------------------------------------------------------

class TestGetSignalWeights:
    def test_default_weights(self):
        ml = MetaLearner()
        sw, mw = ml.get_signal_weights()
        assert sw == 0.6
        assert mw == 0.4

    def test_regime_adjustment_boosts_strategy(self):
        ml = MetaLearner()
        ml.config.regime_adjustments = {
            "trending_up": {"bias": "strategy"}
        }
        sw, mw = ml.get_signal_weights(regime="trending_up")
        assert sw > 0.6
        assert sw + mw == pytest.approx(1.0)

    def test_regime_adjustment_boosts_ml(self):
        ml = MetaLearner()
        ml.config.regime_adjustments = {
            "ranging": {"bias": "ml"}
        }
        sw, mw = ml.get_signal_weights(regime="ranging")
        assert mw > 0.4
        assert sw + mw == pytest.approx(1.0)

    def test_unknown_regime_returns_defaults(self):
        ml = MetaLearner()
        sw, mw = ml.get_signal_weights(regime="unknown_regime")
        assert sw == 0.6
        assert mw == 0.4

    def test_weights_bounded_after_adjustment(self):
        ml = MetaLearner()
        ml.config.strategy_weight = 0.8  # Already at max
        ml.config.ml_weight = 0.2
        ml.config.regime_adjustments = {"trending_up": {"bias": "strategy"}}
        sw, mw = ml.get_signal_weights(regime="trending_up")
        assert sw <= 0.8


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_get_status_shape(self):
        ml = MetaLearner()
        status = ml.get_status()
        assert "config" in status
        assert "observations" in status
        assert "learning_rounds" in status
        assert "drift_events" in status
        assert "regime_data" in status

    def test_to_dict_includes_recent_observations(self):
        ml = MetaLearner()
        _add_observations(ml, 5)
        data = ml.to_dict()
        assert len(data["observations"]) == 5
        assert data["learning_count"] == 0

    def test_from_dict_restores_config(self):
        ml = MetaLearner()
        ml.config.strategy_weight = 0.75
        ml.config.ml_weight = 0.25
        ml.config.position_size_method = "kelly"
        ml.config.kelly_fraction = 0.15
        ml.learning_count = 5
        data = ml.to_dict()

        ml2 = MetaLearner()
        ml2.from_dict(data)
        assert ml2.config.strategy_weight == 0.75
        assert ml2.config.ml_weight == 0.25
        assert ml2.config.position_size_method == "kelly"
        assert ml2.config.kelly_fraction == 0.15
        assert ml2.learning_count == 5

    def test_from_dict_handles_empty(self):
        ml = MetaLearner()
        ml.from_dict({})
        assert ml.learning_count == 0
        assert ml.config.strategy_weight == 0.6
