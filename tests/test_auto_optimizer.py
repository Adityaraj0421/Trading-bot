"""
Unit tests for auto_optimizer.py — HyperparamBound, TrialResult, AutoOptimizer.

Tests parameter sampling, scoring formula, Pareto front maintenance,
trial recording, and state serialization.
"""

import random
from datetime import datetime

import pytest

from auto_optimizer import (
    DEFAULT_SEARCH_SPACE,
    AutoOptimizer,
    HyperparamBound,
    TrialResult,
)


@pytest.fixture()
def optimizer():
    return AutoOptimizer(max_trials=50)


def make_metrics(sharpe=1.0, ret=5.0, drawdown=3.0, trades=10):
    return {
        "sharpe_ratio": sharpe,
        "total_return_pct": ret,
        "max_drawdown_pct": -drawdown,
        "total_trades": trades,
    }


def make_params():
    """Return realistic params matching DEFAULT_SEARCH_SPACE keys.

    v9.0: record_result() feeds back to Optuna, which validates
    that params keys match the study's search space distributions.
    """
    return {
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "trailing_stop_pct": 1.5,
        "confidence_threshold": 0.6,
        "lookback_bars": 200,
        "max_hold_bars": 80,
        "position_size_pct": 5.0,
        "max_open_positions": 3,
        "retrain_hours": 6.0,
    }


# ---------------------------------------------------------------------------
# HyperparamBound
# ---------------------------------------------------------------------------


class TestHyperparamBound:
    def test_sample_within_bounds(self):
        bound = HyperparamBound("test", low=1.0, high=10.0, dtype=float)
        random.seed(42)
        for _ in range(100):
            val = bound.sample()
            assert 1.0 <= val <= 10.0

    def test_sample_int_dtype(self):
        bound = HyperparamBound("test", low=1.0, high=10.0, dtype=int)
        random.seed(42)
        val = bound.sample()
        assert isinstance(val, int)

    def test_sample_float_dtype(self):
        bound = HyperparamBound("test", low=0.0, high=1.0, dtype=float)
        random.seed(42)
        val = bound.sample()
        assert isinstance(val, float)
        # Should be rounded to 4 decimals
        assert val == round(val, 4)

    def test_default_search_space_count(self):
        assert len(DEFAULT_SEARCH_SPACE) == 9


# ---------------------------------------------------------------------------
# TrialResult
# ---------------------------------------------------------------------------


class TestTrialResult:
    def test_to_dict_keys(self):
        tr = TrialResult(params={"a": 1}, metrics={}, score=2.0, sharpe=1.0)
        d = tr.to_dict()
        expected_keys = {"params", "total_return", "sharpe", "max_drawdown", "total_trades", "score", "timestamp"}
        assert set(d.keys()) == expected_keys

    def test_timestamp_iso_format(self):
        tr = TrialResult(params={}, metrics={})
        d = tr.to_dict()
        # Should parse without error
        datetime.fromisoformat(d["timestamp"])


# ---------------------------------------------------------------------------
# suggest_params
# ---------------------------------------------------------------------------


class TestSuggestParams:
    def test_returns_all_keys(self, optimizer):
        params = optimizer.suggest_params()
        assert set(params.keys()) == set(DEFAULT_SEARCH_SPACE.keys())

    def test_values_within_bounds(self, optimizer):
        random.seed(42)
        params = optimizer.suggest_params()
        for name, bound in DEFAULT_SEARCH_SPACE.items():
            assert bound.low <= params[name] <= bound.high, f"{name} out of bounds"

    def test_different_per_call(self, optimizer):
        random.seed(42)
        p1 = optimizer.suggest_params()
        p2 = optimizer.suggest_params()
        assert p1 != p2


# ---------------------------------------------------------------------------
# suggest_nearby
# ---------------------------------------------------------------------------


class TestSuggestNearby:
    def test_within_bounds(self, optimizer):
        random.seed(42)
        base = optimizer.suggest_params()
        nearby = optimizer.suggest_nearby(base)
        for name, bound in DEFAULT_SEARCH_SPACE.items():
            assert bound.low <= nearby[name] <= bound.high, f"{name} out of bounds"

    def test_respects_dtype(self, optimizer):
        random.seed(42)
        base = optimizer.suggest_params()
        nearby = optimizer.suggest_nearby(base)
        for name, bound in DEFAULT_SEARCH_SPACE.items():
            if bound.dtype == int:
                assert isinstance(nearby[name], int), f"{name} should be int"


# ---------------------------------------------------------------------------
# _compute_score
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_sharpe_base(self, optimizer):
        tr = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=-1.0, max_drawdown=3.0, total_trades=10)
        score = optimizer._compute_score(tr)
        # base = 2.0 * 2.0 = 4.0, no return bonus, no drawdown penalty,
        # trades=10 is NOT > 10 so no trade bonus
        assert score == pytest.approx(4.0, abs=0.01)

    def test_positive_return_bonus(self, optimizer):
        tr = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=10.0, max_drawdown=3.0, total_trades=10)
        score = optimizer._compute_score(tr)
        # base = 4.0, return bonus = 10*0.1 = 1.0, trades=10 (no trade bonus) → 5.0
        assert score == pytest.approx(5.0, abs=0.01)

    def test_no_bonus_negative_return(self, optimizer):
        tr1 = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=-5.0, max_drawdown=3.0, total_trades=10)
        tr2 = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=0.0, max_drawdown=3.0, total_trades=10)
        # Both should have no return bonus
        assert optimizer._compute_score(tr1) == optimizer._compute_score(tr2)

    def test_drawdown_penalty(self, optimizer):
        tr = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=0.0, max_drawdown=10.0, total_trades=10)
        score = optimizer._compute_score(tr)
        # base = 4.0, drawdown penalty = -(10-5)*0.3 = -1.5, no trade bonus → 2.5
        assert score == pytest.approx(2.5, abs=0.01)

    def test_low_trade_penalty(self, optimizer):
        tr = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=0.0, max_drawdown=3.0, total_trades=3)
        score = optimizer._compute_score(tr)
        # base = 4.0, * 0.5 = 2.0 (no other bonuses/penalties)
        assert score == pytest.approx(2.0, abs=0.01)

    def test_high_trade_bonus(self, optimizer):
        tr = TrialResult(params={}, metrics={}, sharpe=2.0, total_return=0.0, max_drawdown=3.0, total_trades=15)
        score = optimizer._compute_score(tr)
        # base = 4.0 + 0.5 (trade bonus) = 4.5
        assert score == pytest.approx(4.5, abs=0.01)


# ---------------------------------------------------------------------------
# record_result
# ---------------------------------------------------------------------------


class TestRecordResult:
    def test_appends_to_trials(self, optimizer):
        assert len(optimizer.trials) == 0
        optimizer.record_result(make_params(), make_metrics())
        assert len(optimizer.trials) == 1

    def test_updates_best(self, optimizer):
        optimizer.record_result(make_params(), make_metrics(sharpe=1.0))
        first_score = optimizer.best_result.score
        p2 = make_params()
        p2["stop_loss_pct"] = 3.0  # Vary so Optuna sees a different trial
        optimizer.record_result(p2, make_metrics(sharpe=5.0))
        assert optimizer.best_result.score > first_score

    def test_updates_pareto(self, optimizer):
        optimizer.record_result(make_params(), make_metrics(sharpe=2.0, trades=15, drawdown=3.0))
        assert len(optimizer.pareto_front) >= 1


# ---------------------------------------------------------------------------
# _update_pareto
# ---------------------------------------------------------------------------


class TestUpdatePareto:
    def test_non_dominated_added(self, optimizer):
        # Good sharpe, bad drawdown
        tr1 = TrialResult(params={}, metrics={}, sharpe=3.0, max_drawdown=10.0, total_trades=10)
        optimizer._update_pareto(tr1)
        # Good drawdown, bad sharpe
        tr2 = TrialResult(params={}, metrics={}, sharpe=1.0, max_drawdown=2.0, total_trades=10)
        optimizer._update_pareto(tr2)
        # Both non-dominated → both should be in front
        assert len(optimizer.pareto_front) == 2

    def test_dominated_rejected(self, optimizer):
        # Superior result first
        tr1 = TrialResult(params={}, metrics={}, sharpe=3.0, max_drawdown=2.0, total_trades=15)
        optimizer._update_pareto(tr1)
        # Dominated: worse on all objectives
        tr2 = TrialResult(params={}, metrics={}, sharpe=1.0, max_drawdown=5.0, total_trades=5)
        optimizer._update_pareto(tr2)
        assert len(optimizer.pareto_front) == 1

    def test_dominating_removes_existing(self, optimizer):
        # Inferior result first
        tr1 = TrialResult(params={}, metrics={}, sharpe=1.0, max_drawdown=5.0, total_trades=5)
        optimizer._update_pareto(tr1)
        assert len(optimizer.pareto_front) == 1
        # Superior on all objectives
        tr2 = TrialResult(params={}, metrics={}, sharpe=3.0, max_drawdown=2.0, total_trades=15)
        optimizer._update_pareto(tr2)
        assert len(optimizer.pareto_front) == 1
        assert optimizer.pareto_front[0].sharpe == 3.0


# ---------------------------------------------------------------------------
# Serialization & status
# ---------------------------------------------------------------------------


class TestSerializationAndStatus:
    def test_to_dict_structure(self, optimizer):
        optimizer.record_result(make_params(), make_metrics())
        d = optimizer.to_dict()
        # v9.0: to_dict() now includes "engine" key (optuna/random)
        assert "engine" in d
        assert "optimization_count" in d
        assert "best_result" in d
        assert "pareto_front" in d
        assert "trials_count" in d

    def test_from_dict_restores_best(self, optimizer):
        optimizer.record_result(make_params(), make_metrics(sharpe=3.0))
        original_score = optimizer.best_result.score
        saved = optimizer.to_dict()

        new_opt = AutoOptimizer()
        new_opt.from_dict(saved)
        assert new_opt.best_result.score == original_score
