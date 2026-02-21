"""
Tests for StrategyMetaSelector — DQN strategy selection with fallback.
Tests the static fallback path (always available) and serialization.
DQN tests only run if PyTorch is installed.
"""

import numpy as np
import pytest
from strategy_selector import StrategyMetaSelector, SelectionResult


STRATEGIES = ["momentum", "mean_reversion", "breakout"]
STATE_DIM = 12


@pytest.fixture()
def selector():
    return StrategyMetaSelector(strategy_names=STRATEGIES, state_dim=STATE_DIM)


@pytest.fixture()
def random_state():
    return np.random.randn(STATE_DIM).astype(np.float32)


# ── Static Fallback ───────────────────────────────────────────────


class TestStaticFallback:
    def test_initial_selection_uses_fallback(self, selector, random_state):
        result = selector.select_strategy(random_state)
        assert isinstance(result, SelectionResult)
        assert result.source == "static_fallback"

    def test_uniform_weights_initial(self, selector, random_state):
        result = selector.select_strategy(random_state)
        # Without any performance data, weights should be roughly uniform
        for w in result.strategy_weights.values():
            assert w > 0

    def test_primary_strategy_in_list(self, selector, random_state):
        result = selector.select_strategy(random_state)
        assert result.primary_strategy in STRATEGIES

    def test_confidence_between_0_and_1(self, selector, random_state):
        result = selector.select_strategy(random_state)
        assert 0 <= result.confidence <= 1.0


# ── Reward Recording ──────────────────────────────────────────────


class TestRewardRecording:
    def test_record_reward_stores_pnl(self, selector, random_state):
        selector.select_strategy(random_state)
        selector.record_reward(reward=0.05, next_state=random_state,
                               strategy_name="momentum")
        assert len(selector._strategy_pnl["momentum"]) == 1

    def test_win_rate_updated(self, selector, random_state):
        selector.select_strategy(random_state)
        selector.record_reward(reward=0.05, next_state=random_state,
                               strategy_name="momentum")
        assert selector._strategy_win_rates["momentum"] == 1.0

    def test_multiple_rewards_tracked(self, selector, random_state):
        for reward in [0.1, -0.05, 0.03, -0.02]:
            selector.select_strategy(random_state)
            selector.record_reward(reward=reward, next_state=random_state,
                                   strategy_name="momentum")
        perf = selector.get_strategy_performance()
        assert perf["momentum"]["trades"] == 4

    def test_reward_without_prior_select(self, selector, random_state):
        """record_reward before any select should be safe."""
        selector.record_reward(reward=0.1, next_state=random_state,
                               strategy_name="momentum")


# ── Performance-Based Selection ───────────────────────────────────


class TestPerformanceSelection:
    def test_after_rewards_uses_performance(self, selector, random_state):
        # Feed good performance for momentum
        for _ in range(5):
            selector.select_strategy(random_state)
            selector.record_reward(0.1, random_state, "momentum")

        # Feed bad performance for others
        for _ in range(5):
            selector.select_strategy(random_state)
            selector.record_reward(-0.1, random_state, "mean_reversion")

        result = selector.select_strategy(random_state)
        # Momentum should have higher weight (it performed well)
        assert result.strategy_weights["momentum"] > result.strategy_weights.get("mean_reversion", 0)


# ── Status ────────────────────────────────────────────────────────


class TestStatus:
    def test_status_structure(self, selector):
        status = selector.get_status()
        assert "mode" in status
        assert "epsilon" in status
        assert "buffer_size" in status
        assert "total_selections" in status
        assert "strategy_performance" in status

    def test_status_mode_fallback(self, selector):
        status = selector.get_status()
        assert status["mode"] == "static_fallback"


# ── Serialization ─────────────────────────────────────────────────


class TestSerialization:
    def test_save_state_structure(self, selector, random_state):
        selector.select_strategy(random_state)
        selector.record_reward(0.05, random_state, "momentum")
        state = selector.save_state()
        assert "strategy_names" in state
        assert "epsilon" in state
        assert "strategy_pnl" in state

    def test_load_state_roundtrip(self, selector, random_state):
        selector.select_strategy(random_state)
        selector.record_reward(0.05, random_state, "momentum")
        saved = selector.save_state()

        # Create new selector and restore
        new_selector = StrategyMetaSelector(STRATEGIES, STATE_DIM)
        new_selector.load_state(saved)
        assert new_selector._epsilon == selector._epsilon
        assert len(new_selector._strategy_pnl["momentum"]) == 1

    def test_load_empty_state_safe(self, selector):
        selector.load_state({})
