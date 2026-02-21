"""
Tests for RLEnsemble — multi-agent RL voting, feature extraction, serialization.
Works with both PyTorch (DQN+PPO) and tabular Q-learning fallback.
"""

import numpy as np
import pandas as pd
import pytest
from rl_ensemble import (
    RLEnsemble, QLearnerAgent, ReplayBuffer, ACTIONS, FEATURE_NAMES, STATE_DIM,
)


@pytest.fixture()
def ensemble():
    return RLEnsemble()


@pytest.fixture()
def sample_df_ind():
    """Minimal DataFrame with indicator columns for feature extraction."""
    np.random.seed(42)
    n = 10
    data = {
        "close": np.random.uniform(49000, 51000, n),
        "rsi": np.random.uniform(30, 70, n),
        "macd_hist": np.random.randn(n) * 0.01,
        "returns_5": np.random.randn(n) * 0.02,
        "volume_ratio": np.random.uniform(0.5, 2.0, n),
        "adx": np.random.uniform(15, 45, n),
        "bb_position": np.random.uniform(0, 1, n),
        "close_to_vwap": np.random.randn(n) * 0.01,
        "stoch_k": np.random.uniform(10, 90, n),
        "obv_divergence": np.random.choice([-1, 0, 1], n).astype(float),
        "ema_cross": np.random.choice([-1, 0, 1], n).astype(float),
        "rolling_vol_10": np.random.uniform(0.005, 0.05, n),
    }
    return pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=n, freq="h"))


# ── Ensemble Predict ──────────────────────────────────────────────


class TestEnsemblePredict:
    def test_returns_valid_signal(self, ensemble, sample_df_ind):
        signal, conf = ensemble.predict(sample_df_ind, "trending_up")
        assert signal in ACTIONS
        assert 0 <= conf <= 1.0

    def test_predict_with_different_regimes(self, ensemble, sample_df_ind):
        for regime in ["ranging", "trending_up", "trending_down", "high_volatility"]:
            signal, conf = ensemble.predict(sample_df_ind, regime)
            assert signal in ACTIONS

    def test_hold_default_for_low_confidence(self, ensemble, sample_df_ind):
        """With random initial weights, votes may be low."""
        signal, conf = ensemble.predict(sample_df_ind, "ranging")
        # Just verify it returns something valid
        assert signal in ACTIONS


# ── Feature Extraction ────────────────────────────────────────────


class TestFeatureExtraction:
    def test_extract_all_features(self, ensemble, sample_df_ind):
        features = ensemble._extract_features(sample_df_ind, "trending_up")
        assert len(features) == STATE_DIM
        for name in FEATURE_NAMES:
            assert name in features

    def test_regime_encoding(self, ensemble):
        assert ensemble._encode_regime("ranging") == 0
        assert ensemble._encode_regime("trending_up") == 1
        assert ensemble._encode_regime("trending_down") == 2
        assert ensemble._encode_regime("high_volatility") == 3
        assert ensemble._encode_regime("unknown_regime") == 0


# ── Update Reward ─────────────────────────────────────────────────


class TestUpdateReward:
    def test_update_without_prior_predict(self, ensemble, sample_df_ind):
        """update_reward should be safe even if predict wasn't called."""
        ensemble.update_reward(reward=0.05)  # No crash

    def test_update_with_prior_predict(self, ensemble, sample_df_ind):
        ensemble.predict(sample_df_ind, "ranging")
        ensemble.update_reward(reward=0.05, next_df_ind=sample_df_ind,
                               regime="ranging")
        # Agents should have recorded the trade
        for agent in ensemble.agents:
            assert agent.total_trades >= 1

    def test_multiple_predict_update_cycles(self, ensemble, sample_df_ind):
        for _ in range(5):
            ensemble.predict(sample_df_ind, "trending_up")
            ensemble.update_reward(reward=np.random.randn() * 0.05,
                                   next_df_ind=sample_df_ind,
                                   regime="trending_up")
        for agent in ensemble.agents:
            assert agent.total_trades == 5


# ── Stats ─────────────────────────────────────────────────────────


class TestStats:
    def test_stats_structure(self, ensemble):
        stats = ensemble.get_stats()
        assert "engine" in stats
        # Should have entries for each agent
        for agent in ensemble.agents:
            key = f"agent_{agent.agent_id}"
            assert key in stats

    def test_stats_after_trading(self, ensemble, sample_df_ind):
        ensemble.predict(sample_df_ind, "ranging")
        ensemble.update_reward(0.05, sample_df_ind, "ranging")
        stats = ensemble.get_stats()
        for agent in ensemble.agents:
            key = f"agent_{agent.agent_id}"
            assert stats[key]["total_trades"] >= 1


# ── Serialization ─────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_structure(self, ensemble):
        data = ensemble.to_dict()
        assert "version" in data
        assert "engine" in data

    def test_roundtrip_serialization(self, ensemble, sample_df_ind):
        # Do some trading first
        for _ in range(3):
            ensemble.predict(sample_df_ind, "ranging")
            ensemble.update_reward(0.02, sample_df_ind)

        saved = ensemble.to_dict()

        # Restore into new ensemble
        new_ensemble = RLEnsemble()
        new_ensemble.from_dict(saved)

        # Verify state was restored
        for i, agent in enumerate(new_ensemble.agents):
            assert agent.total_trades == ensemble.agents[i].total_trades

    def test_from_dict_empty(self, ensemble):
        """Loading empty dict should not crash."""
        ensemble.from_dict({})


# ── Tabular Q-Learner (Fallback) ─────────────────────────────────


class TestQLearnerAgent:
    def test_predict_returns_valid_action(self):
        agent = QLearnerAgent(
            agent_id=99,
            state_bins={"rsi": [30, 50, 70]},
            epsilon=0.0,  # Greedy for determinism
        )
        features = {"rsi": 45.0}
        action, conf = agent.predict(features)
        assert action in ACTIONS
        assert 0 <= conf <= 1.0

    def test_update_creates_q_entry(self):
        agent = QLearnerAgent(
            agent_id=99,
            state_bins={"rsi": [30, 50, 70]},
        )
        features = {"rsi": 45.0}
        agent.update(features, "BUY", reward=0.05)
        assert len(agent.q_table) > 0

    def test_sharpe_with_insufficient_data(self):
        agent = QLearnerAgent(agent_id=99, state_bins={"rsi": [30, 50, 70]})
        assert agent.get_sharpe() == 0.0

    def test_sharpe_with_data(self):
        agent = QLearnerAgent(agent_id=99, state_bins={"rsi": [30, 50, 70]})
        features = {"rsi": 45.0}
        for reward in [0.01, 0.02, -0.01, 0.03, 0.01,
                       0.02, -0.005, 0.015, 0.01, 0.005]:
            agent.update(features, "BUY", reward)
        sharpe = agent.get_sharpe()
        assert isinstance(sharpe, float)


# ── ReplayBuffer ──────────────────────────────────────────────────


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        state = np.zeros(4)
        buf.push(state, 0, 1.0, state, False)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        state = np.zeros(4)
        for i in range(10):
            buf.push(state, 0, float(i), state, False)
        assert len(buf) == 5

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100)
        state = np.zeros(4)
        for i in range(20):
            buf.push(state, i % 3, float(i), state, False)
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert states.shape == (5, 4)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
