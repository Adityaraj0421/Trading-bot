"""
Reinforcement Learning Ensemble v2.0 — Deep RL
================================================
Multi-agent deep RL ensemble with DQN + PPO agents.
Replaces tabular Q-learning with neural network function approximators,
experience replay, and target networks for dramatically better performance.

Architecture:
  - Agent 1 (DQN): Momentum agent — learns from regime, RSI, MACD, returns
  - Agent 2 (DQN): Volume agent — learns from volume, ADX, BB, VWAP
  - Agent 3 (DQN): Mean-reversion agent — learns from stoch, OBV, EMA, vol
  - Agent 4 (PPO): Meta-agent — learns portfolio-level policy from all features

Research basis:
  - PPO achieves 78% returns vs 28% for LSTM-only (Coinmonks 2025)
  - Multi-agent deep RL: 142% annual returns vs 12% rule-based (NeuralArb)
  - DQN strategy selection: 120x NAV growth (Cogent Economics Dec 2025)

API preserved: predict(df_ind, regime) → (signal, confidence)
               update_reward(reward, next_df_ind, regime)
               get_stats(), to_dict(), from_dict()
"""

import ast
import logging
from typing import Any

import numpy as np
from collections import deque
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ---------- PyTorch import with graceful fallback ----------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    _log.warning("PyTorch not installed — falling back to tabular Q-learning")


@dataclass
class RLTrade:
    """Track an RL agent's trade outcome."""
    agent_id: int
    action: str   # BUY, SELL, HOLD
    reward: float
    state_key: tuple


# ============================================================
#  FEATURE NAMES — shared across all agents
# ============================================================
FEATURE_NAMES = [
    "regime_code", "rsi", "macd_hist", "returns_5",
    "volume_ratio", "adx", "bb_position", "close_to_vwap",
    "stoch_k", "obv_divergence", "ema_cross", "rolling_vol_10",
]
STATE_DIM = len(FEATURE_NAMES)
NUM_ACTIONS = 3  # BUY, SELL, HOLD
ACTIONS = ["BUY", "SELL", "HOLD"]


# ============================================================
#  Experience Replay Buffer
# ============================================================
class ReplayBuffer:
    """Experience replay with uniform sampling."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action_idx: int, reward: float,
             next_state: np.ndarray, done: float) -> None:
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================
#  Deep Q-Network (DQN) Agent
# ============================================================
if HAS_TORCH:

    class QNetwork(nn.Module):
        """Dueling DQN architecture for better value estimation."""

        def __init__(self, state_dim: int, num_actions: int, hidden: int = 64):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            # Dueling streams
            self.value_stream = nn.Linear(hidden, 1)
            self.advantage_stream = nn.Linear(hidden, num_actions)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)


    class DQNAgent:
        """
        Deep Q-Network agent with experience replay and target network.
        Uses continuous state space (no discretization) for better generalization.
        """

        def __init__(self, agent_id: int, feature_indices: list[int],
                     lr: float = 1e-3, gamma: float = 0.95, epsilon: float = 0.2,
                     tau: float = 0.005, batch_size: int = 32) -> None:
            self.agent_id = agent_id
            self.feature_indices = feature_indices
            self.state_dim = len(feature_indices)
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = 0.03
            self.epsilon_decay = 0.9995
            self.tau = tau
            self.batch_size = batch_size

            self.q_net = QNetwork(self.state_dim, NUM_ACTIONS)
            self.target_net = QNetwork(self.state_dim, NUM_ACTIONS)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

            self.replay = ReplayBuffer(capacity=10_000)
            self.trade_history: deque[float] = deque(maxlen=200)
            self.total_trades = 0
            self.train_steps = 0

        def _extract_state(self, features: dict[str, float]) -> np.ndarray:
            vals = [features.get(FEATURE_NAMES[i], 0.0) for i in self.feature_indices]
            return np.array(vals, dtype=np.float32)

        def predict(self, features: dict[str, float]) -> tuple[str, float]:
            state = self._extract_state(features)
            if np.random.random() < self.epsilon:
                return ACTIONS[np.random.randint(NUM_ACTIONS)], 0.4

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net(state_t).squeeze(0)

            action_idx = q_values.argmax().item()
            probs = F.softmax(q_values, dim=0).numpy()
            return ACTIONS[action_idx], float(probs[action_idx])

        def update(self, features: dict[str, float], action: str, reward: float,
                   next_features: dict[str, float] | None = None) -> None:
            state = self._extract_state(features)
            action_idx = ACTIONS.index(action)
            done = next_features is None
            next_state = (
                self._extract_state(next_features) if next_features
                else np.zeros(self.state_dim, dtype=np.float32)
            )

            self.replay.push(state, action_idx, reward, next_state, float(done))
            self.trade_history.append(reward)
            self.total_trades += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if len(self.replay) >= self.batch_size * 2:
                self._train_step()

        def _train_step(self) -> None:
            states, actions, rewards, next_states, dones = self.replay.sample(
                self.batch_size
            )
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions).unsqueeze(1)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones).unsqueeze(1)

            current_q = self.q_net(states_t).gather(1, actions_t)

            # Double DQN
            with torch.no_grad():
                next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
                target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

            loss = F.smooth_l1_loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            # Soft update target network
            for param, target_param in zip(
                self.q_net.parameters(), self.target_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            self.train_steps += 1

        def get_sharpe(self) -> float:
            if len(self.trade_history) < 10:
                return 0.0
            returns = np.array(self.trade_history)
            std_r = returns.std()
            if std_r < 1e-8:
                return 0.0
            return float(returns.mean() / std_r)


    # ============================================================
    #  PPO Meta-Agent
    # ============================================================

    class PPONetwork(nn.Module):
        """Actor-Critic network for PPO meta-agent."""

        def __init__(self, state_dim: int, num_actions: int, hidden: int = 64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.actor = nn.Linear(hidden, num_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            shared = self.shared(x)
            return self.actor(shared), self.critic(shared)


    class PPOAgent:
        """
        Proximal Policy Optimization meta-agent.
        Sees ALL features and learns portfolio-level policy.
        """

        def __init__(self, agent_id: int = 4, lr: float = 3e-4,
                     gamma: float = 0.99, clip_eps: float = 0.2,
                     epochs: int = 4, batch_size: int = 32) -> None:
            self.agent_id = agent_id
            self.gamma = gamma
            self.clip_eps = clip_eps
            self.epochs = epochs
            self.batch_size = batch_size
            self.epsilon = 0.1
            self.epsilon_min = 0.02
            self.epsilon_decay = 0.9998

            self.network = PPONetwork(STATE_DIM, NUM_ACTIONS)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

            self.trajectory: list[dict] = []
            self.trade_history: deque[float] = deque(maxlen=200)
            self.total_trades = 0
            self.train_steps = 0

        def predict(self, features: dict[str, float]) -> tuple[str, float]:
            state = np.array(
                [features.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32
            )
            if np.random.random() < self.epsilon:
                return ACTIONS[np.random.randint(NUM_ACTIONS)], 0.4

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                logits, _ = self.network(state_t)
                probs = F.softmax(logits, dim=-1).squeeze(0)

            action_idx = torch.multinomial(probs, 1).item()
            return ACTIONS[action_idx], float(probs[action_idx])

        def update(self, features: dict[str, float], action: str, reward: float,
                   next_features: dict[str, float] | None = None) -> None:
            state = np.array(
                [features.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32
            )
            action_idx = ACTIONS.index(action)

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                logits, value = self.network(state_t)
                probs = F.softmax(logits, dim=-1)
                log_prob = torch.log(probs[0, action_idx])

            self.trajectory.append({
                "state": state, "action": action_idx, "reward": reward,
                "log_prob": log_prob.item(), "value": value.item(),
            })
            self.trade_history.append(reward)
            self.total_trades += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if len(self.trajectory) >= self.batch_size * 2:
                self._train_ppo()

        def _train_ppo(self) -> None:
            if len(self.trajectory) < 4:
                return

            returns = []
            G = 0
            for step in reversed(self.trajectory):
                G = step["reward"] + self.gamma * G
                returns.insert(0, G)

            returns = np.array(returns, dtype=np.float32)
            values = np.array([s["value"] for s in self.trajectory], dtype=np.float32)
            advantages = returns - values
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

            states_t = torch.FloatTensor(np.array([s["state"] for s in self.trajectory]))
            actions_t = torch.LongTensor([s["action"] for s in self.trajectory])
            old_log_probs_t = torch.FloatTensor([s["log_prob"] for s in self.trajectory])
            returns_t = torch.FloatTensor(returns)
            advantages_t = torch.FloatTensor(advantages)

            for _ in range(self.epochs):
                logits, values = self.network(states_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * advantages_t
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.smooth_l1_loss(values.squeeze(), returns_t)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

            self.trajectory.clear()
            self.train_steps += 1

        def get_sharpe(self) -> float:
            if len(self.trade_history) < 10:
                return 0.0
            returns = np.array(self.trade_history)
            std_r = returns.std()
            if std_r < 1e-8:
                return 0.0
            return float(returns.mean() / std_r)


# ============================================================
#  Tabular Fallback (when PyTorch unavailable)
# ============================================================

class QLearnerAgent:
    """Tabular Q-learning fallback agent (no PyTorch needed)."""

    def __init__(self, agent_id: int, state_bins: dict[str, list[float]],
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.15) -> None:
        self.agent_id = agent_id
        self.state_bins = state_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = ACTIONS
        self.q_table: dict[tuple, dict[str, float]] = {}
        self.trade_history: deque[float] = deque(maxlen=100)
        self.total_trades = 0

    def _discretize(self, features: dict[str, float]) -> tuple[int, ...]:
        state = []
        for name, bins in self.state_bins.items():
            val = features.get(name, 0.0)
            state.append(int(np.digitize(val, bins)))
        return tuple(state)

    def _get_q(self, state_key: tuple) -> dict[str, float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        return self.q_table[state_key]

    def predict(self, features: dict[str, float]) -> tuple[str, float]:
        state_key = self._discretize(features)
        q_vals = self._get_q(state_key)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions), 0.4
        action = max(q_vals, key=q_vals.get)
        q_arr = np.array(list(q_vals.values()))
        if q_arr.max() - q_arr.min() > 0.01:
            exp_q = np.exp(q_arr - q_arr.max())
            probs = exp_q / exp_q.sum()
            confidence = float(probs[self.actions.index(action)])
        else:
            confidence = 0.34
        return action, confidence

    def update(self, features: dict[str, float], action: str, reward: float,
               next_features: dict[str, float] | None = None) -> None:
        state_key = self._discretize(features)
        q_vals = self._get_q(state_key)
        max_next_q = 0.0
        if next_features is not None:
            next_q = self._get_q(self._discretize(next_features))
            max_next_q = max(next_q.values())
        old_q = q_vals[action]
        q_vals[action] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.trade_history.append(reward)
        self.total_trades += 1
        self.epsilon = max(0.05, self.epsilon * 0.999)

    def get_sharpe(self) -> float:
        if len(self.trade_history) < 10:
            return 0.0
        returns = np.array(self.trade_history)
        std_r = returns.std()
        if std_r < 1e-8:
            return 0.0
        return float(returns.mean() / std_r)


# ============================================================
#  RLEnsemble — main entry point (API preserved from v1.0)
# ============================================================

class RLEnsemble:
    """
    Multi-agent Deep RL ensemble (v2.0).

    When PyTorch is available:
      - 3 DQN agents with experience replay + target networks (specialized)
      - 1 PPO meta-agent seeing all features (portfolio-level)
    Fallback: 3 tabular Q-learning agents (v1.0 behavior).

    Agents vote with Sharpe-weighted confidence. PPO meta-agent gets 1.5x weight.
    """

    def __init__(self) -> None:
        self._use_deep = HAS_TORCH

        if self._use_deep:
            self.agent_momentum = DQNAgent(
                agent_id=1,
                feature_indices=[0, 1, 2, 3],
                lr=1e-3, gamma=0.95, epsilon=0.2,
            )
            self.agent_volume = DQNAgent(
                agent_id=2,
                feature_indices=[4, 5, 6, 7],
                lr=8e-4, gamma=0.9, epsilon=0.2,
            )
            self.agent_reversion = DQNAgent(
                agent_id=3,
                feature_indices=[8, 9, 10, 11],
                lr=1.2e-3, gamma=0.85, epsilon=0.2,
            )
            self.agent_meta = PPOAgent(agent_id=4, lr=3e-4, gamma=0.99)
            self.agents = [
                self.agent_momentum, self.agent_volume,
                self.agent_reversion, self.agent_meta,
            ]
            _log.info(
                "RL Ensemble v2.0: 3 DQN agents + 1 PPO meta-agent (PyTorch)"
            )
        else:
            self.agent_momentum = QLearnerAgent(
                agent_id=1,
                state_bins={
                    "regime_code": [0.5, 1.5, 2.5, 3.5],
                    "rsi": [20, 30, 40, 50, 60, 70, 80],
                    "macd_hist": [-0.02, -0.01, 0, 0.01, 0.02],
                    "returns_5": [-0.03, -0.01, 0, 0.01, 0.03],
                },
                alpha=0.1, gamma=0.95, epsilon=0.2,
            )
            self.agent_volume = QLearnerAgent(
                agent_id=2,
                state_bins={
                    "volume_ratio": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
                    "adx": [10, 20, 25, 30, 40, 50],
                    "bb_position": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "close_to_vwap": [-0.02, -0.01, 0, 0.01, 0.02],
                },
                alpha=0.08, gamma=0.9, epsilon=0.2,
            )
            self.agent_reversion = QLearnerAgent(
                agent_id=3,
                state_bins={
                    "stoch_k": [10, 20, 30, 50, 70, 80, 90],
                    "obv_divergence": [-0.5, 0.5],
                    "ema_cross": [-0.5, 0.5],
                    "rolling_vol_10": [0.005, 0.01, 0.02, 0.03, 0.05],
                },
                alpha=0.12, gamma=0.85, epsilon=0.2,
            )
            self.agents = [self.agent_momentum, self.agent_volume, self.agent_reversion]
            _log.info("RL Ensemble v2.0: 3 tabular Q-learning agents (fallback)")

        self._last_features: dict | None = None
        self._last_actions: dict[int, str] = {}

    def _encode_regime(self, regime: str) -> float:
        mapping = {
            "ranging": 0, "trending_up": 1, "trending_down": 2,
            "high_volatility": 3, "volatile": 3, "crash": 3,
            "breakout": 1,
        }
        return float(mapping.get(regime, 0))

    def _extract_features(self, df_ind: Any, regime: str = "") -> dict[str, float]:
        last = df_ind.iloc[-1]
        return {
            "regime_code": self._encode_regime(regime),
            "rsi": float(last.get("rsi", 50)),
            "macd_hist": float(last.get("macd_hist", 0)),
            "returns_5": float(last.get("returns_5", 0)),
            "volume_ratio": float(last.get("volume_ratio", 1)),
            "adx": float(last.get("adx", 20)),
            "bb_position": float(last.get("bb_position", 0.5)),
            "close_to_vwap": float(last.get("close_to_vwap", 0)),
            "stoch_k": float(last.get("stoch_k", 50)),
            "obv_divergence": float(last.get("obv_divergence", 0)),
            "ema_cross": float(last.get("ema_cross", 0)),
            "rolling_vol_10": float(last.get("rolling_vol_10", 0.02)),
        }

    def predict(self, df_ind: Any, regime: str = "") -> tuple[str, float]:
        """Get ensemble vote from all RL agents. Returns (signal, confidence)."""
        features = self._extract_features(df_ind, regime)

        votes = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        agent_actions = {}
        total_weight = 0.0

        for agent in self.agents:
            action, conf = agent.predict(features)
            agent_actions[agent.agent_id] = action

            sharpe = max(0.1, agent.get_sharpe() + 0.5)
            weight = sharpe * conf

            # PPO meta-agent gets 1.5x weight
            if self._use_deep and agent.agent_id == 4:
                weight *= 1.5

            votes[action] += weight
            total_weight += weight

        self._last_features = features
        self._last_actions = agent_actions

        if total_weight < 0.01:
            return "HOLD", 0.3

        winner = max(votes, key=votes.get)
        confidence = min(0.85, votes[winner] / total_weight)
        return winner, confidence

    def update_reward(self, reward: float, next_df_ind: Any = None, regime: str = "") -> None:
        """Update all agents with the trade outcome."""
        if self._last_features is None:
            return

        next_features = None
        if next_df_ind is not None:
            next_features = self._extract_features(next_df_ind, regime)

        for agent in self.agents:
            action = self._last_actions.get(agent.agent_id, "HOLD")
            agent.update(self._last_features, action, reward, next_features)

    def get_stats(self) -> dict[str, Any]:
        stats = {"engine": "deep_rl" if self._use_deep else "tabular_q"}
        for agent in self.agents:
            name = f"agent_{agent.agent_id}"
            agent_stats = {
                "total_trades": agent.total_trades,
                "sharpe": round(agent.get_sharpe(), 3),
            }
            if hasattr(agent, "epsilon"):
                agent_stats["epsilon"] = round(agent.epsilon, 4)
            if hasattr(agent, "train_steps"):
                agent_stats["train_steps"] = agent.train_steps
            if hasattr(agent, "replay"):
                agent_stats["replay_size"] = len(agent.replay)
            if hasattr(agent, "q_table"):
                agent_stats["q_table_size"] = len(agent.q_table)
            stats[name] = agent_stats
        return stats

    def to_dict(self) -> dict[str, Any]:
        data = {"version": "2.0", "engine": "deep_rl" if self._use_deep else "tabular_q"}

        if self._use_deep:
            for agent in self.agents:
                key = f"agent_{agent.agent_id}"
                agent_data = {
                    "total_trades": agent.total_trades,
                    "trade_history": list(agent.trade_history),
                    "epsilon": agent.epsilon,
                }
                if hasattr(agent, "q_net"):
                    agent_data["q_net_state"] = {
                        k: v.tolist()
                        for k, v in agent.q_net.state_dict().items()
                    }
                    agent_data["target_net_state"] = {
                        k: v.tolist()
                        for k, v in agent.target_net.state_dict().items()
                    }
                if hasattr(agent, "network"):
                    agent_data["network_state"] = {
                        k: v.tolist()
                        for k, v in agent.network.state_dict().items()
                    }
                data[key] = agent_data
        else:
            for agent in self.agents:
                key = f"agent_{agent.agent_id}"
                data[key] = {
                    "q_table": {str(k): v for k, v in agent.q_table.items()},
                    "epsilon": agent.epsilon,
                    "total_trades": agent.total_trades,
                    "trade_history": list(agent.trade_history),
                }
        return data

    def from_dict(self, data: dict[str, Any]) -> None:
        version = data.get("version", "1.0")

        if version == "2.0" and self._use_deep and data.get("engine") == "deep_rl":
            for agent in self.agents:
                key = f"agent_{agent.agent_id}"
                if key not in data:
                    continue
                agent_data = data[key]
                agent.total_trades = agent_data.get("total_trades", 0)
                agent.epsilon = agent_data.get("epsilon", 0.2)
                agent.trade_history = deque(
                    agent_data.get("trade_history", []), maxlen=200
                )
                if hasattr(agent, "q_net") and "q_net_state" in agent_data:
                    try:
                        state_dict = {
                            k: torch.tensor(v)
                            for k, v in agent_data["q_net_state"].items()
                        }
                        agent.q_net.load_state_dict(state_dict)
                        target_dict = {
                            k: torch.tensor(v)
                            for k, v in agent_data["target_net_state"].items()
                        }
                        agent.target_net.load_state_dict(target_dict)
                    except Exception as e:
                        _log.warning("Could not restore DQN agent %d: %s", agent.agent_id, e)
                if hasattr(agent, "network") and "network_state" in agent_data:
                    try:
                        state_dict = {
                            k: torch.tensor(v)
                            for k, v in agent_data["network_state"].items()
                        }
                        agent.network.load_state_dict(state_dict)
                    except Exception as e:
                        _log.warning("Could not restore PPO agent: %s", e)
        else:
            for agent in self.agents:
                if not hasattr(agent, "q_table"):
                    continue
                key = f"agent_{agent.agent_id}"
                if key not in data:
                    continue
                agent_data = data[key]
                agent.q_table = {}
                for k, v in agent_data.get("q_table", {}).items():
                    try:
                        agent.q_table[ast.literal_eval(k)] = v
                    except (ValueError, SyntaxError):
                        pass
                agent.epsilon = agent_data.get("epsilon", 0.15)
                agent.total_trades = agent_data.get("total_trades", 0)
                agent.trade_history = deque(
                    agent_data.get("trade_history", []), maxlen=100
                )

        _log.info("RL Ensemble state restored (v%s)", version)
