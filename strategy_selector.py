"""
Strategy Meta-Selection DQN v1.0
=================================
Learns which trading strategy (or blend) performs best in current
market conditions, replacing static REGIME_STRATEGY_MAP.

Architecture:
  - State: [regime features, recent strategy PnL, volatility, momentum, sentiment]
  - Actions: Select strategy weights (softmax over N strategies)
  - Reward: P&L from the selected strategy over the next N bars
  - Network: 2-layer MLP with target network for stability

The key insight: a DQN that selects among strategies can outperform
any individual strategy because it learns regime-strategy-performance
associations that are impossible to hard-code.

Falls back to static REGIME_STRATEGY_MAP when not enough experience.
"""

import logging
from typing import Any

import numpy as np
from collections import deque
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    pass


# ── Data Structures ───────────────────────────────────────────────

@dataclass
class StrategyExperience:
    """A single experience tuple for the DQN replay buffer."""
    state: np.ndarray
    action: int           # Index of selected strategy
    reward: float         # P&L from that strategy
    next_state: np.ndarray
    done: bool = False


@dataclass
class SelectionResult:
    """Result of strategy meta-selection."""
    strategy_weights: dict[str, float]    # {strategy_name: weight}
    primary_strategy: str     # Highest-weighted strategy
    confidence: float         # How confident the selector is
    source: str               # "dqn" or "static_fallback"
    q_values: dict[str, float] = field(default_factory=dict)


# ── DQN Network ───────────────────────────────────────────────────

if _HAS_TORCH:
    class StrategyQNetwork(nn.Module):
        """
        Q-network for strategy selection.
        Maps state → Q-value per strategy.
        """

        def __init__(self, state_dim: int, n_strategies: int,
                     hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_strategies),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return Q-values for each strategy given a state vector."""
            return self.net(x)


# ── Meta-Selector ─────────────────────────────────────────────────

class StrategyMetaSelector:
    """
    DQN-based meta-selector that learns which strategy to use
    based on market conditions.

    When insufficient experience, falls back to static mapping.
    """

    # DQN hyperparameters
    BUFFER_SIZE = 5000
    BATCH_SIZE = 32
    GAMMA = 0.95         # Discount factor
    TAU = 0.005          # Target network soft update
    LR = 0.001
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    MIN_EXPERIENCES = 100  # Minimum before DQN takes over

    # Performance tracking
    PERF_WINDOW = 50      # Rolling window of strategy performance

    def __init__(self, strategy_names: list[str], state_dim: int = 12) -> None:
        self.strategy_names = strategy_names
        self.n_strategies = len(strategy_names)
        self.state_dim = state_dim

        # Experience replay
        self._replay_buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._epsilon = self.EPSILON_START

        # Per-strategy performance tracking
        self._strategy_pnl: dict = {s: deque(maxlen=self.PERF_WINDOW)
                                     for s in strategy_names}
        self._strategy_win_rates: dict = {s: 0.0 for s in strategy_names}
        self._total_selections: int = 0

        # Last state for experience building
        self._last_state: np.ndarray | None = None
        self._last_action: int | None = None

        # DQN networks
        self._has_dqn = _HAS_TORCH
        if self._has_dqn:
            self._q_net = StrategyQNetwork(state_dim, self.n_strategies)
            self._target_net = StrategyQNetwork(state_dim, self.n_strategies)
            self._target_net.load_state_dict(self._q_net.state_dict())
            self._optimizer = optim.Adam(self._q_net.parameters(), lr=self.LR)
            self._loss_fn = nn.MSELoss()
            self._train_step = 0

    # ── Public Interface ──────────────────────────────────────────

    def select_strategy(self, state: np.ndarray) -> SelectionResult:
        """
        Select the best strategy (or blend) for the current state.

        Args:
            state: Feature vector [regime_encoded, volatility, momentum,
                   rsi, macd, recent_pnl_per_strategy, ...]

        Returns:
            SelectionResult with strategy weights
        """
        self._total_selections += 1

        # Not enough experience → fallback
        if not self._has_dqn or len(self._replay_buffer) < self.MIN_EXPERIENCES:
            return self._static_fallback(state)

        # Epsilon-greedy exploration
        if np.random.random() < self._epsilon:
            action = np.random.randint(self.n_strategies)
            source = "dqn_explore"
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self._q_net(state_t).squeeze().numpy()
            action = int(np.argmax(q_values))
            source = "dqn"

        # Decay epsilon
        self._epsilon = max(
            self.EPSILON_END,
            self._epsilon * self.EPSILON_DECAY,
        )

        # Build weights: primary strategy gets 50%, rest share 50% by Q-value
        weights = self._action_to_weights(action, state)

        # Save state for experience building
        self._last_state = state.copy()
        self._last_action = action

        # Q-values for reporting
        q_dict = {}
        if self._has_dqn:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                qv = self._q_net(state_t).squeeze().numpy()
            q_dict = {self.strategy_names[i]: round(float(qv[i]), 4)
                      for i in range(self.n_strategies)}

        return SelectionResult(
            strategy_weights=weights,
            primary_strategy=self.strategy_names[action],
            confidence=max(weights.values()),
            source=source,
            q_values=q_dict,
        )

    def record_reward(self, reward: float, next_state: np.ndarray,
                      strategy_name: str | None = None) -> None:
        """
        Record the outcome of the last strategy selection.

        Args:
            reward: P&L or return from the strategy
            next_state: Current state after the strategy played out
            strategy_name: Which strategy was used (for performance tracking)
        """
        # Track per-strategy performance
        if strategy_name and strategy_name in self._strategy_pnl:
            self._strategy_pnl[strategy_name].append(reward)
            pnl_list = list(self._strategy_pnl[strategy_name])
            if pnl_list:
                wins = sum(1 for p in pnl_list if p > 0)
                self._strategy_win_rates[strategy_name] = wins / len(pnl_list)

        # Store experience
        if self._last_state is not None and self._last_action is not None:
            exp = StrategyExperience(
                state=self._last_state,
                action=self._last_action,
                reward=reward,
                next_state=next_state.copy(),
            )
            self._replay_buffer.append(exp)

            # Train DQN if enough experiences
            if (self._has_dqn and
                    len(self._replay_buffer) >= self.MIN_EXPERIENCES):
                self._train_step_dqn()

    def get_strategy_performance(self) -> dict[str, Any]:
        """Return per-strategy performance stats."""
        stats = {}
        for name in self.strategy_names:
            pnl_list = list(self._strategy_pnl[name])
            n = len(pnl_list)
            stats[name] = {
                "trades": n,
                "win_rate": round(self._strategy_win_rates[name], 3),
                "avg_pnl": round(np.mean(pnl_list), 4) if pnl_list else 0.0,
                "total_pnl": round(sum(pnl_list), 4) if pnl_list else 0.0,
            }
        return stats

    def get_status(self) -> dict[str, Any]:
        """Dashboard-friendly status."""
        return {
            "mode": "dqn" if (self._has_dqn and
                              len(self._replay_buffer) >= self.MIN_EXPERIENCES)
                    else "static_fallback",
            "epsilon": round(self._epsilon, 4),
            "buffer_size": len(self._replay_buffer),
            "total_selections": self._total_selections,
            "train_steps": self._train_step if self._has_dqn else 0,
            "strategy_performance": self.get_strategy_performance(),
        }

    # ── Internal: DQN Training ────────────────────────────────────

    def _train_step_dqn(self) -> None:
        """Single training step with experience replay."""
        if len(self._replay_buffer) < self.BATCH_SIZE:
            return

        # Sample mini-batch
        indices = np.random.choice(len(self._replay_buffer),
                                    self.BATCH_SIZE, replace=False)
        batch = [self._replay_buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))

        # Current Q-values
        q_values = self._q_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values (from target network)
        with torch.no_grad():
            next_q = self._target_net(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target = rewards + self.GAMMA * max_next_q

        # Update
        loss = self._loss_fn(q_selected, target)
        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), 1.0)
        self._optimizer.step()

        # Soft update target network
        self._soft_update_target()
        self._train_step += 1

    def _soft_update_target(self) -> None:
        """Polyak averaging: target = τ * online + (1-τ) * target."""
        for tp, op in zip(self._target_net.parameters(),
                          self._q_net.parameters()):
            tp.data.copy_(self.TAU * op.data + (1 - self.TAU) * tp.data)

    # ── Internal: Action → Weights ────────────────────────────────

    def _action_to_weights(self, action: int,
                           state: np.ndarray) -> dict[str, float]:
        """
        Convert a discrete action (strategy index) to a weight dict.

        Primary strategy gets 50%, rest share 50% proportional to
        their Q-values (if available) or uniformly.
        """
        weights = {}
        primary = self.strategy_names[action]

        if self._has_dqn and len(self._replay_buffer) >= self.MIN_EXPERIENCES:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_vals = self._q_net(state_t).squeeze().numpy()

            # Softmax over Q-values for secondary weighting
            q_shifted = q_vals - q_vals.max()  # Numerical stability
            exp_q = np.exp(q_shifted)
            softmax = exp_q / (exp_q.sum() + 1e-10)

            for i, name in enumerate(self.strategy_names):
                if name == primary:
                    weights[name] = 0.50
                else:
                    weights[name] = 0.50 * softmax[i] / (1 - softmax[action] + 1e-10)
        else:
            # Uniform secondary weights
            secondary_weight = 0.50 / max(self.n_strategies - 1, 1)
            for name in self.strategy_names:
                weights[name] = 0.50 if name == primary else secondary_weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        return weights

    # ── Internal: Static Fallback ─────────────────────────────────

    def _static_fallback(self, state: np.ndarray) -> SelectionResult:
        """
        Fallback strategy selection based on performance history.

        Uses win-rate weighted selection when DQN isn't ready.
        """
        # If we have some performance data, use it
        if any(list(self._strategy_pnl[s]) for s in self.strategy_names):
            # Weight by historical performance
            scores = {}
            for name in self.strategy_names:
                pnl_list = list(self._strategy_pnl[name])
                if pnl_list:
                    avg = np.mean(pnl_list)
                    wr = self._strategy_win_rates[name]
                    scores[name] = avg * 0.6 + wr * 0.4
                else:
                    scores[name] = 0.0

            best = max(scores, key=scores.get)
            total_score = sum(max(s, 0) for s in scores.values()) + 1e-10
            weights = {k: max(v, 0) / total_score for k, v in scores.items()}
        else:
            # No data at all — uniform
            w = 1.0 / self.n_strategies
            weights = {s: w for s in self.strategy_names}
            best = self.strategy_names[0]

        # Save state for experience building
        self._last_state = state.copy()
        self._last_action = self.strategy_names.index(best)

        return SelectionResult(
            strategy_weights=weights,
            primary_strategy=best,
            confidence=weights.get(best, 0.0),
            source="static_fallback",
        )

    # ── Serialization ─────────────────────────────────────────────

    def save_state(self) -> dict[str, Any]:
        """Serialize selector state for persistence."""
        state = {
            "strategy_names": self.strategy_names,
            "epsilon": self._epsilon,
            "total_selections": self._total_selections,
            "train_step": self._train_step if self._has_dqn else 0,
            "strategy_pnl": {k: list(v) for k, v in self._strategy_pnl.items()},
            "strategy_win_rates": self._strategy_win_rates,
        }
        if self._has_dqn:
            state["q_net_state"] = {
                k: v.tolist() for k, v in self._q_net.state_dict().items()
            }
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore selector state."""
        self._epsilon = state.get("epsilon", self.EPSILON_START)
        self._total_selections = state.get("total_selections", 0)
        self._train_step = state.get("train_step", 0)

        for k, v in state.get("strategy_pnl", {}).items():
            if k in self._strategy_pnl:
                self._strategy_pnl[k] = deque(v, maxlen=self.PERF_WINDOW)

        self._strategy_win_rates = state.get("strategy_win_rates",
                                             self._strategy_win_rates)

        if self._has_dqn and "q_net_state" in state:
            try:
                sd = {k: torch.tensor(v) for k, v in state["q_net_state"].items()}
                self._q_net.load_state_dict(sd)
                self._target_net.load_state_dict(sd)
            except Exception as e:
                _log.warning("Failed to load DQN state: %s", e)
