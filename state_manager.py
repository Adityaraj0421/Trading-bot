"""
State Persistence Module v5.0
==============================
Save and restore agent state (positions, capital, model, config).
Survives restarts — agent picks up where it left off.
v5.0: Also persists autonomous subsystem state (decision engine,
meta-learner, evolver, optimizer, healer).
"""

import json
import logging
import os
import pickle
from datetime import datetime
from config import Config

_log = logging.getLogger(__name__)


class StateManager:
    """
    Persists agent state to disk so the agent can survive restarts.
    Saves: risk manager state, model weights, regime history, cycle count,
           and all v5.0 autonomous subsystem state.
    """

    def __init__(self, state_file: str | None = None) -> None:
        self.state_file = state_file or Config.STATE_FILE
        self.model_file = self.state_file.replace(".json", "_model.pkl")
        self.autonomous_file = self.state_file.replace(".json", "_autonomous.json")

    def save(self, agent) -> bool:
        """Save complete agent state to disk."""
        try:
            state = {
                "saved_at": datetime.now().isoformat(),
                "version": "5.0",
                "cycle_count": agent.cycle_count,
                "risk_manager": agent.risk.to_dict(),
                "last_train_time": (
                    agent.last_train_time.isoformat()
                    if agent.last_train_time else None
                ),
                "last_data_hash": agent._last_data_hash,
                "regime_history_len": len(agent.regime_detector.regime_history),
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Save ML model separately (binary)
            if agent.model.is_trained:
                model_state = {
                    "rf_model": agent.model.rf_model,
                    "gb_model": agent.model.gb_model,
                    "scaler": agent.model.scaler,
                    "is_trained": True,
                    "last_train_accuracy": agent.model.last_train_accuracy,
                }
                with open(self.model_file, "wb") as f:
                    pickle.dump(model_state, f)

            # v5.0: Save autonomous subsystem state
            if hasattr(agent, 'decision'):
                autonomous_state = agent.decision.to_dict()
                with open(self.autonomous_file, "w") as f:
                    json.dump(autonomous_state, f, indent=2)

            return True
        except Exception as e:
            _log.error("[StateManager] Save error: %s", e)
            return False

    def load(self, agent) -> bool:
        """Restore agent state from disk."""
        if not os.path.exists(self.state_file):
            return False

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            agent.cycle_count = state.get("cycle_count", 0)
            agent.risk.from_dict(state["risk_manager"])

            last_train = state.get("last_train_time")
            if last_train:
                agent.last_train_time = datetime.fromisoformat(last_train)

            agent._last_data_hash = state.get("last_data_hash")

            # Restore ML model
            if os.path.exists(self.model_file):
                with open(self.model_file, "rb") as f:
                    model_state = pickle.load(f)
                agent.model.rf_model = model_state["rf_model"]
                agent.model.gb_model = model_state["gb_model"]
                agent.model.scaler = model_state["scaler"]
                agent.model.is_trained = model_state["is_trained"]
                agent.model.last_train_accuracy = model_state["last_train_accuracy"]

            # v5.0: Restore autonomous subsystem state
            if hasattr(agent, 'decision') and os.path.exists(self.autonomous_file):
                with open(self.autonomous_file, "r") as f:
                    autonomous_state = json.load(f)
                agent.decision.from_dict(autonomous_state)
                _log.info("[State] Autonomous state restored: "
                         "decision=%s | meta_rounds=%s | evolution_gen=%s",
                         agent.decision.state.value,
                         agent.decision.meta.learning_count,
                         agent.decision.evolver.generation)

            saved_at = state.get("saved_at", "unknown")
            version = state.get("version", "4.0")
            positions = len(state["risk_manager"].get("positions", []))
            _log.info("[State] Restored from %s (v%s)", saved_at, version)
            _log.info("[State] Cycles: %d | Positions: %d | Capital: $%,.2f | PnL: $%,.2f",
                      agent.cycle_count, positions, agent.risk.capital, agent.risk.total_pnl)
            return True

        except Exception as e:
            _log.error("[StateManager] Load error: %s", e)
            return False

    def exists(self) -> bool:
        """Check whether a saved state file exists on disk."""
        return os.path.exists(self.state_file)

    def clear(self) -> None:
        """Delete saved state."""
        for f in [self.state_file, self.model_file, self.autonomous_file]:
            if os.path.exists(f):
                os.remove(f)
