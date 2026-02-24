"""
State Persistence Module v5.0
==============================
Save and restore agent state (positions, capital, model, config).
Survives restarts — agent picks up where it left off.
v5.0: Also persists autonomous subsystem state (decision engine,
meta-learner, evolver, optimizer, healer).
"""

from __future__ import annotations

import json
import logging
import os
import pickle  # noqa: S403 — ML model serialisation, trusted local file only
from datetime import datetime
from typing import Any

from config import Config

_log = logging.getLogger(__name__)


class StateManager:
    """
    Persists agent state to disk so the agent can survive restarts.
    Saves: risk manager state, model weights, regime history, cycle count,
           and all v5.0 autonomous subsystem state.
    """

    def __init__(self, state_file: str | None = None) -> None:
        """Initialise the state manager with paths for the three persistence files.

        Args:
            state_file: Path to the primary JSON state file.  Defaults to
                ``Config.STATE_FILE`` when not provided.  The model pickle and
                autonomous JSON paths are derived from this value automatically.
        """
        self.state_file = state_file or Config.STATE_FILE
        self.model_file = self.state_file.replace(".json", "_model.pkl")
        self.autonomous_file = self.state_file.replace(".json", "_autonomous.json")

    def save(self, agent: Any) -> bool:
        """Persist the complete agent state to disk.

        Writes three files:

        1. ``<state_file>`` — JSON with cycle count, risk manager state, and
           metadata.
        2. ``<state_file_model>.pkl`` — Pickle of the trained ML model
           (only when the model has been trained).
        3. ``<state_file_autonomous>.json`` — JSON snapshot of the autonomous
           decision engine subsystems (v5.0+).

        Args:
            agent: The live :class:`agent.TradingAgent` instance whose state
                should be serialised.

        Returns:
            True on success; False if any exception is raised during
            serialisation (the error is logged but not re-raised).
        """
        try:
            state = {
                "saved_at": datetime.now().isoformat(),
                "version": "5.0",
                "cycle_count": agent.cycle_count,
                "risk_manager": agent.risk.to_dict(),
                "last_train_time": (agent.last_train_time.isoformat() if agent.last_train_time else None),
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
            if hasattr(agent, "decision"):
                autonomous_state = agent.decision.to_dict()
                with open(self.autonomous_file, "w") as f:
                    json.dump(autonomous_state, f, indent=2)

            return True
        except Exception as e:
            _log.error("[StateManager] Save error: %s", e)
            return False

    def load(self, agent: Any) -> bool:
        """Restore the agent state from the files written by :meth:`save`.

        Reads the JSON state file and, if present, the model pickle and
        autonomous subsystem JSON.  If the state file does not exist, returns
        False immediately without modifying *agent*.

        Args:
            agent: The :class:`agent.TradingAgent` instance to restore into.
                Its attributes (``cycle_count``, ``risk``, ``model``,
                ``decision``) are mutated in-place.

        Returns:
            True when the state was successfully restored; False if the state
            file is missing or any exception occurs during deserialisation.
        """
        if not os.path.exists(self.state_file):
            return False

        try:
            with open(self.state_file) as f:
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
            if hasattr(agent, "decision") and os.path.exists(self.autonomous_file):
                with open(self.autonomous_file) as f:
                    autonomous_state = json.load(f)
                agent.decision.from_dict(autonomous_state)
                _log.info(
                    "[State] Autonomous state restored: decision=%s | meta_rounds=%s | evolution_gen=%s",
                    agent.decision.state.value,
                    agent.decision.meta.learning_count,
                    agent.decision.evolver.generation,
                )

            saved_at = state.get("saved_at", "unknown")
            version = state.get("version", "4.0")
            positions = len(state["risk_manager"].get("positions", []))
            _log.info("[State] Restored from %s (v%s)", saved_at, version)
            _log.info(
                "[State] Cycles: %d | Positions: %d | Capital: $%,.2f | PnL: $%,.2f",
                agent.cycle_count,
                positions,
                agent.risk.capital,
                agent.risk.total_pnl,
            )
            return True

        except Exception as e:
            _log.error("[StateManager] Load error: %s", e)
            return False

    def exists(self) -> bool:
        """Check whether a saved state file exists on disk.

        Returns:
            True when the primary state JSON file is present at
            :attr:`state_file`; False otherwise.
        """
        return os.path.exists(self.state_file)

    def clear(self) -> None:
        """Delete all saved state files from disk.

        Removes the JSON state file, the model pickle file, and the
        autonomous subsystem JSON file if they exist.  Missing files are
        silently skipped.
        """
        for f in [self.state_file, self.model_file, self.autonomous_file]:
            if os.path.exists(f):
                os.remove(f)
