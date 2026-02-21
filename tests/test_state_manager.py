"""
Unit tests for state_manager.py — StateManager save/load/clear.

Uses tmp_path for file I/O and MagicMock for the agent object.
"""

import json
import os
import pytest
from unittest.mock import MagicMock
from datetime import datetime
from state_manager import StateManager


def _make_mock_agent(trained=False, has_decision=False):
    """Build a MagicMock agent with the attributes StateManager expects."""
    agent = MagicMock()
    agent.cycle_count = 5
    agent.last_train_time = datetime(2024, 6, 15, 12, 0, 0)
    agent._last_data_hash = "abc123"

    # risk manager
    agent.risk.to_dict.return_value = {
        "positions": [],
        "capital": 1000.0,
        "total_pnl": 25.0,
    }
    agent.risk.capital = 1000.0
    agent.risk.total_pnl = 25.0

    # regime detector
    agent.regime_detector.regime_history = ["ranging", "trending_up"]

    # model
    agent.model.is_trained = trained
    if trained:
        agent.model.rf_model = "mock_rf"
        agent.model.gb_model = "mock_gb"
        agent.model.scaler = "mock_scaler"
        agent.model.last_train_accuracy = 0.72

    # decision engine (v5 autonomous)
    if has_decision:
        agent.decision.to_dict.return_value = {"state": "normal", "meta": {}}
    else:
        del agent.decision  # Remove attribute so hasattr returns False

    return agent


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestStateManagerInit:
    def test_custom_state_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        assert sm.state_file == path

    def test_model_file_derived(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        assert sm.model_file.endswith("_model.pkl")
        assert "state" not in sm.model_file or "state_model" in sm.model_file

    def test_autonomous_file_derived(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        assert sm.autonomous_file.endswith("_autonomous.json")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------

class TestExists:
    def test_false_initially(self, tmp_path):
        sm = StateManager(state_file=str(tmp_path / "state.json"))
        assert sm.exists() is False

    def test_true_after_save(self, tmp_path):
        sm = StateManager(state_file=str(tmp_path / "state.json"))
        agent = _make_mock_agent()
        sm.save(agent)
        assert sm.exists() is True


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_returns_true(self, tmp_path):
        sm = StateManager(state_file=str(tmp_path / "state.json"))
        agent = _make_mock_agent()
        assert sm.save(agent) is True

    def test_save_creates_json_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())
        assert os.path.exists(path)

    def test_save_json_contains_version(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == "5.0"

    def test_save_json_contains_cycle_count(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())
        with open(path) as f:
            data = json.load(f)
        assert data["cycle_count"] == 5

    def test_save_json_contains_risk_manager(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())
        with open(path) as f:
            data = json.load(f)
        assert "risk_manager" in data
        assert data["risk_manager"]["capital"] == 1000.0

    def test_load_restores_cycle_count(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())

        agent2 = _make_mock_agent()
        agent2.cycle_count = 0
        sm.load(agent2)
        assert agent2.cycle_count == 5

    def test_load_restores_risk_state(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())

        agent2 = _make_mock_agent()
        sm.load(agent2)
        agent2.risk.from_dict.assert_called_once()

    def test_load_restores_train_time(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent())

        agent2 = _make_mock_agent()
        agent2.last_train_time = None
        sm.load(agent2)
        assert agent2.last_train_time == datetime(2024, 6, 15, 12, 0, 0)

    def test_load_no_file_returns_false(self, tmp_path):
        sm = StateManager(state_file=str(tmp_path / "nonexistent.json"))
        agent = _make_mock_agent()
        assert sm.load(agent) is False

    def test_save_trained_model_creates_pkl(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent(trained=True))
        assert os.path.exists(sm.model_file)

    def test_untrained_model_no_pkl(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent(trained=False))
        assert not os.path.exists(sm.model_file)

    def test_save_autonomous_state(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent(has_decision=True))
        assert os.path.exists(sm.autonomous_file)

    def test_no_decision_no_autonomous_file(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent(has_decision=False))
        assert not os.path.exists(sm.autonomous_file)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_removes_all_files(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        sm.save(_make_mock_agent(trained=True, has_decision=True))
        assert os.path.exists(sm.state_file)
        assert os.path.exists(sm.model_file)
        assert os.path.exists(sm.autonomous_file)

        sm.clear()
        assert not os.path.exists(sm.state_file)
        assert not os.path.exists(sm.model_file)
        assert not os.path.exists(sm.autonomous_file)

    def test_no_error_on_nonexistent(self, tmp_path):
        sm = StateManager(state_file=str(tmp_path / "nope.json"))
        sm.clear()  # Should not raise


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_save_error_returns_false(self, tmp_path):
        path = str(tmp_path / "state.json")
        sm = StateManager(state_file=path)
        agent = MagicMock()
        agent.risk.to_dict.side_effect = RuntimeError("broken")
        assert sm.save(agent) is False

    def test_load_corrupt_json_returns_false(self, tmp_path):
        path = str(tmp_path / "state.json")
        with open(path, "w") as f:
            f.write("{invalid json!!!")
        sm = StateManager(state_file=path)
        agent = _make_mock_agent()
        assert sm.load(agent) is False
