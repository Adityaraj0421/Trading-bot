"""
Unit tests for config.py — Config class defaults, validation, path resolution.

Uses monkeypatch to override Config class attributes since env vars are read
at import time.
"""

import pytest
from config import Config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_default_exchange_id(self):
        assert Config.EXCHANGE_ID == "binance"

    def test_default_trading_pair(self):
        assert Config.TRADING_PAIR == "BTC/USDT"

    def test_default_timeframe(self):
        assert Config.TIMEFRAME == "1h"

    def test_default_initial_capital(self):
        assert Config.INITIAL_CAPITAL == 1000.0

    def test_default_trading_mode_paper(self):
        assert Config.TRADING_MODE == "paper"

    def test_default_max_position_pct(self):
        assert Config.MAX_POSITION_PCT == 0.02

    def test_default_fee_pct(self):
        assert Config.FEE_PCT == 0.001

    def test_default_api_port(self):
        assert Config.API_PORT == 8000

    def test_default_max_open_positions(self):
        assert Config.MAX_OPEN_POSITIONS == 3

    def test_default_min_confidence(self):
        # v9.0: lowered from 0.6 to 0.35 to allow more signal generation
        assert Config.MIN_CONFIDENCE == 0.35


# ---------------------------------------------------------------------------
# is_paper_mode
# ---------------------------------------------------------------------------

class TestIsPaperMode:
    def test_paper_mode_true(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "paper")
        assert Config.is_paper_mode() is True

    def test_paper_mode_false(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "live")
        assert Config.is_paper_mode() is False

    def test_paper_mode_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "PAPER")
        assert Config.is_paper_mode() is True


# ---------------------------------------------------------------------------
# any_intelligence_enabled
# ---------------------------------------------------------------------------

class TestAnyIntelligenceEnabled:
    # v9.0: any_intelligence_enabled() now checks 7 flags
    # (added ENABLE_FUNDING_OI and ENABLE_LIQUIDATION)
    INTEL_FLAGS = [
        "ENABLE_ONCHAIN", "ENABLE_WHALE_TRACKING", "ENABLE_NEWS_NLP",
        "ENABLE_CORRELATION", "ENABLE_ORDERBOOK",
        "ENABLE_FUNDING_OI", "ENABLE_LIQUIDATION",
    ]

    def test_all_disabled(self, monkeypatch):
        for attr in self.INTEL_FLAGS:
            monkeypatch.setattr(Config, attr, False)
        assert Config.any_intelligence_enabled() is False

    def test_one_enabled(self, monkeypatch):
        for attr in self.INTEL_FLAGS:
            monkeypatch.setattr(Config, attr, False)
        monkeypatch.setattr(Config, "ENABLE_ONCHAIN", True)
        assert Config.any_intelligence_enabled() is True

    def test_all_enabled(self, monkeypatch):
        for attr in self.INTEL_FLAGS:
            monkeypatch.setattr(Config, attr, True)
        assert Config.any_intelligence_enabled() is True


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_paper_mode_no_keys_ok(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "paper")
        monkeypatch.setattr(Config, "API_KEY", "")
        monkeypatch.setattr(Config, "API_SECRET", "")
        monkeypatch.setattr(Config, "STATE_FILE", "")
        monkeypatch.setattr(Config, "LOG_FILE", "")
        Config.validate()  # Should not raise

    def test_live_mode_missing_keys_raises(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "live")
        monkeypatch.setattr(Config, "API_KEY", "")
        monkeypatch.setattr(Config, "API_SECRET", "")
        monkeypatch.setattr(Config, "STATE_FILE", "")
        monkeypatch.setattr(Config, "LOG_FILE", "")
        with pytest.raises(ValueError, match="API_KEY"):
            Config.validate()

    def test_live_mode_with_keys_ok(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "live")
        monkeypatch.setattr(Config, "API_KEY", "test_key")
        monkeypatch.setattr(Config, "API_SECRET", "test_secret")
        monkeypatch.setattr(Config, "STATE_FILE", "")
        monkeypatch.setattr(Config, "LOG_FILE", "")
        Config.validate()  # Should not raise

    def test_validate_resolves_state_file(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "paper")
        monkeypatch.setattr(Config, "STATE_FILE", "")
        monkeypatch.setattr(Config, "LOG_FILE", "")
        Config.validate()
        assert Config.STATE_FILE != ""
        assert Config.STATE_FILE.endswith(".json")

    def test_validate_resolves_log_file(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "paper")
        monkeypatch.setattr(Config, "STATE_FILE", "")
        monkeypatch.setattr(Config, "LOG_FILE", "")
        Config.validate()
        assert Config.LOG_FILE != ""


# ---------------------------------------------------------------------------
# _resolve_paths
# ---------------------------------------------------------------------------

class TestResolvePaths:
    def test_resolve_uses_data_dir(self, monkeypatch):
        monkeypatch.setattr(Config, "DATA_DIR", "/tmp/data")
        monkeypatch.setattr(Config, "STATE_FILE", "")
        monkeypatch.setattr(Config, "LOG_FILE", "")
        Config._resolve_paths()
        assert Config.STATE_FILE.startswith("/tmp/data")

    def test_resolve_does_not_overwrite_explicit(self, monkeypatch):
        monkeypatch.setattr(Config, "STATE_FILE", "/custom/state.json")
        monkeypatch.setattr(Config, "LOG_FILE", "/custom/agent.log")
        Config._resolve_paths()
        assert Config.STATE_FILE == "/custom/state.json"
        assert Config.LOG_FILE == "/custom/agent.log"
