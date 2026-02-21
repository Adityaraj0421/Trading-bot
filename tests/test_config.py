"""
Unit tests for config.py — Config class defaults, validation, path resolution.

Uses monkeypatch to override Config class attributes since env vars are read
at import time.
"""

import logging
import pytest
from config import Config


# ---------------------------------------------------------------------------
# Shared fixture — sets all validated fields to safe defaults
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config(monkeypatch):
    """Set Config attributes to valid defaults so validate() passes cleanly."""
    monkeypatch.setattr(Config, "TRADING_MODE", "paper")
    monkeypatch.setattr(Config, "API_KEY", "")
    monkeypatch.setattr(Config, "API_SECRET", "")
    monkeypatch.setattr(Config, "STATE_FILE", "")
    monkeypatch.setattr(Config, "LOG_FILE", "")
    monkeypatch.setattr(Config, "TRADE_DB_PATH", "")
    monkeypatch.setattr(Config, "INITIAL_CAPITAL", 1000.0)
    monkeypatch.setattr(Config, "STOP_LOSS_PCT", 0.02)
    monkeypatch.setattr(Config, "TAKE_PROFIT_PCT", 0.05)
    monkeypatch.setattr(Config, "FEE_PCT", 0.001)
    monkeypatch.setattr(Config, "SLIPPAGE_PCT", 0.0005)
    monkeypatch.setattr(Config, "MAX_POSITION_PCT", 0.02)
    monkeypatch.setattr(Config, "MAX_OPEN_POSITIONS", 3)
    monkeypatch.setattr(Config, "MIN_CONFIDENCE", 0.6)
    monkeypatch.setattr(Config, "AGENT_INTERVAL_SECONDS", 300)
    monkeypatch.setattr(Config, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(Config, "TELEGRAM_CHAT_ID", "")


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
# validate — existing tests (now using valid_config fixture)
# ---------------------------------------------------------------------------

class TestValidate:
    def test_paper_mode_no_keys_ok(self, valid_config):
        Config.validate()  # Should not raise

    def test_live_mode_missing_keys_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "live")
        monkeypatch.setattr(Config, "API_KEY", "")
        monkeypatch.setattr(Config, "API_SECRET", "")
        with pytest.raises(ValueError, match="API_KEY"):
            Config.validate()

    def test_live_mode_with_keys_ok(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "live")
        monkeypatch.setattr(Config, "API_KEY", "test_key")
        monkeypatch.setattr(Config, "API_SECRET", "test_secret")
        Config.validate()  # Should not raise

    def test_validate_resolves_state_file(self, valid_config):
        Config.validate()
        assert Config.STATE_FILE != ""
        assert Config.STATE_FILE.endswith(".json")

    def test_validate_resolves_log_file(self, valid_config):
        Config.validate()
        assert Config.LOG_FILE != ""


# ---------------------------------------------------------------------------
# validate — hard failure tests
# ---------------------------------------------------------------------------

class TestValidateHardChecks:
    def test_zero_initial_capital_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "INITIAL_CAPITAL", 0)
        with pytest.raises(ValueError, match="INITIAL_CAPITAL"):
            Config.validate()

    def test_negative_initial_capital_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "INITIAL_CAPITAL", -100)
        with pytest.raises(ValueError, match="INITIAL_CAPITAL"):
            Config.validate()

    def test_stop_loss_zero_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "STOP_LOSS_PCT", 0)
        with pytest.raises(ValueError, match="STOP_LOSS_PCT"):
            Config.validate()

    def test_stop_loss_exceeds_one_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "STOP_LOSS_PCT", 1.5)
        with pytest.raises(ValueError, match="STOP_LOSS_PCT"):
            Config.validate()

    def test_take_profit_zero_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "TAKE_PROFIT_PCT", 0)
        with pytest.raises(ValueError, match="TAKE_PROFIT_PCT"):
            Config.validate()

    def test_stop_loss_gte_take_profit_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "STOP_LOSS_PCT", 0.05)
        monkeypatch.setattr(Config, "TAKE_PROFIT_PCT", 0.05)
        with pytest.raises(ValueError, match="STOP_LOSS_PCT.*TAKE_PROFIT_PCT"):
            Config.validate()

    def test_stop_loss_gt_take_profit_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "STOP_LOSS_PCT", 0.06)
        monkeypatch.setattr(Config, "TAKE_PROFIT_PCT", 0.05)
        with pytest.raises(ValueError, match="STOP_LOSS_PCT.*TAKE_PROFIT_PCT"):
            Config.validate()

    def test_fees_consume_stop_loss_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "FEE_PCT", 0.01)
        monkeypatch.setattr(Config, "SLIPPAGE_PCT", 0.01)
        monkeypatch.setattr(Config, "STOP_LOSS_PCT", 0.02)
        with pytest.raises(ValueError, match="Transaction costs"):
            Config.validate()

    def test_overallocated_capital_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "MAX_POSITION_PCT", 0.5)
        monkeypatch.setattr(Config, "MAX_OPEN_POSITIONS", 3)
        with pytest.raises(ValueError, match="overallocated"):
            Config.validate()

    def test_min_confidence_zero_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "MIN_CONFIDENCE", 0)
        with pytest.raises(ValueError, match="MIN_CONFIDENCE"):
            Config.validate()

    def test_min_confidence_over_one_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "MIN_CONFIDENCE", 1.5)
        with pytest.raises(ValueError, match="MIN_CONFIDENCE"):
            Config.validate()

    def test_min_confidence_one_ok(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "MIN_CONFIDENCE", 1.0)
        Config.validate()  # Exactly 1.0 is allowed

    def test_agent_interval_too_low_raises(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "AGENT_INTERVAL_SECONDS", 5)
        with pytest.raises(ValueError, match="AGENT_INTERVAL_SECONDS"):
            Config.validate()

    def test_agent_interval_ten_ok(self, valid_config, monkeypatch):
        monkeypatch.setattr(Config, "AGENT_INTERVAL_SECONDS", 10)
        Config.validate()  # Exactly 10 is the minimum


# ---------------------------------------------------------------------------
# validate — soft warning tests
# ---------------------------------------------------------------------------

class TestValidateSoftWarnings:
    def test_telegram_token_without_chat_id_warns(self, valid_config, monkeypatch, caplog):
        monkeypatch.setattr(Config, "TELEGRAM_BOT_TOKEN", "some-token")
        monkeypatch.setattr(Config, "TELEGRAM_CHAT_ID", "")
        with caplog.at_level(logging.WARNING):
            Config.validate()
        assert "TELEGRAM_CHAT_ID is missing" in caplog.text

    def test_telegram_chat_id_without_token_warns(self, valid_config, monkeypatch, caplog):
        monkeypatch.setattr(Config, "TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setattr(Config, "TELEGRAM_CHAT_ID", "12345")
        with caplog.at_level(logging.WARNING):
            Config.validate()
        assert "TELEGRAM_BOT_TOKEN is missing" in caplog.text

    def test_low_min_confidence_warns(self, valid_config, monkeypatch, caplog):
        monkeypatch.setattr(Config, "MIN_CONFIDENCE", 0.35)
        with caplog.at_level(logging.WARNING):
            Config.validate()
        assert "very low" in caplog.text

    def test_high_allocation_warns(self, valid_config, monkeypatch, caplog):
        monkeypatch.setattr(Config, "MAX_POSITION_PCT", 0.3)
        monkeypatch.setattr(Config, "MAX_OPEN_POSITIONS", 3)
        with caplog.at_level(logging.WARNING):
            Config.validate()
        assert "80%" in caplog.text


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
