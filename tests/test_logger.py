"""
Tests for StructuredLogger — event logging, JSON output, event queries.
"""

import json
import uuid
import pytest
from logger import StructuredLogger


@pytest.fixture()
def logger(tmp_path, monkeypatch):
    """StructuredLogger writing to a temp file.

    Uses a unique logger name each invocation so Python's logging singleton
    cache doesn't recycle a handler pointing at a *different* test's tmp_path.
    """
    monkeypatch.setattr("config.Config.LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setattr("config.Config.LOG_LEVEL", "DEBUG")
    unique = uuid.uuid4().hex[:8]
    log = StructuredLogger(name=f"test_logger_{unique}")
    yield log
    # Tear down — close and remove file handlers to avoid ResourceWarnings
    for h in log.file_logger.handlers[:]:
        h.close()
        log.file_logger.removeHandler(h)


# ── Event Logging ─────────────────────────────────────────────────


class TestEventLogging:
    def test_log_cycle_start(self, logger):
        logger.log_cycle_start(cycle=1, price=50000, pair="BTC/USDT")
        events = logger.get_recent_events()
        assert len(events) == 1
        assert events[0]["type"] == "cycle_start"
        assert events[0]["cycle"] == 1

    def test_log_signal(self, logger):
        logger.log_signal("BUY", 0.85, "strategy", regime="trending_up",
                          strategy="momentum")
        events = logger.get_recent_events(event_type="signal")
        assert len(events) == 1
        assert events[0]["signal"] == "BUY"

    def test_log_trade_open(self, logger):
        logger.log_trade_open("BTC/USDT", "long", 50000, 0.1,
                              sl=49000, tp=52000, trailing=500,
                              strategy="momentum")
        events = logger.get_recent_events(event_type="trade_open")
        assert len(events) == 1
        assert events[0]["symbol"] == "BTC/USDT"

    def test_log_trade_close(self, logger):
        logger.log_trade_close("BTC/USDT", "long", 50000, 51000,
                               pnl_net=90, pnl_gross=100, fees=10,
                               reason="take_profit", hold_bars=5,
                               strategy="momentum")
        events = logger.get_recent_events(event_type="trade_close")
        assert len(events) == 1
        assert events[0]["pnl_net"] == 90

    def test_log_regime_change(self, logger):
        logger.log_regime_change("ranging", "trending_up", 0.8)
        events = logger.get_recent_events(event_type="regime_change")
        assert len(events) == 1

    def test_log_model_train(self, logger):
        logger.log_model_train(accuracy=0.75, samples=500, drift_detected=False)
        events = logger.get_recent_events(event_type="model_train")
        assert len(events) == 1
        assert events[0]["accuracy"] == 0.75

    def test_log_error(self, logger):
        logger.log_error("test_component", "something broke")
        events = logger.get_recent_events(event_type="error")
        assert len(events) == 1
        assert events[0]["level"] == "ERROR"

    def test_log_portfolio(self, logger):
        logger.log_portfolio(capital=10000, positions=2, total_pnl=500,
                             fees=50, win_rate=0.65)
        events = logger.get_recent_events(event_type="portfolio")
        assert len(events) == 1


# ── Event Queries ─────────────────────────────────────────────────


class TestEventQueries:
    def test_filter_by_type(self, logger):
        logger.log_cycle_start(1, 50000, "BTC/USDT")
        logger.log_signal("BUY", 0.8, "test")
        logger.log_error("comp", "err")
        assert len(logger.get_recent_events(event_type="error")) == 1
        assert len(logger.get_recent_events(event_type="signal")) == 1
        assert len(logger.get_recent_events()) == 3

    def test_limit_parameter(self, logger):
        for i in range(10):
            logger.log_cycle_start(i, 50000 + i, "BTC/USDT")
        assert len(logger.get_recent_events(limit=5)) == 5

    def test_events_have_timestamps(self, logger):
        logger.log_cycle_start(1, 50000, "BTC/USDT")
        event = logger.get_recent_events()[0]
        assert "timestamp" in event


# ── JSON File Output ──────────────────────────────────────────────


class TestFileOutput:
    def test_events_written_to_file(self, logger):
        logger.log_cycle_start(1, 50000, "BTC/USDT")
        # Flush and read from the handler's *actual* file path — avoids
        # mismatch when Python's logging singleton caches old handlers.
        for handler in logger.file_logger.handlers:
            handler.flush()
        file_handler = next(
            h for h in logger.file_logger.handlers
            if hasattr(h, "baseFilename")
        )
        content = open(file_handler.baseFilename).read()
        assert content.strip()
        # Should be valid JSON
        data = json.loads(content.strip())
        assert data["type"] == "cycle_start"


# ── Bounded History ───────────────────────────────────────────────


class TestBoundedHistory:
    def test_history_bounded_at_2000(self, logger):
        assert logger.events.maxlen == 2000
