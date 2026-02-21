"""
Shared test fixtures for the crypto trading agent.
Auto-discovered by pytest — no imports needed in test files.
"""

import os
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config isolation: ensure tests never hit real APIs
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _safe_config(monkeypatch):
    """Ensure every test runs with safe defaults (paper mode, no real keys)."""
    monkeypatch.setattr("config.Config.TRADING_MODE", "paper")
    monkeypatch.setattr("config.Config.API_KEY", "")
    monkeypatch.setattr("config.Config.API_SECRET", "")
    monkeypatch.setattr("config.Config.TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr("config.Config.TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# Reusable data generators
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_ohlcv():
    """Generate a realistic 200-bar OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    opn = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    df = pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=pd.date_range("2025-01-01", periods=n, freq="h"))
    return df


@pytest.fixture()
def sample_indicators(sample_ohlcv):
    """OHLCV with indicator columns appended (sma, rsi, atr, etc.)."""
    from indicators import Indicators
    return Indicators.add_all(sample_ohlcv)
