"""Tests for market scenario generator."""

import numpy as np
import pytest

from scenarios import generate_scenario, list_scenarios


def test_list_scenarios():
    scenarios = list_scenarios()
    assert len(scenarios) == 6
    assert "bull_run" in scenarios
    assert "black_swan" in scenarios


@pytest.mark.parametrize("scenario", list_scenarios())
def test_generate_scenario_shape(scenario):
    df = generate_scenario(scenario, periods=100)
    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"


def test_bull_run_trends_up():
    np.random.seed(42)
    df = generate_scenario("bull_run", periods=1000, base_price=100.0)
    assert df["close"].iloc[-1] > df["close"].iloc[0] * 1.1  # at least 10% up


def test_bear_market_trends_down():
    """Bear market should trend down on average across multiple runs."""
    np.random.seed(7)
    df = generate_scenario("bear_market", periods=2000, base_price=100.0)
    # With drift=-0.0006 over 2000 periods, should be well below start
    assert df["close"].iloc[-1] < df["close"].iloc[0]


def test_invalid_scenario():
    with pytest.raises(ValueError, match="Unknown scenario"):
        generate_scenario("nonexistent")


def test_ohlcv_consistency():
    """High should be >= open and close, low should be <= open and close."""
    df = generate_scenario("sideways_chop", periods=200)
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["volume"] > 0).all()
