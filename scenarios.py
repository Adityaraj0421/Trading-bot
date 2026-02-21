"""
Market Scenario Generator
===========================
Generates synthetic OHLCV data for specific market conditions.
Used for stress-testing strategies before going live.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_scenario(scenario: str, periods: int = 500,
                      base_price: float = 100000.0,
                      timeframe_minutes: int = 60) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a named scenario.

    Scenarios:
        bull_run, bear_market, sideways_chop, flash_crash,
        black_swan, accumulation
    """
    generators = {
        "bull_run": _bull_run,
        "bear_market": _bear_market,
        "sideways_chop": _sideways_chop,
        "flash_crash": _flash_crash,
        "black_swan": _black_swan,
        "accumulation": _accumulation,
    }
    if scenario not in generators:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(generators.keys())}")

    prices = generators[scenario](periods, base_price)
    return _prices_to_ohlcv(prices, timeframe_minutes)


def _bull_run(periods: int, base: float) -> np.ndarray:
    """Steady uptrend with pullbacks. +50-100% over period."""
    drift = 0.0008
    volatility = 0.015
    returns = np.random.normal(drift, volatility, periods)
    for _ in range(3):
        start = np.random.randint(periods // 5, periods - 20)
        returns[start:start + 10] = np.random.normal(-0.008, 0.02, 10)
    prices = base * np.exp(np.cumsum(returns))
    return prices


def _bear_market(periods: int, base: float) -> np.ndarray:
    """Sustained downtrend with relief rallies. -40-60% over period."""
    drift = -0.0006
    volatility = 0.018
    returns = np.random.normal(drift, volatility, periods)
    for _ in range(3):
        start = np.random.randint(periods // 5, periods - 15)
        returns[start:start + 8] = np.random.normal(0.006, 0.015, 8)
    prices = base * np.exp(np.cumsum(returns))
    return prices


def _sideways_chop(periods: int, base: float) -> np.ndarray:
    """Range-bound with false breakouts. +/-5% around base."""
    volatility = 0.012
    returns = np.random.normal(0, volatility, periods)
    prices = [base]
    for r in returns:
        new_price = prices[-1] * (1 + r)
        deviation = (new_price - base) / base
        new_price *= (1 - deviation * 0.05)
        prices.append(new_price)
    return np.array(prices[1:])


def _flash_crash(periods: int, base: float) -> np.ndarray:
    """Normal market, then sudden 30% crash, then V-shaped recovery."""
    prices = np.zeros(periods)
    normal_end = int(periods * 0.6)
    drift = 0.0003
    vol = 0.012
    returns = np.random.normal(drift, vol, normal_end)
    prices[:normal_end] = base * np.exp(np.cumsum(returns))

    crash_bars = max(int(periods * 0.05), 5)
    crash_returns = np.random.normal(-0.06, 0.03, crash_bars)
    crash_start = prices[normal_end - 1]
    crash_prices = crash_start * np.exp(np.cumsum(crash_returns))
    prices[normal_end:normal_end + crash_bars] = crash_prices

    recovery_start = normal_end + crash_bars
    recovery_bars = periods - recovery_start
    if recovery_bars > 0:
        recovery_returns = np.random.normal(0.004, 0.02, recovery_bars)
        recovery_base = prices[recovery_start - 1]
        prices[recovery_start:] = recovery_base * np.exp(np.cumsum(recovery_returns))

    return prices


def _black_swan(periods: int, base: float) -> np.ndarray:
    """COVID-style: 50% crash in 2 days, slow multi-month recovery."""
    prices = np.zeros(periods)
    pre = int(periods * 0.4)
    returns = np.random.normal(0.0004, 0.01, pre)
    prices[:pre] = base * np.exp(np.cumsum(returns))

    crash_bars = max(int(periods * 0.02), 3)
    crash_returns = np.random.normal(-0.15, 0.05, crash_bars)
    crash_base = prices[pre - 1]
    prices[pre:pre + crash_bars] = crash_base * np.exp(np.cumsum(crash_returns))

    recovery_start = pre + crash_bars
    recovery_bars = periods - recovery_start
    if recovery_bars > 0:
        recovery_returns = np.random.normal(0.002, 0.015, recovery_bars)
        rec_base = prices[recovery_start - 1]
        prices[recovery_start:] = rec_base * np.exp(np.cumsum(recovery_returns))

    return prices


def _accumulation(periods: int, base: float) -> np.ndarray:
    """Low volatility consolidation. Tight range, decreasing volume."""
    volatility = 0.005
    returns = np.random.normal(0, volatility, periods)
    decay = np.linspace(1.0, 0.3, periods)
    returns *= decay
    prices = base * np.exp(np.cumsum(returns))
    return prices


def _prices_to_ohlcv(prices: np.ndarray, tf_minutes: int) -> pd.DataFrame:
    """Convert close prices to OHLCV DataFrame with realistic open/high/low/volume."""
    n = len(prices)
    now = datetime.now()
    timestamps = [now - timedelta(minutes=tf_minutes * (n - i)) for i in range(n)]

    volatility = np.std(np.diff(prices) / prices[:-1]) if n > 1 else 0.01
    spread = prices * volatility * 0.5

    opens = prices * (1 + np.random.uniform(-0.002, 0.002, n))
    highs = np.maximum(prices, opens) + np.abs(np.random.normal(0, 1, n)) * spread
    lows = np.minimum(prices, opens) - np.abs(np.random.normal(0, 1, n)) * spread
    volumes = np.random.lognormal(10, 1.5, n) * (prices / prices[0])

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))

    return df


def list_scenarios() -> list[str]:
    """Return all available scenario names."""
    return ["bull_run", "bear_market", "sideways_chop", "flash_crash", "black_swan", "accumulation"]
