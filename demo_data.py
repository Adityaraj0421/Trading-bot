"""
Demo data generator for offline testing.
Generates realistic-looking OHLCV data when exchange APIs are unreachable.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def generate_ohlcv(
    symbol: str = "BTC/USDT",
    periods: int = 200,
    timeframe_minutes: int = 60,
    start_price: float = 95000.0,
    volatility: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic properties:
    - Trending behavior (random walk with drift)
    - Mean-reverting volatility
    - Correlated volume spikes with price moves
    """
    rng = np.random.default_rng(seed)

    # Generate returns with slight trend and clustering
    returns = rng.normal(0.0001, volatility, periods)

    # Add some momentum / trend
    trend = np.cumsum(rng.normal(0, 0.002, periods))
    returns += trend * 0.01

    # Build close prices
    close = start_price * np.exp(np.cumsum(returns))

    # Build OHLCV
    high = close * (1 + rng.uniform(0.001, 0.02, periods))
    low = close * (1 - rng.uniform(0.001, 0.02, periods))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, periods)

    # Volume: higher on bigger moves
    base_volume = rng.lognormal(mean=8, sigma=0.5, size=periods)
    move_size = np.abs(returns) / volatility
    volume = base_volume * (1 + move_size * 2)

    # Timestamps
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [end - timedelta(minutes=timeframe_minutes * (periods - i - 1)) for i in range(periods)]

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    return df


if __name__ == "__main__":
    df = generate_ohlcv()
    print(f"Generated {len(df)} candles")
    print(f"Price range: ${df['low'].min():,.0f} - ${df['high'].max():,.0f}")
    print(f"Last close: ${df['close'].iloc[-1]:,.2f}")
    print(df.tail())
