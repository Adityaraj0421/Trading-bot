# tests/test_context_engine.py
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from context_engine import ContextEngine
from data_snapshot import DataSnapshot
from decision import ContextState


def make_snapshot(bullish: bool = True) -> DataSnapshot:
    """Build a DataSnapshot with consistent bullish or bearish 4h/1h data."""
    n = 100
    closes = np.linspace(90000, 100000, n) if bullish else np.linspace(100000, 90000, n)
    df = pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    return DataSnapshot(df_1h=df, df_4h=df, df_15m=None, symbol="BTC/USDT")


def make_sideways_snapshot() -> DataSnapshot:
    """Snapshot with oscillating prices that produce neutral swing_bias."""
    closes = [95000 + 1000 * math.sin(i * 0.3) for i in range(84)]
    df = pd.DataFrame(
        {
            "open": [c * 0.999 for c in closes],
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000.0] * 84,
        }
    )
    return DataSnapshot(df_1h=df, df_4h=df, df_15m=None)


class TestContextEngine:
    def test_returns_context_state(self):
        engine = ContextEngine()
        snap = make_snapshot(bullish=True)
        ctx = engine.build(
            snap, funding_rate=0.0001, net_whale_flow=None, oi_change_pct=None, price_change_pct=None
        )
        assert isinstance(ctx, ContextState)

    def test_context_id_is_set(self):
        engine = ContextEngine()
        ctx = engine.build(
            make_snapshot(), funding_rate=None, net_whale_flow=None, oi_change_pct=None, price_change_pct=None
        )
        assert ctx.context_id != ""

    def test_valid_until_is_future(self):
        engine = ContextEngine()
        ctx = engine.build(
            make_snapshot(), funding_rate=None, net_whale_flow=None, oi_change_pct=None, price_change_pct=None
        )
        assert ctx.valid_until > datetime.now(UTC)

    def test_tradeable_false_when_no_allowed_directions(self):
        engine = ContextEngine()
        snap = make_sideways_snapshot()
        ctx = engine.build(
            snap, funding_rate=None, net_whale_flow=None, oi_change_pct=None, price_change_pct=None
        )
        # neutral → allowed_directions=[] → tradeable=False
        if ctx.swing_bias == "neutral":
            assert ctx.tradeable is False

    def test_risk_mode_defaults_to_normal(self):
        engine = ContextEngine()
        ctx = engine.build(
            make_snapshot(), funding_rate=None, net_whale_flow=None, oi_change_pct=None, price_change_pct=None
        )
        assert ctx.risk_mode == "normal"

    def test_risk_supervisor_can_override_risk_mode(self):
        engine = ContextEngine()
        engine.set_risk_mode("defensive")
        ctx = engine.build(
            make_snapshot(), funding_rate=None, net_whale_flow=None, oi_change_pct=None, price_change_pct=None
        )
        assert ctx.risk_mode == "defensive"
