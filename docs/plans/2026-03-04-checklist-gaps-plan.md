# Checklist Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close 7 Phase 9 trading-checklist gaps across context, decision, trigger, and risk layers.

**Architecture:** New analyzers/trigger added as standalone modules; ContextEngine wires them in; evaluate() gains a funding gate; agent.py gains BTC dominance gate + partial TP exit management. TDD throughout — write failing test, implement, pass, commit.

**Tech Stack:** Python 3.14, pytest, pandas, numpy, ruff. All tests run via `./venv/bin/python -m pytest`. Lint via `./venv/bin/python -m ruff check --fix`.

---

## Task 1: VolatilityAnalyzer

**Files:**
- Create: `context/volatility.py`
- Create: `tests/test_context_volatility.py`

### Step 1: Write failing tests

```python
# tests/test_context_volatility.py
import numpy as np
import pandas as pd

from context.volatility import VolatilityAnalyzer


def make_df(n: int = 60, atr_multiplier: float = 1.0) -> pd.DataFrame:
    """Build 1h OHLCV DataFrame with controllable ATR size."""
    closes = np.linspace(90000.0, 91000.0, n)
    spread = 500.0 * atr_multiplier
    return pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes + spread,
        "low":    closes - spread,
        "close":  closes,
        "volume": [1000.0] * n,
    })


class TestVolatilityAnalyzer:
    def test_returns_dict_with_key(self):
        result = VolatilityAnalyzer().analyze(make_df())
        assert "volatility_regime" in result

    def test_normal_regime_baseline(self):
        # Stable ATR throughout → ratio ≈ 1.0 → normal
        result = VolatilityAnalyzer().analyze(make_df(n=60))
        assert result["volatility_regime"] == "normal"

    def test_extreme_regime_when_atr_spikes(self):
        # First 44 bars low ATR, last 16 bars 10× ATR → ratio >> 2.5 → extreme
        n_low, n_high = 44, 16
        closes = np.linspace(90000.0, 91000.0, n_low + n_high)
        spread_low = 100.0
        spread_high = 10000.0
        spreads = [spread_low] * n_low + [spread_high] * n_high
        df = pd.DataFrame({
            "open":   closes * 0.999,
            "high":   closes + np.array(spreads),
            "low":    closes - np.array(spreads),
            "close":  closes,
            "volume": [1000.0] * (n_low + n_high),
        })
        result = VolatilityAnalyzer().analyze(df)
        assert result["volatility_regime"] == "extreme"

    def test_low_regime_when_atr_shrinks(self):
        # First 44 bars high ATR, last 16 bars 10× smaller → ratio << 0.5 → low
        n_high, n_low = 44, 16
        closes = np.linspace(90000.0, 91000.0, n_high + n_low)
        spread_high = 5000.0
        spread_low = 50.0
        spreads = [spread_high] * n_high + [spread_low] * n_low
        df = pd.DataFrame({
            "open":   closes * 0.999,
            "high":   closes + np.array(spreads),
            "low":    closes - np.array(spreads),
            "close":  closes,
            "volume": [1000.0] * (n_high + n_low),
        })
        result = VolatilityAnalyzer().analyze(df)
        assert result["volatility_regime"] == "low"

    def test_insufficient_data_returns_normal(self):
        result = VolatilityAnalyzer().analyze(make_df(n=10))
        assert result["volatility_regime"] == "normal"

    def test_none_returns_normal(self):
        result = VolatilityAnalyzer().analyze(None)
        assert result["volatility_regime"] == "normal"
```

### Step 2: Run to verify tests fail
```bash
./venv/bin/python -m pytest tests/test_context_volatility.py -v
```
Expected: `ModuleNotFoundError: No module named 'context.volatility'`

### Step 3: Implement VolatilityAnalyzer

```python
# context/volatility.py
"""VolatilityAnalyzer — classify current ATR vs rolling baseline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

_ATR_PERIOD = 14
_VOL_LOOKBACK = 30
_MIN_BARS = _ATR_PERIOD + _VOL_LOOKBACK  # 44


class VolatilityAnalyzer:
    """Computes 14-bar ATR% and classifies vs 30-bar rolling mean.

    Returns one of: "low", "normal", "elevated", "extreme".
    Falls back to "normal" on insufficient data.
    """

    def analyze(self, df: pd.DataFrame | None) -> dict[str, Any]:
        """Classify volatility regime from 1h OHLCV data.

        Args:
            df: 1h OHLCV DataFrame. Needs ≥44 rows. None → "normal".

        Returns:
            Dict with key ``volatility_regime``.
        """
        if df is None or len(df) < _MIN_BARS:
            return {"volatility_regime": "normal"}

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(_ATR_PERIOD).mean()
        atr_pct = atr / close

        rolling_mean = atr_pct.rolling(_VOL_LOOKBACK).mean()

        current = atr_pct.iloc[-1]
        baseline = rolling_mean.iloc[-1]

        if pd.isna(current) or pd.isna(baseline) or baseline == 0:
            return {"volatility_regime": "normal"}

        ratio = current / baseline

        if ratio < 0.5:
            regime = "low"
        elif ratio < 1.5:
            regime = "normal"
        elif ratio < 2.5:
            regime = "elevated"
        else:
            regime = "extreme"

        _log.debug("VolatilityAnalyzer: ATR%% ratio=%.2f → %s", ratio, regime)
        return {"volatility_regime": regime}
```

### Step 4: Run to verify tests pass
```bash
./venv/bin/python -m pytest tests/test_context_volatility.py -v
```
Expected: 6 tests PASS

### Step 5: Lint
```bash
./venv/bin/python -m ruff check --fix context/volatility.py tests/test_context_volatility.py
```
Expected: 0 errors

### Step 6: Commit
```bash
git add context/volatility.py tests/test_context_volatility.py
git commit -m "feat: VolatilityAnalyzer — classify ATR% vs rolling baseline"
```

---

## Task 2: SessionAnalyzer

**Files:**
- Create: `context/session.py`
- Create: `tests/test_context_session.py`

### Step 1: Write failing tests

```python
# tests/test_context_session.py
from datetime import UTC, datetime

from context.session import SessionAnalyzer


def dt(weekday: int, hour: int) -> datetime:
    """Build a UTC datetime on the given ISO weekday (0=Mon, 6=Sun) and hour."""
    # 2026-03-02 = Monday (weekday 0), use offset
    base_monday = datetime(2026, 3, 2, hour, 0, tzinfo=UTC)
    from datetime import timedelta
    return base_monday + timedelta(days=weekday)


class TestSessionAnalyzer:
    def test_us_session_full_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(0, 15))  # Monday 15:00 UTC
        assert result["session"] == "US"
        assert result["confidence_multiplier"] == 1.00

    def test_eu_session_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(0, 10))  # Monday 10:00 UTC
        assert result["session"] == "EU"
        assert result["confidence_multiplier"] == 0.90

    def test_asia_session_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(0, 4))  # Monday 04:00 UTC
        assert result["session"] == "Asia"
        assert result["confidence_multiplier"] == 0.75

    def test_weekend_saturday_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(5, 14))  # Saturday 14:00 UTC
        assert result["session"] == "weekend"
        assert result["confidence_multiplier"] == 0.60

    def test_weekend_sunday_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(6, 20))  # Sunday 20:00 UTC
        assert result["session"] == "weekend"
        assert result["confidence_multiplier"] == 0.60

    def test_us_eu_overlap_returns_us(self):
        # 14:00 UTC is in both EU (07-15) and US (13-22) → US wins (higher multiplier)
        result = SessionAnalyzer().analyze(now=dt(1, 14))
        assert result["session"] == "US"
        assert result["confidence_multiplier"] == 1.00

    def test_dead_zone_returns_asia(self):
        # 23:00 UTC is outside US/EU/Asia windows
        result = SessionAnalyzer().analyze(now=dt(2, 23))
        assert result["session"] == "Asia"
        assert result["confidence_multiplier"] == 0.75
```

### Step 2: Run to verify tests fail
```bash
./venv/bin/python -m pytest tests/test_context_session.py -v
```
Expected: `ModuleNotFoundError: No module named 'context.session'`

### Step 3: Implement SessionAnalyzer

```python
# context/session.py
"""SessionAnalyzer — maps UTC time to session confidence multiplier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class SessionAnalyzer:
    """Maps current UTC time to a confidence multiplier.

    US session (peak liquidity) = 1.00 multiplier.
    EU session = 0.90. Asia = 0.75. Weekend = 0.60.
    When sessions overlap, the higher multiplier wins.
    """

    def analyze(self, now: datetime | None = None) -> dict[str, Any]:
        """Return session label and confidence multiplier for the given time.

        Args:
            now: UTC datetime to evaluate. Defaults to datetime.now(UTC).

        Returns:
            Dict with keys ``session`` and ``confidence_multiplier``.
        """
        if now is None:
            now = datetime.now(UTC)

        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return {"session": "weekend", "confidence_multiplier": 0.60}

        hour = now.hour
        in_us = 13 <= hour < 22
        in_eu = 7 <= hour < 15
        in_asia = hour < 8

        if in_us:
            return {"session": "US", "confidence_multiplier": 1.00}
        if in_eu:
            return {"session": "EU", "confidence_multiplier": 0.90}
        if in_asia:
            return {"session": "Asia", "confidence_multiplier": 0.75}
        # Dead zone 22:00–00:00 UTC: thin liquidity, treat as Asia
        return {"session": "Asia", "confidence_multiplier": 0.75}
```

### Step 4: Run to verify tests pass
```bash
./venv/bin/python -m pytest tests/test_context_session.py -v
```
Expected: 7 tests PASS

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix context/session.py tests/test_context_session.py
git add context/session.py tests/test_context_session.py
git commit -m "feat: SessionAnalyzer — UTC session confidence multiplier"
```

---

## Task 3: prev_day_high/low in SwingAnalyzer

**Files:**
- Modify: `context/swing.py`
- Modify: `tests/test_context_swing.py` (may not exist — create if needed)

First check: `ls tests/test_context_swing.py` — if it doesn't exist, check `tests/test_swing.py`.

### Step 1: Write failing test

Add this test class to the existing swing test file (or create `tests/test_context_swing.py`):

```python
# Add to existing swing test file, or create tests/test_context_swing.py
import numpy as np
import pandas as pd
from datetime import UTC, datetime, timedelta

from context.swing import SwingAnalyzer


def make_df_with_dates(n_days: int = 3, bars_per_day: int = 6) -> pd.DataFrame:
    """4h OHLCV with a real DatetimeIndex spanning n_days."""
    n = n_days * bars_per_day
    closes = np.linspace(90000.0, 100000.0, n)
    start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    index = [start + timedelta(hours=4 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "open":   closes * 0.999,
            "high":   closes * 1.01,   # predictable pdh
            "low":    closes * 0.99,   # predictable pdl
            "close":  closes,
            "volume": [1000.0] * n,
        },
        index=pd.DatetimeIndex(index),
    )


class TestSwingAnalyzerPdh:
    def test_pdh_and_pdl_present_with_datetime_index(self):
        df = make_df_with_dates(n_days=3, bars_per_day=6)
        result = SwingAnalyzer().analyze(df)
        assert "pdh" in result["key_levels"]
        assert "pdl" in result["key_levels"]

    def test_pdh_greater_than_pdl(self):
        df = make_df_with_dates(n_days=3, bars_per_day=6)
        result = SwingAnalyzer().analyze(df)
        assert result["key_levels"]["pdh"] > result["key_levels"]["pdl"]

    def test_pdh_pdl_absent_without_datetime_index(self):
        # Integer-indexed df (existing tests use this) → no pdh/pdl
        n = 60
        closes = np.linspace(90000.0, 100000.0, n)
        df = pd.DataFrame({
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low":  closes * 0.995,
            "close": closes,
            "volume": [1000.0] * n,
        })
        result = SwingAnalyzer().analyze(df)
        assert "pdh" not in result["key_levels"]
        assert "pdl" not in result["key_levels"]

    def test_pdh_pdl_absent_with_only_one_day(self):
        df = make_df_with_dates(n_days=1, bars_per_day=6)
        result = SwingAnalyzer().analyze(df)
        assert "pdh" not in result["key_levels"]
```

### Step 2: Run to verify new tests fail
```bash
./venv/bin/python -m pytest tests/ -k "TestSwingAnalyzerPdh" -v
```
Expected: 4 FAIL — `pdh` key not in result

### Step 3: Implement pdh/pdl in SwingAnalyzer

At the end of `SwingAnalyzer.analyze()`, before the `return` statement, add:

```python
        # Compute prev_day_high / prev_day_low when DatetimeIndex is available
        try:
            dates = pd.DatetimeIndex(df.index).normalize()
            unique_dates = sorted(dates.unique())
            if len(unique_dates) >= 2:
                yesterday = unique_dates[-2]
                mask = dates == yesterday
                key_levels_dict = {
                    "support": support,
                    "resistance": resistance,
                    "poc": poc,
                    "pdh": float(df.loc[mask, "high"].max()),
                    "pdl": float(df.loc[mask, "low"].min()),
                }
            else:
                key_levels_dict = {"support": support, "resistance": resistance, "poc": poc}
        except Exception:
            key_levels_dict = {"support": support, "resistance": resistance, "poc": poc}
```

Replace the existing `return` dict to use `key_levels_dict`:
```python
        return {
            "swing_bias": bias,
            "allowed_directions": allowed,
            "key_levels": key_levels_dict,
            "confidence": round(confidence, 3),
        }
```

> **Note:** The existing `return` statement currently builds key_levels inline. Replace that entire block with the try/except above.

### Step 4: Run full swing + new tests
```bash
./venv/bin/python -m pytest tests/ -k "swing or TestSwingAnalyzerPdh" -v
```
Expected: all PASS (no regressions)

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix context/swing.py
git add context/swing.py tests/test_context_swing.py
git commit -m "feat: add prev_day_high/low to SwingAnalyzer key_levels"
```

---

## Task 4: Wire VolatilityAnalyzer + SessionAnalyzer into ContextEngine

**Files:**
- Modify: `context_engine.py`
- Modify: `tests/test_context_engine.py`

### Step 1: Write failing tests

Add to `tests/test_context_engine.py`:

```python
from datetime import UTC, datetime
from context.session import SessionAnalyzer


def us_session_time() -> datetime:
    return datetime(2026, 3, 2, 16, 0, tzinfo=UTC)  # Monday 16:00 UTC = US session


def weekend_time() -> datetime:
    return datetime(2026, 3, 7, 16, 0, tzinfo=UTC)  # Saturday 16:00 UTC


class TestContextEngineVolatilitySession:
    def test_volatility_regime_is_populated(self):
        engine = ContextEngine()
        snap = make_snapshot(bullish=True)
        ctx = engine.build(
            snap, funding_rate=None, net_whale_flow=None,
            oi_change_pct=None, price_change_pct=None,
            _now=us_session_time(),
        )
        assert ctx.volatility_regime in ("low", "normal", "elevated", "extreme")

    def test_us_session_does_not_reduce_confidence(self):
        engine = ContextEngine()
        snap = make_snapshot(bullish=True)
        ctx = engine.build(
            snap, funding_rate=None, net_whale_flow=None,
            oi_change_pct=None, price_change_pct=None,
            _now=us_session_time(),
        )
        assert ctx.tradeable is True

    def test_weekend_reduces_confidence_and_may_block(self):
        engine = ContextEngine()
        snap = make_sideways_snapshot()  # produces confidence=0.3
        ctx = engine.build(
            snap, funding_rate=None, net_whale_flow=None,
            oi_change_pct=None, price_change_pct=None,
            _now=weekend_time(),
        )
        # Sideways confidence 0.3 × weekend 0.60 = 0.18 < 0.30 threshold
        assert ctx.tradeable is False

    def test_confidence_is_scaled_by_session(self):
        engine = ContextEngine()
        snap = make_snapshot(bullish=True)
        ctx_us = engine.build(
            snap, funding_rate=None, net_whale_flow=None,
            oi_change_pct=None, price_change_pct=None,
            _now=us_session_time(),
        )
        ctx_weekend = engine.build(
            snap, funding_rate=None, net_whale_flow=None,
            oi_change_pct=None, price_change_pct=None,
            _now=weekend_time(),
        )
        assert ctx_us.confidence > ctx_weekend.confidence
```

### Step 2: Run to verify new tests fail
```bash
./venv/bin/python -m pytest tests/test_context_engine.py::TestContextEngineVolatilitySession -v
```
Expected: TypeError (unexpected `_now` kwarg) or AttributeError

### Step 3: Implement changes to ContextEngine

**In `__init__`**, add:
```python
from context.volatility import VolatilityAnalyzer
from context.session import SessionAnalyzer

# inside __init__:
self._vol = VolatilityAnalyzer()
self._session = SessionAnalyzer()
```

**Update `build()` signature** to add `_now: datetime | None = None`:
```python
def build(
    self,
    snapshot: DataSnapshot,
    funding_rate: float | None,
    net_whale_flow: float | None,
    oi_change_pct: float | None,
    price_change_pct: float | None,
    _now: datetime | None = None,
) -> ContextState:
```

**Inside `build()`**, replace the hardcoded `volatility_regime="normal"` block:
```python
        vol = self._vol.analyze(snapshot.df_1h)
        session = self._session.analyze(now=_now)

        # Scale confidence by session liquidity
        raw_confidence = swing["confidence"]
        confidence = round(raw_confidence * session["confidence_multiplier"], 3)

        allowed_directions = swing["allowed_directions"]
        tradeable = len(allowed_directions) > 0 and confidence >= 0.30

        return ContextState(
            context_id=context_id,
            swing_bias=swing["swing_bias"],
            allowed_directions=allowed_directions,
            volatility_regime=vol["volatility_regime"],   # was hardcoded "normal"
            funding_pressure=funding["funding_pressure"],
            whale_flow=whale["whale_flow"],
            oi_trend=oi["oi_trend"],
            key_levels=swing["key_levels"],
            risk_mode=self._risk_mode,
            confidence=confidence,
            tradeable=tradeable,
            valid_until=now + timedelta(minutes=_CONTEXT_WINDOW_MINUTES),
            updated_at=now,
        )
```

### Step 4: Update existing tests that don't pass `_now`

Existing tests call `engine.build(snap, ...)` without `_now`. They will now get a real-time session multiplier. This is fine — `_now` is optional. But tests that assert `ctx.tradeable is True` for bullish snapshots must still pass:
- Bullish `confidence ≈ 0.767–0.9`, worst multiplier = 0.60 (weekend) → `0.767 × 0.60 = 0.46 > 0.30`. Still tradeable.
- Sideways `confidence = 0.3`, US session multiplier = 1.0 → `0.3 × 1.0 = 0.3 ≥ 0.30`. Still tradeable if run during US hours. To avoid flakiness, inject `_now=us_session_time()` in any existing test asserting `tradeable=True` for neutral data.

Run full context engine tests:
```bash
./venv/bin/python -m pytest tests/test_context_engine.py -v
```
Fix any failures by adding `_now=us_session_time()` to the relevant test calls.

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix context_engine.py tests/test_context_engine.py
git add context_engine.py tests/test_context_engine.py
git commit -m "feat: wire VolatilityAnalyzer + SessionAnalyzer into ContextEngine"
```

---

## Task 5: Funding Extreme Gate in evaluate()

**Files:**
- Modify: `decision.py`
- Modify: `tests/test_decision.py`

### Step 1: Write failing tests

Find the existing decision test file:
```bash
ls tests/test_decision.py
```

Add these tests:

```python
# Add to tests/test_decision.py

def make_context(**overrides):
    """Build a minimal tradeable ContextState. Pass keyword overrides."""
    from datetime import UTC, datetime, timedelta
    from decision import ContextState
    defaults = dict(
        context_id="2026-03-02T16:00Z",
        swing_bias="bullish",
        allowed_directions=["long"],
        volatility_regime="normal",
        funding_pressure="neutral",
        whale_flow="neutral",
        oi_trend="neutral",
        key_levels={"support": 90000.0, "resistance": 110000.0, "poc": 100000.0},
        risk_mode="normal",
        confidence=0.8,
        tradeable=True,
        valid_until=datetime.now(UTC) + timedelta(hours=1),
        updated_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return ContextState(**defaults)


def make_trigger(direction: str = "long", strength: float = 0.8, urgency: str = "normal"):
    from datetime import UTC, datetime, timedelta
    import uuid
    from decision import TriggerSignal
    return TriggerSignal(
        trigger_id=str(uuid.uuid4()),
        source="test",
        direction=direction,
        strength=strength,
        urgency=urgency,
        symbol_scope="BTC",
        reason="test",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


class TestFundingExtremeGate:
    def test_long_crowded_extreme_blocks_long_entry(self):
        ctx = make_context(funding_pressure="long_crowded_extreme")
        triggers = [make_trigger("long"), make_trigger("long")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "reject"
        assert "funding_extreme" in decision.reason

    def test_short_crowded_extreme_blocks_short_entry(self):
        ctx = make_context(
            swing_bias="bearish",
            allowed_directions=["short"],
            funding_pressure="short_crowded_extreme",
        )
        triggers = [make_trigger("short"), make_trigger("short")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "reject"
        assert "funding_extreme" in decision.reason

    def test_long_crowded_extreme_allows_short_entry(self):
        # Extreme long funding → only blocks longs, not shorts
        ctx = make_context(
            swing_bias="bearish",
            allowed_directions=["short"],
            funding_pressure="long_crowded_extreme",
        )
        triggers = [make_trigger("short"), make_trigger("short")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "trade"
        assert decision.direction == "short"

    def test_long_crowded_mild_does_not_block(self):
        ctx = make_context(funding_pressure="long_crowded_mild")
        triggers = [make_trigger("long"), make_trigger("long")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "trade"
```

### Step 2: Run to verify new tests fail
```bash
./venv/bin/python -m pytest tests/test_decision.py::TestFundingExtremeGate -v
```
Expected: 2 FAIL (blocks return trade instead of reject), 2 PASS

### Step 3: Implement funding gate in evaluate()

In `decision.py`, inside `evaluate()`, insert after the `not context.allowed_directions` check (after line ~191) and before the trigger filtering loop:

```python
    # Step 1.5: Funding extreme gate — block trading the crowded side
    effective_allowed = list(context.allowed_directions)
    if context.funding_pressure == "long_crowded_extreme":
        effective_allowed = [d for d in effective_allowed if d != "long"]
    if context.funding_pressure == "short_crowded_extreme":
        effective_allowed = [d for d in effective_allowed if d != "short"]
    if not effective_allowed:
        return Decision(action="reject", reason="funding_extreme_blocks_direction")
```

Then in Step 2 (trigger filter), replace `context.allowed_directions` with `effective_allowed`:
```python
    valid = [
        t for t in triggers
        if t.direction in effective_allowed and not t.is_expired()
    ]
```

### Step 4: Run full decision tests
```bash
./venv/bin/python -m pytest tests/test_decision.py -v
```
Expected: all PASS

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix decision.py tests/test_decision.py
git add decision.py tests/test_decision.py
git commit -m "feat: funding extreme gate in evaluate() — block crowded-side entries"
```

---

## Task 6: LiquiditySweepTrigger

**Files:**
- Create: `triggers/liquidity_sweep.py`
- Modify: `tests/test_triggers.py` (add new class)

### Step 1: Write failing tests

```python
# Add to tests/test_triggers.py

import numpy as np
import pandas as pd
from triggers.liquidity_sweep import LiquiditySweepTrigger


def make_sweep_long_df() -> pd.DataFrame:
    """20 bars declining to form equal lows, then a sweep bar that wicks below and closes above."""
    n = 21
    # Declining channel: multiple bars touching ~88000 low (equal lows)
    closes = [90000.0] * 10 + [89000.0] * 9 + [89500.0]  # recovery close on last bar
    lows =   [88100.0] * 10 + [88050.0] * 9 + [87500.0]  # last bar wicks below zone
    highs =  [91000.0] * 10 + [90000.0] * 9 + [90000.0]
    opens =  [c * 0.999 for c in closes]
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1000.0] * n,
    })


def make_sweep_short_df() -> pd.DataFrame:
    """20 bars rising to form equal highs, then a sweep bar that wicks above and closes below."""
    n = 21
    closes = [95000.0] * 10 + [96000.0] * 9 + [95500.0]  # rejection close
    highs =  [97100.0] * 10 + [97050.0] * 9 + [97600.0]  # last bar wicks above zone
    lows =   [94000.0] * 10 + [95000.0] * 9 + [94000.0]
    opens =  [c * 1.001 for c in closes]
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1000.0] * n,
    })


def make_no_sweep_df() -> pd.DataFrame:
    """Trending up bars — no equal highs/lows cluster."""
    closes = np.linspace(88000.0, 96000.0, 21)
    return pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": [1000.0] * 21,
    })


class TestLiquiditySweepTrigger:
    def test_bullish_sweep_fires_long_signal(self):
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_long_df())
        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].source == "liquidity_sweep"
        assert signals[0].strength == 0.65

    def test_bearish_sweep_fires_short_signal(self):
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_short_df())
        assert len(signals) == 1
        assert signals[0].direction == "short"

    def test_no_sweep_returns_empty(self):
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_no_sweep_df())
        assert signals == []

    def test_insufficient_data_returns_empty(self):
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_no_sweep_df().head(10))
        assert signals == []

    def test_signal_not_expired(self):
        from datetime import UTC, datetime
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_long_df())
        assert len(signals) == 1
        assert signals[0].expires_at > datetime.now(UTC)
```

### Step 2: Run to verify tests fail
```bash
./venv/bin/python -m pytest tests/test_triggers.py::TestLiquiditySweepTrigger -v
```
Expected: `ModuleNotFoundError: No module named 'triggers.liquidity_sweep'`

### Step 3: Implement LiquiditySweepTrigger

```python
# triggers/liquidity_sweep.py
"""LiquiditySweepTrigger — equal highs/lows sweep detection.

Fires when price wicks through an equal-highs or equal-lows cluster
(≥2 bars within 0.3%) and closes back inside — indicating a stop-hunt
reversal rather than a genuine breakout.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_LOOKBACK = 20
_CLUSTER_TOL = 0.003   # 0.3% tolerance for "equal" highs/lows
_MIN_BARS = _LOOKBACK + 1
_SIGNAL_TTL_MINUTES = 75
_SWEEP_STRENGTH = 0.65


class LiquiditySweepTrigger:
    """Detects equal-highs/lows sweep patterns on 1h OHLCV data.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Sets symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(self, df: pd.DataFrame) -> list[TriggerSignal]:
        """Scan the last 20 bars for sweep patterns.

        Args:
            df: 1h OHLCV DataFrame. Needs ≥21 rows.

        Returns:
            List of TriggerSignal (empty if no sweep detected).
        """
        if len(df) < _MIN_BARS:
            return []

        window = df.tail(_LOOKBACK)
        current = df.iloc[-1]

        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)
        signals: list[TriggerSignal] = []

        # Equal highs → bearish sweep zone
        equal_highs_zone = self._find_extreme_cluster(window["high"].values, find_max=True)
        if equal_highs_zone is not None:
            if current["high"] > equal_highs_zone and current["close"] < equal_highs_zone:
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="liquidity_sweep",
                        direction="short",
                        strength=_SWEEP_STRENGTH,
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=(
                            f"Bearish sweep: wick {current['high']:.0f}"
                            f" > zone {equal_highs_zone:.0f},"
                            f" close {current['close']:.0f}"
                        ),
                        expires_at=expiry,
                        raw_data={
                            "type": "equal_highs_sweep",
                            "zone": round(float(equal_highs_zone), 2),
                        },
                    )
                )

        # Equal lows → bullish sweep zone
        equal_lows_zone = self._find_extreme_cluster(window["low"].values, find_max=False)
        if equal_lows_zone is not None:
            if current["low"] < equal_lows_zone and current["close"] > equal_lows_zone:
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="liquidity_sweep",
                        direction="long",
                        strength=_SWEEP_STRENGTH,
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=(
                            f"Bullish sweep: wick {current['low']:.0f}"
                            f" < zone {equal_lows_zone:.0f},"
                            f" close {current['close']:.0f}"
                        ),
                        expires_at=expiry,
                        raw_data={
                            "type": "equal_lows_sweep",
                            "zone": round(float(equal_lows_zone), 2),
                        },
                    )
                )

        return signals

    def _find_extreme_cluster(
        self, values: np.ndarray, find_max: bool
    ) -> float | None:
        """Return the extreme value if ≥2 bars cluster within _CLUSTER_TOL%.

        For equal highs (find_max=True): clusters near the maximum.
        For equal lows (find_max=False): clusters near the minimum.

        Args:
            values: Array of high or low prices.
            find_max: True to find equal highs, False for equal lows.

        Returns:
            The extreme zone price, or None if no cluster found.
        """
        extreme = float(values.max() if find_max else values.min())
        if extreme == 0.0:
            return None
        cluster_count = int(np.sum(np.abs(values - extreme) / extreme <= _CLUSTER_TOL))
        return extreme if cluster_count >= 2 else None
```

### Step 4: Run trigger tests
```bash
./venv/bin/python -m pytest tests/test_triggers.py -v
```
Expected: all existing tests PASS + 5 new tests PASS

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix triggers/liquidity_sweep.py tests/test_triggers.py
git add triggers/liquidity_sweep.py tests/test_triggers.py
git commit -m "feat: LiquiditySweepTrigger — equal highs/lows stop-hunt reversal signal"
```

---

## Task 7: Wire LiquiditySweepTrigger into TriggerEngine

**Files:**
- Modify: `trigger_engine.py`
- Modify: `tests/test_trigger_engine.py` (check it exists first: `ls tests/test_trigger_engine.py`)

### Step 1: Write failing test

```python
# Add to tests/test_trigger_engine.py (or create it)
import numpy as np
import pandas as pd
from trigger_engine import TriggerEngine


def make_sweep_df_for_engine() -> pd.DataFrame:
    """21-bar DataFrame with a bullish sweep on the last bar."""
    n = 21
    closes = [89500.0] * 10 + [89000.0] * 9 + [89500.0]
    lows   = [88100.0] * 10 + [88050.0] * 9 + [87500.0]  # sweep bar
    highs  = [91000.0] * 10 + [90000.0] * 9 + [90000.0]
    opens  = [c * 0.999 for c in closes]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1000.0] * n,
    })


class TestTriggerEngineSweep:
    def test_sweep_signals_buffered_from_on_1h_close(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        new = engine.on_1h_close(make_sweep_df_for_engine())
        sweep_signals = [s for s in new if s.source == "liquidity_sweep"]
        # At least one sweep signal should be generated
        assert len(sweep_signals) >= 1

    def test_sweep_trigger_instantiated_per_symbol(self):
        from triggers.liquidity_sweep import LiquiditySweepTrigger
        engine = TriggerEngine(symbol="ETH/USDT")
        assert hasattr(engine, "_sweep")
        assert isinstance(engine._sweep, LiquiditySweepTrigger)
        assert engine._sweep.symbol == "ETH/USDT"
```

### Step 2: Run to verify tests fail
```bash
./venv/bin/python -m pytest tests/test_trigger_engine.py::TestTriggerEngineSweep -v
```
Expected: AttributeError (`_sweep` not found)

### Step 3: Implement wiring in TriggerEngine

In `trigger_engine.py`:

**Add import at top:**
```python
from triggers.liquidity_sweep import LiquiditySweepTrigger
```

**In `__init__`, after `self._orderflow = OrderFlowTrigger(symbol=symbol)`:**
```python
        self._sweep = LiquiditySweepTrigger(symbol=symbol)
```

**In `on_1h_close()`, update the new_signals aggregation:**
```python
    def on_1h_close(self, df: pd.DataFrame) -> list[TriggerSignal]:
        new_signals = self._momentum.evaluate(df) + self._sweep.evaluate(df)
        self._extend(new_signals)
        if new_signals:
            _log.info(
                "TriggerEngine[%s] on_1h_close: %d new signal(s)",
                self.symbol,
                len(new_signals),
            )
        return new_signals
```

### Step 4: Run all trigger engine tests
```bash
./venv/bin/python -m pytest tests/test_trigger_engine.py -v
```
Expected: all PASS

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix trigger_engine.py tests/test_trigger_engine.py
git add trigger_engine.py tests/test_trigger_engine.py
git commit -m "feat: wire LiquiditySweepTrigger into TriggerEngine.on_1h_close"
```

---

## Task 8: Position partial_tp_levels + RiskManager.check_partial_tp()

**Files:**
- Modify: `risk_manager.py`
- Modify: `tests/test_risk_manager.py`

### Step 1: Write failing tests

```python
# Add to tests/test_risk_manager.py

class TestPartialTP:
    def _make_long_position(self, entry=100_000.0, sl=98_000.0, qty=0.1):
        from risk_manager import Position
        from datetime import UTC, datetime
        return Position(
            symbol="BTC/USDT",
            side="long",
            entry_price=entry,
            quantity=qty,
            entry_time=datetime.now(UTC),
            stop_loss=sl,
            take_profit=105_000.0,
            partial_tp_levels=[102_000.0, 104_000.0],
        )

    def _make_short_position(self, entry=100_000.0, sl=102_000.0, qty=0.1):
        from risk_manager import Position
        from datetime import UTC, datetime
        return Position(
            symbol="BTC/USDT",
            side="short",
            entry_price=entry,
            quantity=qty,
            entry_time=datetime.now(UTC),
            stop_loss=sl,
            take_profit=96_000.0,
            partial_tp_levels=[98_000.0, 96_500.0],
        )

    def test_returns_fraction_when_long_price_reaches_level(self):
        from risk_manager import RiskManager
        rm = RiskManager(symbol="BTC/USDT")
        pos = self._make_long_position()
        fraction = rm.check_partial_tp(pos, 102_500.0, {})
        assert fraction == 0.50

    def test_returns_none_when_long_price_below_level(self):
        from risk_manager import RiskManager
        rm = RiskManager(symbol="BTC/USDT")
        pos = self._make_long_position()
        fraction = rm.check_partial_tp(pos, 101_000.0, {})
        assert fraction is None

    def test_marks_level_used_to_prevent_refire(self):
        from risk_manager import RiskManager
        rm = RiskManager(symbol="BTC/USDT")
        pos = self._make_long_position()
        rm.check_partial_tp(pos, 102_500.0, {})   # fires first level
        fraction2 = rm.check_partial_tp(pos, 102_500.0, {})  # same price, already used
        assert fraction2 is None   # first level consumed; second level (104k) not reached

    def test_short_triggers_on_price_below_level(self):
        from risk_manager import RiskManager
        rm = RiskManager(symbol="BTC/USDT")
        pos = self._make_short_position()
        fraction = rm.check_partial_tp(pos, 97_500.0, {})
        assert fraction == 0.50

    def test_empty_partial_tp_levels_returns_none(self):
        from risk_manager import RiskManager, Position
        from datetime import UTC, datetime
        rm = RiskManager(symbol="BTC/USDT")
        pos = Position(
            symbol="BTC/USDT", side="long",
            entry_price=100_000.0, quantity=0.1,
            entry_time=datetime.now(UTC),
            stop_loss=98_000.0, take_profit=105_000.0,
        )
        assert rm.check_partial_tp(pos, 105_000.0, {}) is None
```

### Step 2: Run to verify tests fail
```bash
./venv/bin/python -m pytest tests/test_risk_manager.py::TestPartialTP -v
```
Expected: TypeError (`partial_tp_levels` unknown field)

### Step 3: Add fields to Position

In `risk_manager.py`, in the `Position` dataclass, add after existing fields:
```python
    partial_tp_levels: list[float] = field(default_factory=list)
    partial_exits: list[dict] = field(default_factory=list)
```

### Step 4: Add check_partial_tp() to RiskManager

Add this method to the `RiskManager` class (after `calculate_stop_take()`):

```python
    def check_partial_tp(
        self,
        position: Position,
        current_price: float,
        key_levels: dict,
    ) -> float | None:
        """Return 0.50 exit fraction if the next unused partial TP level is hit.

        Args:
            position: The open Position (mutated to record the exit).
            current_price: Latest price for the symbol.
            key_levels: ContextState key_levels dict (unused currently, reserved).

        Returns:
            0.50 if a partial TP fires, None otherwise.
        """
        if not position.partial_tp_levels:
            return None

        used_levels = {e["level"] for e in position.partial_exits}

        for level in position.partial_tp_levels:
            if level in used_levels:
                continue
            if position.side == "long" and current_price >= level:
                position.partial_exits.append(
                    {"level": level, "price": current_price}
                )
                _log.info(
                    "[PartialTP] %s long hit %.2f @ %.2f",
                    position.symbol, level, current_price,
                )
                return 0.50
            if position.side == "short" and current_price <= level:
                position.partial_exits.append(
                    {"level": level, "price": current_price}
                )
                _log.info(
                    "[PartialTP] %s short hit %.2f @ %.2f",
                    position.symbol, level, current_price,
                )
                return 0.50

        return None
```

### Step 5: Run full risk manager tests
```bash
./venv/bin/python -m pytest tests/test_risk_manager.py -v
```
Expected: all existing + new PASS

### Step 6: Lint and commit
```bash
./venv/bin/python -m ruff check --fix risk_manager.py tests/test_risk_manager.py
git add risk_manager.py tests/test_risk_manager.py
git commit -m "feat: Position.partial_tp_levels + RiskManager.check_partial_tp()"
```

---

## Task 9: Executor.partial_close()

**Files:**
- Modify: `executor.py`
- Modify: `tests/test_executor.py`

### Step 1: Write failing test

```python
# Add to tests/test_executor.py

class TestPartialClose:
    def _make_position(self):
        from risk_manager import Position
        from datetime import UTC, datetime
        return Position(
            symbol="BTC/USDT", side="long",
            entry_price=100_000.0, quantity=1.0,
            entry_time=datetime.now(UTC),
            stop_loss=98_000.0, take_profit=105_000.0,
        )

    def test_partial_close_reduces_quantity(self):
        from executor import PaperExecutor
        pos = self._make_position()
        executor = PaperExecutor()
        executor.partial_close(pos, fraction=0.5, current_price=102_000.0, reason="partial_tp")
        assert abs(pos.quantity - 0.5) < 1e-9

    def test_partial_close_records_order(self):
        from executor import PaperExecutor
        pos = self._make_position()
        executor = PaperExecutor()
        order = executor.partial_close(pos, fraction=0.5, current_price=102_000.0, reason="partial_tp")
        assert order["status"] == "filled"
        assert abs(order["quantity"] - 0.5) < 1e-9
        assert order["reason"] == "partial_tp"

    def test_partial_close_pnl_positive_for_long_above_entry(self):
        from executor import PaperExecutor
        pos = self._make_position()
        executor = PaperExecutor()
        order = executor.partial_close(pos, fraction=0.5, current_price=102_000.0, reason="partial_tp")
        # PnL = (102000 - 100000) * 0.5 = 1000
        assert order["pnl"] == pytest.approx(1000.0)
```

### Step 2: Run to verify tests fail
```bash
./venv/bin/python -m pytest tests/test_executor.py::TestPartialClose -v
```
Expected: AttributeError (`partial_close` not found)

### Step 3: Add partial_close() to PaperExecutor

In `executor.py`, add after `cancel_order()` in `PaperExecutor`:

```python
    def partial_close(
        self,
        position: Any,
        fraction: float,
        current_price: float,
        reason: str = "partial_tp",
    ) -> dict[str, Any]:
        """Simulate a partial position close.

        Reduces ``position.quantity`` by ``fraction`` and records the
        closed portion as a filled order. PnL is calculated for the
        closed quantity only.

        Args:
            position: Open Position object (quantity is mutated).
            fraction: Fraction to close, e.g. 0.50 for 50%.
            current_price: Simulated fill price.
            reason: Tag for audit trail (e.g. ``"partial_tp_1"``).

        Returns:
            Order dict with keys including ``pnl`` and ``reason``.
        """
        closed_qty = position.quantity * fraction
        if position.side == "long":
            pnl = (current_price - position.entry_price) * closed_qty
        else:
            pnl = (position.entry_price - current_price) * closed_qty

        position.quantity -= closed_qty

        order = {
            "id": f"paper_partial_{len(self.orders) + 1}",
            "symbol": position.symbol,
            "side": "sell" if position.side == "long" else "buy",
            "quantity": closed_qty,
            "price": current_price,
            "pnl": pnl,
            "reason": reason,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER",
        }
        self.orders.append(order)
        _log.info(
            "[Paper] Partial close %.0f%% of %s @ $%,.2f (PnL: $%.2f)",
            fraction * 100,
            position.symbol,
            current_price,
            pnl,
        )
        return order
```

### Step 4: Run executor tests
```bash
./venv/bin/python -m pytest tests/test_executor.py -v
```
Expected: all PASS

### Step 5: Lint and commit
```bash
./venv/bin/python -m ruff check --fix executor.py tests/test_executor.py
git add executor.py tests/test_executor.py
git commit -m "feat: PaperExecutor.partial_close() — partial position scale-out"
```

---

## Task 10: Agent wiring — BTC dominance gate + phase9 context cache + partial TP exits

**Files:**
- Modify: `agent.py`

This task has no new test file — the agent is integration-tested via the API. After changes, run the full test suite.

### Step 1: Locate the three sections to modify

```bash
grep -n "_btc_swing_bias\|_phase9_context\|_run_phase9_cycle\|_run_pair_cycle\|_manage_exits\|_execute_trade" agent.py | head -40
```

### Step 2: Add attributes to __init__

Find `__init__` in `TradingAgent`. After existing `self._last_df_ind: dict = {}` (or similar per-pair caches), add:

```python
        # Phase 9: cross-pair BTC dominance gate
        self._btc_swing_bias: str = "neutral"
        # Phase 9: per-symbol latest ContextState (for partial TP key_levels)
        self._phase9_context: dict = {}
```

### Step 3: Store ContextState and update BTC bias in _run_phase9_cycle()

Find the section in `_run_phase9_cycle()` where `context = context_engine.build(...)` is called.

After the `context = ...` line, add:
```python
            # Cache context for exit management (partial TP key_levels)
            self._phase9_context[symbol] = context

            # Update BTC swing bias for cross-pair dominance gate
            if symbol == "BTC/USDT":
                self._btc_swing_bias = context.swing_bias
```

### Step 4: Add BTC dominance gate before evaluate()

In `_run_phase9_cycle()`, find the loop that iterates over triggers and calls `evaluate()`. Before the `evaluate()` call, add:

```python
                # BTC dominance gate (SOL only)
                if symbol == "SOL/USDT":
                    sol_direction = triggers[0].direction if triggers else None
                    if (
                        self._btc_swing_bias == "bearish"
                        and sol_direction == "long"
                    ) or (
                        self._btc_swing_bias == "bullish"
                        and sol_direction == "short"
                    ):
                        _log.info(
                            "[Phase9] SOL entry blocked by BTC dominance"
                            " (btc_bias=%s, sol_direction=%s)",
                            self._btc_swing_bias,
                            sol_direction,
                        )
                        continue
```

> **Note:** The exact loop structure depends on how `_run_phase9_cycle()` calls `evaluate()`. Read the current code around the `evaluate()` call and insert the gate immediately before it (outside the evaluate call, before its arguments are passed).

### Step 5: Add partial TP check in _run_pair_cycle() exit management

Find the section in `_run_pair_cycle()` that iterates open positions for exit management. After the trailing stop check and before the full TP check, add:

```python
                    # Partial TP at structural levels
                    if context := self._phase9_context.get(symbol):
                        frac = self.risk_manager.check_partial_tp(
                            position, current_price, context.key_levels
                        )
                        if frac is not None:
                            self.executor.partial_close(
                                position, frac, current_price, reason="partial_tp"
                            )
```

> **Note:** `self.executor` may be `PaperExecutor` or `LiveExecutor`. `partial_close` is only defined on `PaperExecutor` for now. Wrap with `if hasattr(self.executor, "partial_close"):` until `LiveExecutor.partial_close()` is added.

### Step 6: Compute partial_tp_levels after _execute_trade()

Find `_execute_trade()` in `agent.py`. After the position is opened (after `self.risk_manager.open_position(...)` or wherever `position` is stored), add:

```python
            # Populate partial TP levels from context key_levels
            if symbol in self._phase9_context and hasattr(position, "partial_tp_levels"):
                key_levels = self._phase9_context[symbol].key_levels
                risk = abs(position.entry_price - position.stop_loss)
                candidates = []
                if position.side == "long":
                    for k in ("resistance", "pdh"):
                        lvl = key_levels.get(k)
                        if lvl and lvl > position.entry_price + risk:
                            candidates.append(lvl)
                    candidates.sort()
                else:
                    for k in ("support", "pdl"):
                        lvl = key_levels.get(k)
                        if lvl and lvl < position.entry_price - risk:
                            candidates.append(lvl)
                    candidates.sort(reverse=True)
                position.partial_tp_levels = candidates[:2]
                if position.partial_tp_levels:
                    _log.info(
                        "[Phase9] Partial TP levels for %s: %s",
                        symbol, position.partial_tp_levels,
                    )
```

### Step 7: Run full test suite
```bash
./venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: all 988+ tests PASS (new tests bring total to ~1010+)

### Step 8: Lint all modified files
```bash
./venv/bin/python -m ruff check --fix agent.py
```
Expected: 0 errors

### Step 9: Final commit
```bash
git add agent.py
git commit -m "feat: agent — BTC dominance gate, phase9 context cache, partial TP exits"
```

---

## Final Verification

```bash
# Run complete test suite
./venv/bin/python -m pytest tests/ -v 2>&1 | tail -20

# Confirm 0 lint errors across all new/modified files
./venv/bin/python -m ruff check context/volatility.py context/session.py context/swing.py \
    context_engine.py decision.py triggers/liquidity_sweep.py trigger_engine.py \
    risk_manager.py executor.py agent.py
```

Expected:
- All tests PASS (no regressions)
- 0 ruff errors
- Git log shows 10 clean commits

---

## Summary of New Files and Changes

| File | Type | Summary |
|------|------|---------|
| `context/volatility.py` | New | ATR% regime classifier |
| `context/session.py` | New | UTC session confidence multiplier |
| `triggers/liquidity_sweep.py` | New | Equal highs/lows sweep detection |
| `tests/test_context_volatility.py` | New | 6 tests |
| `tests/test_context_session.py` | New | 7 tests |
| `context/swing.py` | Modified | pdh/pdl in key_levels |
| `context_engine.py` | Modified | Wire vol + session, `_now` param |
| `decision.py` | Modified | Funding extreme gate |
| `trigger_engine.py` | Modified | Wire LiquiditySweepTrigger |
| `risk_manager.py` | Modified | partial_tp_levels + check_partial_tp |
| `executor.py` | Modified | partial_close() |
| `agent.py` | Modified | BTC bias, phase9 context, partial TP |
