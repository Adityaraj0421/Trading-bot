# Phase 9: Context + Trigger Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the lagging ensemble/regime system with a Context + Trigger architecture
that separates structural market state (Context Engine, 15min) from entry opportunities
(Trigger Engine, per-candle + events), with deterministic routing to spot or perp execution.

**Architecture:** Context Engine builds `ContextState` from 4h swing bias, funding pressure,
whale flow, and OI trend. Trigger Engine fires `TriggerSignal[]` from momentum, order flow,
liquidation cascades, and funding extremes. Decision Layer gates on context, scores
on trigger agreement, routes deterministically to spot or perp execution.

**Tech Stack:** Python 3.14, CCXT (spot + perp), pandas, existing intelligence/ data providers
(reused for raw data), pytest, ruff. No new ML libraries.

**Design doc:** `docs/plans/2026-03-03-context-trigger-architecture-design.md`

**Run tests:** `./venv/bin/python -m pytest tests/ -v`
**Run single test file:** `./venv/bin/python -m pytest tests/test_<name>.py -v`
**Lint:** `./venv/bin/python -m ruff check --fix <file>`
**Activate venv:** `source venv/bin/activate` (or prefix all python calls with `./venv/bin/python`)

---

## Phase 1 — Frozen Interfaces

> Goal: Define the three core schemas. Zero logic. Everything downstream depends on these.

---

### Task 1: Create `decision.py` with frozen schemas

**Files:**
- Create: `decision.py`
- Create: `tests/test_decision_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_decision_schemas.py
from datetime import UTC, datetime, timedelta
from decision import ContextState, Decision, TriggerSignal
import pytest


def make_context(**overrides):
    defaults = dict(
        context_id="2026-03-03T14:15Z",
        swing_bias="bullish",
        allowed_directions=["long"],
        volatility_regime="normal",
        funding_pressure="neutral",
        whale_flow="neutral",
        oi_trend="neutral",
        key_levels={"support": 90000.0, "resistance": 95000.0, "poc": 92000.0},
        risk_mode="normal",
        confidence=0.75,
        tradeable=True,
        valid_until=datetime.now(UTC) + timedelta(minutes=15),
        updated_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return ContextState(**defaults)


def make_trigger(**overrides):
    defaults = dict(
        trigger_id="abc-123",
        source="momentum_1h",
        direction="long",
        strength=0.7,
        urgency="normal",
        symbol_scope="BTC",
        reason="RSI crossed 50 upward + volume 1.8x",
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        raw_data={"rsi": 52.0, "volume_ratio": 1.8},
    )
    defaults.update(overrides)
    return TriggerSignal(**defaults)


class TestContextState:
    def test_valid_construction(self):
        ctx = make_context()
        assert ctx.swing_bias == "bullish"
        assert ctx.tradeable is True
        assert ctx.confidence == 0.75

    def test_invalid_swing_bias_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            make_context(swing_bias="sideways")

    def test_allowed_directions_can_be_empty(self):
        ctx = make_context(allowed_directions=[])
        assert ctx.allowed_directions == []

    def test_tradeable_is_bool(self):
        ctx = make_context(tradeable=False)
        assert ctx.tradeable is False


class TestTriggerSignal:
    def test_valid_construction(self):
        t = make_trigger()
        assert t.source == "momentum_1h"
        assert t.strength == 0.7

    def test_symbol_scope_market(self):
        t = make_trigger(symbol_scope="market")
        assert t.symbol_scope == "market"


class TestDecision:
    def test_reject_is_frozen(self):
        d = Decision(action="reject", reason="context_not_tradeable")
        with pytest.raises((AttributeError, TypeError)):
            d.reason = "mutated"

    def test_trade_decision(self):
        d = Decision(action="trade", direction="long", route="spot", score=0.63, reason="ok")
        assert d.action == "trade"
        assert d.route == "spot"

    def test_reject_has_no_direction(self):
        d = Decision(action="reject", reason="no_context")
        assert d.direction is None
        assert d.score is None
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_decision_schemas.py -v
```
Expected: `ModuleNotFoundError: No module named 'decision'`

**Step 3: Write `decision.py`**

```python
"""
Decision Layer — Core schemas for Context + Trigger architecture.

Three frozen dataclasses define the contract between all system components:
  ContextState  — structural market state, produced by ContextEngine every 15min
  TriggerSignal — entry opportunity, produced by TriggerEngine per-candle/event
  Decision      — trade or reject verdict, produced by evaluate()

These schemas are the system spine. No downstream component mutates them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Optional

# --- Literal type aliases (prevents string creep) ---

SwingBias = Literal["bullish", "bearish", "neutral"]
VolatilityRegime = Literal["low", "normal", "elevated", "extreme"]
FundingPressure = Literal[
    "long_crowded_mild", "long_crowded_extreme",
    "short_crowded_mild", "short_crowded_extreme",
    "neutral",
]
WhaleFlow = Literal["accumulating", "distributing", "neutral"]
OITrend = Literal["expanding_up", "expanding_down", "contracting", "neutral"]
RiskMode = Literal["normal", "cautious", "defensive"]

_VALID_SWING_BIAS: set[str] = {"bullish", "bearish", "neutral"}
_VALID_VOL_REGIME: set[str] = {"low", "normal", "elevated", "extreme"}
_VALID_FUNDING: set[str] = {
    "long_crowded_mild", "long_crowded_extreme",
    "short_crowded_mild", "short_crowded_extreme",
    "neutral",
}
_VALID_WHALE: set[str] = {"accumulating", "distributing", "neutral"}
_VALID_OI: set[str] = {"expanding_up", "expanding_down", "contracting", "neutral"}
_VALID_RISK: set[str] = {"normal", "cautious", "defensive"}


@dataclass
class ContextState:
    """Structural market state snapshot produced by ContextEngine every 15 minutes.

    Immutable within its validity window (valid_until). If context changes,
    a new ContextState with a new context_id is created — this one is never mutated.

    Attributes:
        context_id: ISO-8601 string identifying this context window (e.g. "2026-03-03T14:15Z").
        swing_bias: 4h price structure direction.
        allowed_directions: Directions the Decision Layer may trade. Empty list = no trades.
        volatility_regime: Current volatility classification.
        funding_pressure: Perpetual funding rate signal.
        whale_flow: Net whale accumulation/distribution signal.
        oi_trend: Open interest trend direction.
        key_levels: Support, resistance, and point-of-control prices.
        risk_mode: Current operating mode from RiskSupervisor.
        confidence: 0.0–1.0 overall context clarity (informational; used in scoring).
        tradeable: Derived by ContextEngine. Never set by downstream components.
        valid_until: Datetime after which this ContextState is stale.
        updated_at: When this ContextState was produced.
    """

    context_id: str
    swing_bias: SwingBias
    allowed_directions: list[str]
    volatility_regime: VolatilityRegime
    funding_pressure: FundingPressure
    whale_flow: WhaleFlow
    oi_trend: OITrend
    key_levels: dict
    risk_mode: RiskMode
    confidence: float
    tradeable: bool
    valid_until: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.swing_bias not in _VALID_SWING_BIAS:
            raise ValueError(f"Invalid swing_bias: {self.swing_bias!r}")
        if self.volatility_regime not in _VALID_VOL_REGIME:
            raise ValueError(f"Invalid volatility_regime: {self.volatility_regime!r}")
        if self.funding_pressure not in _VALID_FUNDING:
            raise ValueError(f"Invalid funding_pressure: {self.funding_pressure!r}")
        if self.whale_flow not in _VALID_WHALE:
            raise ValueError(f"Invalid whale_flow: {self.whale_flow!r}")
        if self.oi_trend not in _VALID_OI:
            raise ValueError(f"Invalid oi_trend: {self.oi_trend!r}")
        if self.risk_mode not in _VALID_RISK:
            raise ValueError(f"Invalid risk_mode: {self.risk_mode!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0–1.0, got {self.confidence}")


@dataclass
class TriggerSignal:
    """Entry opportunity produced by TriggerEngine.

    Triggers are observations only — they know nothing about position size,
    instrument choice, or whether a trade will happen.

    Attributes:
        trigger_id: UUID for audit trail.
        source: Which trigger produced this signal.
        direction: "long" or "short".
        strength: 0.0–1.0 signal strength.
        urgency: "normal" (spot) or "high" (event-driven, routes to perp).
        symbol_scope: Symbol this trigger applies to, or "market" for system-wide.
        reason: Human-readable explanation for logging/debugging.
        expires_at: Trigger is stale after this datetime.
        raw_data: Source-specific forensic data for post-mortem analysis.
    """

    trigger_id: str
    source: str
    direction: str
    strength: float
    urgency: str
    symbol_scope: str
    reason: str
    expires_at: datetime
    raw_data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {self.direction!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be 0.0–1.0, got {self.strength}")
        if self.urgency not in ("normal", "high"):
            raise ValueError(f"urgency must be 'normal' or 'high', got {self.urgency!r}")

    def is_expired(self) -> bool:
        """Return True if this trigger has passed its expiry time."""
        return datetime.now(UTC) > self.expires_at


@dataclass(frozen=True)
class Decision:
    """Trade or reject verdict produced by evaluate().

    Pure data — no methods, no logic. Loggable and replayable.
    Execution layers never interpret intent beyond action/direction/route.

    Attributes:
        action: "trade" or "reject".
        reason: Human-readable explanation (always set, even for trades).
        direction: "long" | "short" | None (None when action="reject").
        route: "spot" | "perp" | None (None when action="reject").
        score: Combined context+trigger score (None when action="reject").
    """

    action: Literal["trade", "reject"]
    reason: str
    direction: Optional[str] = None
    route: Optional[str] = None
    score: Optional[float] = None


# --- Score threshold (module-level constant, not buried in evaluate()) ---
SCORE_THRESHOLD: float = 0.50


def evaluate(context: ContextState, triggers: list[TriggerSignal]) -> Decision:
    """Evaluate context + triggers and produce a trade or reject decision.

    Decision authority hierarchy:
      1. Gate (hard): context must allow trading and have valid directions.
      2. Consensus: 2+ triggers must agree on the same direction.
      3. Score: context.confidence × mean(trigger strengths) >= SCORE_THRESHOLD.
      4. Route: event triggers ("high" urgency) → perp; others → spot.
         Event routing is blocked in "defensive" risk_mode.

    Args:
        context: Current ContextState from ContextEngine.
        triggers: List of TriggerSignal from TriggerEngine (may include expired ones).

    Returns:
        Decision with action="trade" or action="reject" plus audit reason.
    """
    # Step 1: Gate — hard stop if context is not tradeable
    if not context.tradeable:
        return Decision(action="reject", reason="context_not_tradeable")
    if not context.allowed_directions:
        return Decision(action="reject", reason="no_allowed_directions")

    now = datetime.now(UTC)

    # Step 2: Filter to allowed directions + non-expired triggers
    valid = [
        t for t in triggers
        if t.direction in context.allowed_directions and not t.is_expired()
    ]

    if not valid:
        return Decision(action="reject", reason="no_valid_triggers")

    # Step 3: Directional consensus — 2+ triggers must agree on same direction
    by_dir: dict[str, list[TriggerSignal]] = {}
    for t in valid:
        by_dir.setdefault(t.direction, []).append(t)

    best_dir, agreeing = max(by_dir.items(), key=lambda g: len(g[1]))
    if len(agreeing) < 2:
        return Decision(action="reject", reason="insufficient_directional_agreement")

    # Step 4: Score
    avg_strength = sum(t.strength for t in agreeing) / len(agreeing)
    score = context.confidence * avg_strength
    if score < SCORE_THRESHOLD:
        return Decision(action="reject", reason=f"score_below_threshold:{score:.2f}")

    # Step 5: Route — event triggers go to perp, but defensive mode blocks them
    if any(t.urgency == "high" for t in agreeing):
        if context.risk_mode == "defensive":
            return Decision(action="reject", reason="event_blocked_by_risk_mode")
        route = "perp"
    else:
        route = "spot"

    return Decision(action="trade", direction=best_dir, route=route, score=score, reason="ok")
```

**Step 4: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_decision_schemas.py -v
./venv/bin/python -m ruff check --fix decision.py
```
Expected: All tests pass, 0 ruff errors.

**Step 5: Commit**

```bash
git add decision.py tests/test_decision_schemas.py
git commit -m "feat: Phase 9 P1 — frozen ContextState, TriggerSignal, Decision schemas + evaluate()"
```

---

### Task 2: Add Decision Layer integration tests (evaluate() logic)

**Files:**
- Modify: `tests/test_decision_schemas.py` (add `TestEvaluate` class)

**Step 1: Write the failing tests**

Append to `tests/test_decision_schemas.py`:

```python
from decision import evaluate, SCORE_THRESHOLD


def make_two_triggers(direction="long", urgency="normal", strength=0.8):
    """Helper: two agreeing triggers for direction."""
    base = dict(
        trigger_id=str(uuid.uuid4()),
        source="momentum_1h",
        direction=direction,
        strength=strength,
        urgency=urgency,
        symbol_scope="BTC",
        reason="test",
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        raw_data={},
    )
    t1 = TriggerSignal(**{**base, "trigger_id": "t1", "source": "momentum_1h"})
    t2 = TriggerSignal(**{**base, "trigger_id": "t2", "source": "orderflow"})
    return [t1, t2]


class TestEvaluate:
    def test_rejects_when_not_tradeable(self):
        ctx = make_context(tradeable=False)
        d = evaluate(ctx, make_two_triggers())
        assert d.action == "reject"
        assert "not_tradeable" in d.reason

    def test_rejects_when_no_allowed_directions(self):
        ctx = make_context(allowed_directions=[])
        d = evaluate(ctx, make_two_triggers())
        assert d.action == "reject"
        assert "no_allowed_directions" in d.reason

    def test_rejects_when_only_one_trigger(self):
        ctx = make_context()
        d = evaluate(ctx, [make_two_triggers()[0]])
        assert d.action == "reject"
        assert "insufficient" in d.reason

    def test_rejects_when_triggers_disagree(self):
        ctx = make_context(allowed_directions=["long", "short"])
        long_t = make_two_triggers("long")[0]
        short_t = make_two_triggers("short")[0]
        d = evaluate(ctx, [long_t, short_t])
        assert d.action == "reject"

    def test_rejects_when_score_too_low(self):
        ctx = make_context(confidence=0.4)
        triggers = make_two_triggers(strength=0.6)  # score = 0.4 * 0.6 = 0.24 < 0.50
        d = evaluate(ctx, triggers)
        assert d.action == "reject"
        assert "score_below_threshold" in d.reason

    def test_trades_spot_on_normal_urgency(self):
        ctx = make_context(confidence=0.8)
        d = evaluate(ctx, make_two_triggers(urgency="normal", strength=0.8))
        assert d.action == "trade"
        assert d.route == "spot"
        assert d.direction == "long"

    def test_trades_perp_on_high_urgency(self):
        ctx = make_context(confidence=0.8)
        d = evaluate(ctx, make_two_triggers(urgency="high", strength=0.8))
        assert d.action == "trade"
        assert d.route == "perp"

    def test_event_blocked_in_defensive_mode(self):
        ctx = make_context(risk_mode="defensive", confidence=0.9)
        d = evaluate(ctx, make_two_triggers(urgency="high", strength=0.9))
        assert d.action == "reject"
        assert "defensive" in d.reason

    def test_expired_triggers_are_excluded(self):
        ctx = make_context(confidence=0.8)
        expired = make_two_triggers()
        for t in expired:
            object.__setattr__(t, "expires_at", datetime.now(UTC) - timedelta(minutes=1))
        # expired triggers are mutable dataclasses so we rebuild them
        from dataclasses import replace
        expired_triggers = [
            TriggerSignal(
                trigger_id=t.trigger_id, source=t.source, direction=t.direction,
                strength=t.strength, urgency=t.urgency, symbol_scope=t.symbol_scope,
                reason=t.reason, expires_at=datetime.now(UTC) - timedelta(minutes=1),
                raw_data={},
            )
            for t in expired
        ]
        d = evaluate(ctx, expired_triggers)
        assert d.action == "reject"
        assert "no_valid_triggers" in d.reason
```

Add `import uuid` to the top of the test file.

**Step 2: Run tests**

```bash
./venv/bin/python -m pytest tests/test_decision_schemas.py -v
```
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/test_decision_schemas.py
git commit -m "test: Phase 9 P1 — evaluate() integration tests (gate, consensus, score, routing)"
```

---

## Phase 2 — Data Layer

> Goal: Immutable timestamped snapshots. Multi-timeframe OHLCV (4h/1h/15m).
> No downstream component fetches raw data directly.

---

### Task 3: Create `data_snapshot.py`

**Files:**
- Create: `data_snapshot.py`
- Create: `tests/test_data_snapshot.py`

**Step 1: Write the failing test**

```python
# tests/test_data_snapshot.py
from datetime import UTC, datetime
import pandas as pd
from data_snapshot import DataSnapshot


class TestDataSnapshot:
    def _make_df(self, n=10):
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        return pd.DataFrame({
            "open": [100.0] * n, "high": [101.0] * n,
            "low": [99.0] * n, "close": [100.5] * n, "volume": [1000.0] * n,
        }, index=idx)

    def test_snapshot_stores_dataframes(self):
        df_1h = self._make_df(10)
        snap = DataSnapshot(df_1h=df_1h, df_4h=None, df_15m=None)
        assert len(snap.df_1h) == 10

    def test_snapshot_is_timestamped(self):
        snap = DataSnapshot(df_1h=self._make_df(), df_4h=None, df_15m=None)
        assert isinstance(snap.captured_at, datetime)
        assert snap.captured_at.tzinfo is not None  # must be tz-aware

    def test_snapshot_dataframes_are_readonly(self):
        df = self._make_df()
        snap = DataSnapshot(df_1h=df, df_4h=None, df_15m=None)
        import pytest
        with pytest.raises(ValueError):
            snap.df_1h.iloc[0, 0] = 999.0  # should raise on read-only df

    def test_snapshot_has_symbol(self):
        snap = DataSnapshot(df_1h=self._make_df(), df_4h=None, df_15m=None, symbol="BTC/USDT")
        assert snap.symbol == "BTC/USDT"
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_data_snapshot.py -v
```

**Step 3: Implement `data_snapshot.py`**

```python
"""Data snapshot — immutable timestamped OHLCV data container.

All market data enters the system through DataSnapshot. Downstream
components read from snapshots only — they never fetch raw data directly.
This ensures consistent state within a processing cycle and makes
debugging deterministic (the snapshot is the ground truth for that cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

import pandas as pd


@dataclass
class DataSnapshot:
    """Immutable multi-timeframe OHLCV snapshot.

    DataFrames are frozen (read-only) at construction time to prevent
    accidental mutation downstream. None values indicate that timeframe
    data was not available for this snapshot.

    Attributes:
        df_1h: 1-hour OHLCV DataFrame (primary timeframe for triggers).
        df_4h: 4-hour OHLCV DataFrame (used by SwingAnalyzer).
        df_15m: 15-minute OHLCV DataFrame (used for fine-grained momentum).
        symbol: Trading pair, e.g. "BTC/USDT".
        captured_at: UTC datetime when this snapshot was taken.
    """

    df_1h: Optional[pd.DataFrame]
    df_4h: Optional[pd.DataFrame]
    df_15m: Optional[pd.DataFrame]
    symbol: str = "BTC/USDT"
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        # Freeze all DataFrames to prevent downstream mutation
        for attr in ("df_1h", "df_4h", "df_15m"):
            df = getattr(self, attr)
            if df is not None:
                df.flags.writeable = False
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_data_snapshot.py -v
./venv/bin/python -m ruff check --fix data_snapshot.py
```
Expected: All tests pass.

**Step 5: Commit**

```bash
git add data_snapshot.py tests/test_data_snapshot.py
git commit -m "feat: Phase 9 P2 — DataSnapshot (immutable timestamped multi-timeframe container)"
```

---

### Task 4: Create `multi_timeframe_fetcher.py`

**Files:**
- Create: `multi_timeframe_fetcher.py`
- Create: `tests/test_multi_timeframe_fetcher.py`

**Step 1: Write the failing test**

```python
# tests/test_multi_timeframe_fetcher.py
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from multi_timeframe_fetcher import MultiTimeframeFetcher


class TestMultiTimeframeFetcher:
    def _make_mock_exchange(self):
        exchange = MagicMock()
        ohlcv_data = [[1704067200000, 100.0, 101.0, 99.0, 100.5, 1000.0]] * 50
        exchange.fetch_ohlcv.return_value = ohlcv_data
        return exchange

    def test_fetch_returns_snapshot(self):
        from data_snapshot import DataSnapshot
        fetcher = MultiTimeframeFetcher(exchange=self._make_mock_exchange())
        snap = fetcher.fetch("BTC/USDT")
        assert isinstance(snap, DataSnapshot)
        assert snap.symbol == "BTC/USDT"

    def test_fetch_populates_all_timeframes(self):
        fetcher = MultiTimeframeFetcher(exchange=self._make_mock_exchange())
        snap = fetcher.fetch("BTC/USDT")
        assert snap.df_1h is not None
        assert snap.df_4h is not None
        assert snap.df_15m is not None

    def test_fetch_calls_exchange_for_each_timeframe(self):
        exchange = self._make_mock_exchange()
        fetcher = MultiTimeframeFetcher(exchange=exchange)
        fetcher.fetch("BTC/USDT")
        # Should call fetch_ohlcv 3 times (1h, 4h, 15m)
        assert exchange.fetch_ohlcv.call_count == 3

    def test_partial_failure_returns_available_timeframes(self):
        exchange = self._make_mock_exchange()
        ohlcv = [[1704067200000, 100.0, 101.0, 99.0, 100.5, 1000.0]] * 50
        # 4h fails
        def side_effect(symbol, timeframe, *args, **kwargs):
            if timeframe == "4h":
                raise Exception("rate limit")
            return ohlcv
        exchange.fetch_ohlcv.side_effect = side_effect
        fetcher = MultiTimeframeFetcher(exchange=exchange)
        snap = fetcher.fetch("BTC/USDT")
        assert snap.df_1h is not None
        assert snap.df_4h is None  # graceful degradation
        assert snap.df_15m is not None
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_multi_timeframe_fetcher.py -v
```

**Step 3: Implement `multi_timeframe_fetcher.py`**

```python
"""Multi-timeframe OHLCV fetcher.

Fetches 4h, 1h, and 15m bars from the exchange and packages them
into an immutable DataSnapshot. Each timeframe failure is isolated —
a rate-limit on 4h does not prevent 1h and 15m from being fetched.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from data_snapshot import DataSnapshot

_log = logging.getLogger(__name__)

# Number of bars to fetch per timeframe
_BARS: dict[str, int] = {
    "4h": 200,   # ~33 days for swing context
    "1h": 200,   # ~8 days for trigger analysis
    "15m": 100,  # ~25 hours for fine momentum
}

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class MultiTimeframeFetcher:
    """Fetches 4h/1h/15m OHLCV data and returns an immutable DataSnapshot.

    Args:
        exchange: A CCXT exchange instance with fetch_ohlcv support.
    """

    def __init__(self, exchange: Any) -> None:
        self.exchange = exchange

    def fetch(self, symbol: str) -> DataSnapshot:
        """Fetch all timeframes for a symbol and return a DataSnapshot.

        Failed timeframes are returned as None (graceful degradation).

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".

        Returns:
            DataSnapshot with available DataFrames and UTC timestamp.
        """
        frames: dict[str, Optional[pd.DataFrame]] = {}
        for tf, n_bars in _BARS.items():
            try:
                raw = self.exchange.fetch_ohlcv(symbol, tf, limit=n_bars)
                df = pd.DataFrame(raw, columns=_COLUMNS)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                frames[tf] = df
            except Exception as e:
                _log.warning("Failed to fetch %s %s: %s", symbol, tf, e)
                frames[tf] = None

        return DataSnapshot(
            df_1h=frames.get("1h"),
            df_4h=frames.get("4h"),
            df_15m=frames.get("15m"),
            symbol=symbol,
        )
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_multi_timeframe_fetcher.py -v
./venv/bin/python -m ruff check --fix multi_timeframe_fetcher.py
```

**Step 5: Commit**

```bash
git add multi_timeframe_fetcher.py tests/test_multi_timeframe_fetcher.py
git commit -m "feat: Phase 9 P2 — MultiTimeframeFetcher (4h/1h/15m → DataSnapshot)"
```

---

## Phase 3 — Context Engine

> Goal: Build the 4 analyzers and ContextEngine. Output: versioned ContextState every 15min.
> The intelligence/ providers are reused for raw data — don't duplicate their fetch logic.

---

### Task 5: Create `context/` package with `SwingAnalyzer`

**Files:**
- Create: `context/__init__.py`
- Create: `context/swing.py`
- Create: `tests/test_context_analyzers.py`

**Step 1: Write the failing test**

```python
# tests/test_context_analyzers.py
import pandas as pd
import numpy as np
import pytest
from context.swing import SwingAnalyzer


def make_trending_up_df(n=100):
    """Steadily rising prices — should produce bullish bias."""
    closes = np.linspace(90000, 100000, n)
    df = pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": [1000.0] * n,
    })
    return df


def make_trending_down_df(n=100):
    """Steadily falling prices — should produce bearish bias."""
    closes = np.linspace(100000, 90000, n)
    df = pd.DataFrame({
        "open": closes * 1.001,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": [1000.0] * n,
    })
    return df


def make_sideways_df(n=100):
    """Oscillating prices — should produce neutral bias."""
    import math
    closes = [95000 + 1000 * math.sin(i * 0.3) for i in range(n)]
    df = pd.DataFrame({
        "open": [c * 0.999 for c in closes],
        "high": [c * 1.002 for c in closes],
        "low": [c * 0.998 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })
    return df


class TestSwingAnalyzer:
    def test_bullish_bias_on_uptrend(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_up_df())
        assert result["swing_bias"] == "bullish"

    def test_bearish_bias_on_downtrend(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_down_df())
        assert result["swing_bias"] == "bearish"

    def test_neutral_on_sideways(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_sideways_df())
        assert result["swing_bias"] == "neutral"

    def test_returns_key_levels(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_up_df())
        assert "support" in result["key_levels"]
        assert "resistance" in result["key_levels"]
        assert result["key_levels"]["support"] > 0

    def test_returns_allowed_directions(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_up_df())
        assert "long" in result["allowed_directions"]

    def test_insufficient_data_returns_neutral(self):
        analyzer = SwingAnalyzer()
        df = make_trending_up_df(n=10)  # too few bars
        result = analyzer.analyze(df)
        assert result["swing_bias"] == "neutral"
        assert result["allowed_directions"] == []
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_context_analyzers.py -v
```

**Step 3: Implement `context/__init__.py` and `context/swing.py`**

```python
# context/__init__.py
"""Context analyzers — produce components of ContextState."""
```

```python
# context/swing.py
"""SwingAnalyzer — 4h price structure analysis.

Uses EMA alignment (21/50/200) and recent swing high/low structure
to determine directional bias. This is the primary gate for spot trades.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_MIN_BARS = 50  # Need at least 50 bars for EMA-200 to be meaningful


class SwingAnalyzer:
    """Analyses 4h price structure to determine swing bias and key levels.

    Uses EMA 21/50/200 alignment as the primary bias signal. Falls back
    to neutral with no allowed directions when data is insufficient.
    """

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyse price structure and return swing context components.

        Args:
            df: 4h OHLCV DataFrame with at least 50 rows. Columns: open, high, low, close, volume.

        Returns:
            Dict with keys: swing_bias, allowed_directions, key_levels, confidence.
        """
        if len(df) < _MIN_BARS:
            _log.debug("Insufficient data for swing analysis (%d bars)", len(df))
            return {
                "swing_bias": "neutral",
                "allowed_directions": [],
                "key_levels": {"support": 0.0, "resistance": 0.0, "poc": 0.0},
                "confidence": 0.0,
            }

        close = df["close"]

        # EMA alignment
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else None

        latest = close.iloc[-1]
        e21 = ema21.iloc[-1]
        e50 = ema50.iloc[-1]
        e200 = ema200.iloc[-1] if ema200 is not None else None

        # Count bullish/bearish EMA conditions
        bullish_count = sum([
            latest > e21,
            e21 > e50,
            e200 is not None and e50 > e200,
        ])
        bearish_count = sum([
            latest < e21,
            e21 < e50,
            e200 is not None and e50 < e200,
        ])

        if bullish_count >= 2 and bullish_count > bearish_count:
            bias = "bullish"
            allowed = ["long"]
            confidence = 0.5 + (bullish_count / 3) * 0.4
        elif bearish_count >= 2 and bearish_count > bullish_count:
            bias = "bearish"
            allowed = ["short"]
            confidence = 0.5 + (bearish_count / 3) * 0.4
        else:
            bias = "neutral"
            allowed = []
            confidence = 0.3

        # Key levels: recent 20-bar swing high/low as resistance/support
        window = min(20, len(df))
        recent = df.tail(window)
        support = float(recent["low"].min())
        resistance = float(recent["high"].max())
        poc = float(recent["close"].median())

        return {
            "swing_bias": bias,
            "allowed_directions": allowed,
            "key_levels": {"support": support, "resistance": resistance, "poc": poc},
            "confidence": round(confidence, 3),
        }
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_context_analyzers.py::TestSwingAnalyzer -v
./venv/bin/python -m ruff check --fix context/swing.py
```

**Step 5: Commit**

```bash
git add context/__init__.py context/swing.py tests/test_context_analyzers.py
git commit -m "feat: Phase 9 P3 — SwingAnalyzer (4h EMA alignment → swing_bias + key_levels)"
```

---

### Task 6: Add `FundingAnalyzer`, `WhaleFlowAnalyzer`, `OITrendAnalyzer`

**Files:**
- Create: `context/funding.py`
- Create: `context/whale_flow.py`
- Create: `context/oi_trend.py`
- Modify: `tests/test_context_analyzers.py` (add test classes)

**Step 1: Write the failing tests**

Append to `tests/test_context_analyzers.py`:

```python
from context.funding import FundingAnalyzer
from context.whale_flow import WhaleFlowAnalyzer
from context.oi_trend import OITrendAnalyzer


class TestFundingAnalyzer:
    def test_positive_funding_is_long_crowded(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=0.0005)  # 0.05%
        assert "long_crowded" in result["funding_pressure"]

    def test_negative_funding_is_short_crowded(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=-0.0006)
        assert "short_crowded" in result["funding_pressure"]

    def test_near_zero_is_neutral(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=0.0001)
        assert result["funding_pressure"] == "neutral"

    def test_extreme_positive_funding(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=0.0012)  # > 0.10%
        assert result["funding_pressure"] == "long_crowded_extreme"

    def test_none_funding_returns_neutral(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=None)
        assert result["funding_pressure"] == "neutral"


class TestWhaleFlowAnalyzer:
    def test_positive_net_flow_is_accumulating(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=500_000.0)
        assert result["whale_flow"] == "accumulating"

    def test_negative_net_flow_is_distributing(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=-500_000.0)
        assert result["whale_flow"] == "distributing"

    def test_small_flow_is_neutral(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=1000.0)
        assert result["whale_flow"] == "neutral"

    def test_none_flow_is_neutral(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=None)
        assert result["whale_flow"] == "neutral"


class TestOITrendAnalyzer:
    def test_oi_up_price_up_is_expanding_up(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=5.0, price_change_pct=3.0)
        assert result["oi_trend"] == "expanding_up"

    def test_oi_down_price_down_is_expanding_down(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=-5.0, price_change_pct=-3.0)
        assert result["oi_trend"] == "expanding_down"

    def test_oi_down_price_up_is_contracting(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=-3.0, price_change_pct=2.0)
        assert result["oi_trend"] == "contracting"

    def test_small_changes_is_neutral(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=0.5, price_change_pct=0.3)
        assert result["oi_trend"] == "neutral"
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_context_analyzers.py -v -k "Funding or Whale or OI"
```

**Step 3: Implement the three analyzers**

```python
# context/funding.py
"""FundingAnalyzer — classify perpetual funding rate as market pressure signal."""

from __future__ import annotations

from typing import Any, Optional


# Thresholds (annualised: rate × 3 × 365 = APR)
_MILD_THRESHOLD = 0.0003      # 0.03% per 8h = crowded (mild)
_EXTREME_THRESHOLD = 0.0010   # 0.10% per 8h = crowded (extreme)
_SHORT_MILD = -0.0002
_SHORT_EXTREME = -0.0005


class FundingAnalyzer:
    """Classifies current funding rate into a FundingPressure Literal."""

    def analyze(self, funding_rate: Optional[float]) -> dict[str, Any]:
        """Classify funding rate.

        Args:
            funding_rate: Current 8h funding rate as a decimal (e.g. 0.0005 = 0.05%).
                None returns neutral.

        Returns:
            Dict with key ``funding_pressure``.
        """
        if funding_rate is None:
            return {"funding_pressure": "neutral"}

        if funding_rate >= _EXTREME_THRESHOLD:
            return {"funding_pressure": "long_crowded_extreme"}
        if funding_rate >= _MILD_THRESHOLD:
            return {"funding_pressure": "long_crowded_mild"}
        if funding_rate <= _SHORT_EXTREME:
            return {"funding_pressure": "short_crowded_extreme"}
        if funding_rate <= _SHORT_MILD:
            return {"funding_pressure": "short_crowded_mild"}
        return {"funding_pressure": "neutral"}
```

```python
# context/whale_flow.py
"""WhaleFlowAnalyzer — classify net whale exchange flow direction."""

from __future__ import annotations

from typing import Any, Optional

_THRESHOLD = 100_000.0  # USD notional — below this is noise


class WhaleFlowAnalyzer:
    """Classifies net whale flow into accumulating / distributing / neutral."""

    def analyze(self, net_flow: Optional[float]) -> dict[str, Any]:
        """Classify whale flow.

        Args:
            net_flow: Net USD flow (positive = outflows from exchanges = accumulation).
                None returns neutral.

        Returns:
            Dict with key ``whale_flow``.
        """
        if net_flow is None or abs(net_flow) < _THRESHOLD:
            return {"whale_flow": "neutral"}
        return {"whale_flow": "accumulating" if net_flow > 0 else "distributing"}
```

```python
# context/oi_trend.py
"""OITrendAnalyzer — classify open interest trend vs price direction."""

from __future__ import annotations

from typing import Any, Optional

_MIN_CHANGE = 1.0  # % — below this is noise


class OITrendAnalyzer:
    """Classifies OI trend relative to price direction."""

    def analyze(self, oi_change_pct: Optional[float], price_change_pct: Optional[float]) -> dict[str, Any]:
        """Classify OI vs price trend.

        Args:
            oi_change_pct: % change in open interest over lookback period.
            price_change_pct: % change in price over same period.

        Returns:
            Dict with key ``oi_trend``.
        """
        if oi_change_pct is None or price_change_pct is None:
            return {"oi_trend": "neutral"}
        if abs(oi_change_pct) < _MIN_CHANGE and abs(price_change_pct) < _MIN_CHANGE:
            return {"oi_trend": "neutral"}

        oi_up = oi_change_pct > _MIN_CHANGE
        oi_down = oi_change_pct < -_MIN_CHANGE
        price_up = price_change_pct > 0

        if oi_up and price_up:
            return {"oi_trend": "expanding_up"}
        if oi_down and not price_up:
            return {"oi_trend": "expanding_down"}
        if oi_down and price_up:
            return {"oi_trend": "contracting"}
        return {"oi_trend": "neutral"}
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_context_analyzers.py -v
./venv/bin/python -m ruff check --fix context/funding.py context/whale_flow.py context/oi_trend.py
```

**Step 5: Commit**

```bash
git add context/funding.py context/whale_flow.py context/oi_trend.py tests/test_context_analyzers.py
git commit -m "feat: Phase 9 P3 — FundingAnalyzer, WhaleFlowAnalyzer, OITrendAnalyzer"
```

---

### Task 7: Create `context_engine.py`

**Files:**
- Create: `context_engine.py`
- Create: `tests/test_context_engine.py`

**Step 1: Write the failing test**

```python
# tests/test_context_engine.py
from unittest.mock import MagicMock, patch
from datetime import UTC, datetime
import pandas as pd
import numpy as np
import pytest
from context_engine import ContextEngine
from data_snapshot import DataSnapshot
from decision import ContextState


def make_snapshot(bullish=True):
    n = 100
    closes = np.linspace(90000, 100000, n) if bullish else np.linspace(100000, 90000, n)
    df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes, "volume": [1000.0] * n,
    })
    return DataSnapshot(df_1h=df, df_4h=df, df_15m=None, symbol="BTC/USDT")


class TestContextEngine:
    def test_returns_context_state(self):
        engine = ContextEngine()
        snap = make_snapshot(bullish=True)
        ctx = engine.build(snap, funding_rate=0.0001, net_whale_flow=None,
                           oi_change_pct=None, price_change_pct=None)
        assert isinstance(ctx, ContextState)

    def test_context_id_is_set(self):
        engine = ContextEngine()
        ctx = engine.build(make_snapshot(), funding_rate=None,
                           net_whale_flow=None, oi_change_pct=None, price_change_pct=None)
        assert ctx.context_id != ""

    def test_valid_until_is_future(self):
        engine = ContextEngine()
        ctx = engine.build(make_snapshot(), funding_rate=None,
                           net_whale_flow=None, oi_change_pct=None, price_change_pct=None)
        assert ctx.valid_until > datetime.now(UTC)

    def test_tradeable_false_when_no_allowed_directions(self):
        engine = ContextEngine()
        # Sideways data → neutral → no allowed directions → not tradeable
        import math
        closes = [95000 + 500 * math.sin(i * 0.3) for i in range(100)]
        df = pd.DataFrame({
            "open": [c * 0.999 for c in closes], "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes], "close": closes, "volume": [1000.0] * 100,
        })
        snap = DataSnapshot(df_1h=df, df_4h=df, df_15m=None)
        ctx = engine.build(snap, funding_rate=None, net_whale_flow=None,
                           oi_change_pct=None, price_change_pct=None)
        # neutral → allowed_directions=[] → tradeable=False
        if ctx.swing_bias == "neutral":
            assert ctx.tradeable is False

    def test_risk_mode_defaults_to_normal(self):
        engine = ContextEngine()
        ctx = engine.build(make_snapshot(), funding_rate=None,
                           net_whale_flow=None, oi_change_pct=None, price_change_pct=None)
        assert ctx.risk_mode == "normal"

    def test_risk_supervisor_can_override_risk_mode(self):
        engine = ContextEngine()
        engine.set_risk_mode("defensive")
        ctx = engine.build(make_snapshot(), funding_rate=None,
                           net_whale_flow=None, oi_change_pct=None, price_change_pct=None)
        assert ctx.risk_mode == "defensive"
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_context_engine.py -v
```

**Step 3: Implement `context_engine.py`**

```python
"""ContextEngine — produces ContextState every 15 minutes.

Orchestrates the four context analyzers and combines their outputs
into a single versioned ContextState. The RiskSupervisor can push
risk_mode changes via set_risk_mode().
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Optional

from context.funding import FundingAnalyzer
from context.oi_trend import OITrendAnalyzer
from context.swing import SwingAnalyzer
from context.whale_flow import WhaleFlowAnalyzer
from data_snapshot import DataSnapshot
from decision import ContextState

_log = logging.getLogger(__name__)

_CONTEXT_WINDOW_MINUTES = 15


class ContextEngine:
    """Builds ContextState from a DataSnapshot and real-time market data.

    Call build() every 15 minutes (or whenever a new DataSnapshot is available).
    The RiskSupervisor calls set_risk_mode() to inject defensive/cautious states.
    """

    def __init__(self) -> None:
        self._swing = SwingAnalyzer()
        self._funding = FundingAnalyzer()
        self._whale = WhaleFlowAnalyzer()
        self._oi = OITrendAnalyzer()
        self._risk_mode: str = "normal"

    def set_risk_mode(self, mode: str) -> None:
        """Called by RiskSupervisor to override the risk_mode field.

        Args:
            mode: One of "normal", "cautious", "defensive".
        """
        self._risk_mode = mode

    def build(
        self,
        snapshot: DataSnapshot,
        funding_rate: Optional[float],
        net_whale_flow: Optional[float],
        oi_change_pct: Optional[float],
        price_change_pct: Optional[float],
    ) -> ContextState:
        """Build a ContextState from current market data.

        Args:
            snapshot: Multi-timeframe DataSnapshot (4h used for swing analysis).
            funding_rate: Current 8h perpetual funding rate (decimal).
            net_whale_flow: Net whale USD flow (positive = accumulation).
            oi_change_pct: % change in open interest over last 4h.
            price_change_pct: % change in price over last 4h.

        Returns:
            A new ContextState valid for the next 15 minutes.
        """
        now = datetime.now(UTC)
        context_id = now.strftime("%Y-%m-%dT%H:%MZ")

        # Use 4h data for swing; fall back to 1h if 4h unavailable
        df_swing = snapshot.df_4h if snapshot.df_4h is not None else snapshot.df_1h

        swing = self._swing.analyze(df_swing) if df_swing is not None else {
            "swing_bias": "neutral", "allowed_directions": [],
            "key_levels": {"support": 0.0, "resistance": 0.0, "poc": 0.0},
            "confidence": 0.0,
        }
        funding = self._funding.analyze(funding_rate)
        whale = self._whale.analyze(net_whale_flow)
        oi = self._oi.analyze(oi_change_pct, price_change_pct)

        allowed_directions = swing["allowed_directions"]
        tradeable = len(allowed_directions) > 0

        # Overall confidence = swing confidence (primary signal)
        confidence = swing["confidence"]

        return ContextState(
            context_id=context_id,
            swing_bias=swing["swing_bias"],
            allowed_directions=allowed_directions,
            volatility_regime="normal",  # TODO Phase 3 extension: add VolatilityAnalyzer
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

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_context_engine.py -v
./venv/bin/python -m ruff check --fix context_engine.py
```

**Step 5: Commit**

```bash
git add context_engine.py tests/test_context_engine.py
git commit -m "feat: Phase 9 P3 — ContextEngine (orchestrates 4 analyzers → ContextState)"
```

---

## Phase 4 — Trigger Engine + Decision Layer

> Goal: Build spot triggers and wire evaluate(). Start logging every decision.
> No live trades yet — logging only.

---

### Task 8: Create `triggers/` package with `MomentumTrigger`

**Files:**
- Create: `triggers/__init__.py`
- Create: `triggers/momentum.py`
- Create: `tests/test_triggers.py`

**Step 1: Write the failing tests**

```python
# tests/test_triggers.py
from datetime import UTC, datetime, timedelta
import pandas as pd
import numpy as np
import pytest
from triggers.momentum import MomentumTrigger


def make_bullish_1h_df(n=50):
    """Rising prices with RSI > 50, strong volume."""
    closes = np.linspace(90000, 95000, n)
    volumes = [2000.0 if i > n - 5 else 1000.0 for i in range(n)]  # volume spike at end
    df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.003,
        "low": closes * 0.997, "close": closes, "volume": volumes,
    })
    return df


def make_bearish_1h_df(n=50):
    closes = np.linspace(95000, 90000, n)
    volumes = [2000.0 if i > n - 5 else 1000.0 for i in range(n)]
    df = pd.DataFrame({
        "open": closes * 1.001, "high": closes * 1.003,
        "low": closes * 0.997, "close": closes, "volume": volumes,
    })
    return df


def make_flat_df(n=50):
    closes = [95000.0] * n
    df = pd.DataFrame({
        "open": closes, "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes], "close": closes,
        "volume": [1000.0] * n,
    })
    return df


class TestMomentumTrigger:
    def test_returns_trigger_signal_list(self):
        from decision import TriggerSignal
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        assert isinstance(signals, list)

    def test_bullish_momentum_produces_long_signal(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        directions = [s.direction for s in signals]
        assert "long" in directions

    def test_bearish_momentum_produces_short_signal(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bearish_1h_df())
        directions = [s.direction for s in signals]
        assert "short" in directions

    def test_flat_market_produces_no_signals(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_flat_df())
        assert signals == []

    def test_signals_have_correct_symbol_scope(self):
        trigger = MomentumTrigger(symbol="ETH/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        for s in signals:
            assert s.symbol_scope == "ETH"

    def test_signals_have_future_expiry(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        for s in signals:
            assert s.expires_at > datetime.now(UTC)

    def test_insufficient_data_returns_empty(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df(n=5))
        assert signals == []
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_triggers.py -v
```

**Step 3: Implement `triggers/__init__.py` and `triggers/momentum.py`**

```python
# triggers/__init__.py
"""Trigger implementations — each produces TriggerSignal[] from market data."""
```

```python
# triggers/momentum.py
"""MomentumTrigger — 1h RSI + MACD zero-cross + volume confirmation.

Fires when:
  - RSI crosses the 50 line (above for long, below for short)
  - MACD crosses the zero line (bullish for long, bearish for short)
  - Volume confirms (> 1.5× 20-bar average)

At least 2 of these 3 conditions must be true to fire.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_MIN_BARS = 26   # Need 26 bars for MACD
_SIGNAL_TTL_MINUTES = 75   # Trigger expires after ~1 candle + buffer
_VOL_MULT = 1.5


class MomentumTrigger:
    """Generates momentum-based TriggerSignals from 1h OHLCV data.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Used to set symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]  # "BTC/USDT" → "BTC"

    def evaluate(self, df: pd.DataFrame) -> list[TriggerSignal]:
        """Evaluate 1h OHLCV data and return any momentum triggers.

        Args:
            df: 1h OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            List of TriggerSignal (empty if no momentum conditions met).
        """
        if len(df) < _MIN_BARS:
            return []

        close = df["close"]
        volume = df["volume"]

        # RSI (14-period)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD (12/26 EMA, signal line 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26

        # Volume ratio vs 20-bar average
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)

        if rsi.isna().iloc[-1] or macd_line.isna().iloc[-1]:
            return []

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_macd = macd_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        current_vol_ratio = float(vol_ratio.iloc[-1]) if not vol_ratio.isna().iloc[-1] else 1.0

        # Condition checks
        rsi_crossed_up = prev_rsi < 50 <= current_rsi
        rsi_crossed_down = prev_rsi > 50 >= current_rsi
        macd_crossed_up = prev_macd < 0 <= current_macd
        macd_crossed_down = prev_macd > 0 >= current_macd
        vol_confirmed = current_vol_ratio >= _VOL_MULT

        long_score = sum([rsi_crossed_up, macd_crossed_up, vol_confirmed and current_rsi > 50])
        short_score = sum([rsi_crossed_down, macd_crossed_down, vol_confirmed and current_rsi < 50])

        signals = []
        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)

        if long_score >= 2:
            strength = min(0.4 + long_score * 0.2, 0.95)
            reason_parts = []
            if rsi_crossed_up: reason_parts.append(f"RSI crossed 50↑ ({current_rsi:.1f})")
            if macd_crossed_up: reason_parts.append("MACD zero-cross↑")
            if vol_confirmed: reason_parts.append(f"vol {current_vol_ratio:.1f}×")
            signals.append(TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="momentum_1h",
                direction="long",
                strength=strength,
                urgency="normal",
                symbol_scope=self._symbol_scope,
                reason=", ".join(reason_parts),
                expires_at=expiry,
                raw_data={"rsi": round(current_rsi, 2), "macd": round(float(current_macd), 4),
                          "vol_ratio": round(current_vol_ratio, 2)},
            ))

        if short_score >= 2:
            strength = min(0.4 + short_score * 0.2, 0.95)
            reason_parts = []
            if rsi_crossed_down: reason_parts.append(f"RSI crossed 50↓ ({current_rsi:.1f})")
            if macd_crossed_down: reason_parts.append("MACD zero-cross↓")
            if vol_confirmed: reason_parts.append(f"vol {current_vol_ratio:.1f}×")
            signals.append(TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="momentum_1h",
                direction="short",
                strength=strength,
                urgency="normal",
                symbol_scope=self._symbol_scope,
                reason=", ".join(reason_parts),
                expires_at=expiry,
                raw_data={"rsi": round(current_rsi, 2), "macd": round(float(current_macd), 4),
                          "vol_ratio": round(current_vol_ratio, 2)},
            ))

        return signals
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_triggers.py -v
./venv/bin/python -m ruff check --fix triggers/momentum.py
```

**Step 5: Commit**

```bash
git add triggers/__init__.py triggers/momentum.py tests/test_triggers.py
git commit -m "feat: Phase 9 P4 — MomentumTrigger (RSI zero-cross + MACD + volume → TriggerSignal)"
```

---

### Task 9: Create `triggers/orderflow.py` and `trigger_engine.py`

**Files:**
- Create: `triggers/orderflow.py`
- Create: `trigger_engine.py`
- Create: `tests/test_trigger_engine.py`

**Step 1: Write the failing tests**

```python
# tests/test_trigger_engine.py
from unittest.mock import MagicMock
from datetime import UTC, datetime, timedelta
import pandas as pd
import numpy as np
import pytest
from trigger_engine import TriggerEngine
from data_snapshot import DataSnapshot
from decision import TriggerSignal


def make_bullish_snapshot():
    n = 50
    closes = np.linspace(90000, 95000, n)
    volumes = [2000.0 if i > n - 5 else 1000.0 for i in range(n)]
    df = pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.003,
        "low": closes * 0.997, "close": closes, "volume": volumes,
    })
    return DataSnapshot(df_1h=df, df_4h=df, df_15m=None, symbol="BTC/USDT")


class TestTriggerEngine:
    def test_collect_returns_list_of_trigger_signals(self):
        engine = TriggerEngine()
        signals = engine.collect(make_bullish_snapshot(), orderbook_data=None)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, TriggerSignal)

    def test_collect_returns_empty_on_flat_market(self):
        closes = [95000.0] * 50
        df = pd.DataFrame({
            "open": closes, "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes], "close": closes,
            "volume": [1000.0] * 50,
        })
        snap = DataSnapshot(df_1h=df, df_4h=df, df_15m=None)
        engine = TriggerEngine()
        signals = engine.collect(snap, orderbook_data=None)
        assert signals == []

    def test_collect_with_none_snapshot_returns_empty(self):
        engine = TriggerEngine()
        snap = DataSnapshot(df_1h=None, df_4h=None, df_15m=None)
        signals = engine.collect(snap, orderbook_data=None)
        assert signals == []
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_trigger_engine.py -v
```

**Step 3: Implement `triggers/orderflow.py` and `trigger_engine.py`**

```python
# triggers/orderflow.py
"""OrderFlowTrigger — CVD divergence and bid/ask imbalance signals.

Reuses raw CVD data from intelligence/orderbook.py when available.
Falls back gracefully when order book data is unavailable.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_SIGNAL_TTL_MINUTES = 30
_IMBALANCE_THRESHOLD = 0.65  # bid/(bid+ask) > 0.65 = bullish pressure


class OrderFlowTrigger:
    """Generates order-flow based TriggerSignals from order book data.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(self, orderbook_data: Optional[dict[str, Any]]) -> list[TriggerSignal]:
        """Evaluate order book snapshot and return flow triggers.

        Args:
            orderbook_data: Dict from intelligence/orderbook.py get_signal().
                Expected keys: cvd_divergence, imbalances. None = skip gracefully.

        Returns:
            List of TriggerSignal (empty if no order flow signal).
        """
        if orderbook_data is None:
            return []

        signals = []
        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)

        cvd_divergence = orderbook_data.get("cvd_divergence", "none")
        imbalances = orderbook_data.get("imbalances", {})
        top20 = imbalances.get("top_20", {}).get("imbalance", 0.5)

        # CVD divergence signals
        if cvd_divergence == "bullish_divergence":
            signals.append(TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="orderflow",
                direction="long",
                strength=0.65,
                urgency="normal",
                symbol_scope=self._symbol_scope,
                reason="CVD bullish divergence (price down, buy pressure up)",
                expires_at=expiry,
                raw_data={"cvd_divergence": cvd_divergence, "top20_imbalance": top20},
            ))
        elif cvd_divergence == "bearish_divergence":
            signals.append(TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="orderflow",
                direction="short",
                strength=0.65,
                urgency="normal",
                symbol_scope=self._symbol_scope,
                reason="CVD bearish divergence (price up, sell pressure up)",
                expires_at=expiry,
                raw_data={"cvd_divergence": cvd_divergence, "top20_imbalance": top20},
            ))

        # Bid/ask imbalance signals (supplementary)
        if top20 > _IMBALANCE_THRESHOLD:
            signals.append(TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="orderflow",
                direction="long",
                strength=min(0.3 + (top20 - 0.5) * 1.2, 0.75),
                urgency="normal",
                symbol_scope=self._symbol_scope,
                reason=f"Bid-side imbalance {top20:.2f} > {_IMBALANCE_THRESHOLD}",
                expires_at=expiry,
                raw_data={"top20_imbalance": top20},
            ))
        elif top20 < (1 - _IMBALANCE_THRESHOLD):
            signals.append(TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="orderflow",
                direction="short",
                strength=min(0.3 + (0.5 - top20) * 1.2, 0.75),
                urgency="normal",
                symbol_scope=self._symbol_scope,
                reason=f"Ask-side imbalance {top20:.2f} < {1 - _IMBALANCE_THRESHOLD}",
                expires_at=expiry,
                raw_data={"top20_imbalance": top20},
            ))

        return signals
```

```python
# trigger_engine.py
"""TriggerEngine — orchestrates all triggers and collects TriggerSignal[].

Runs per-candle (called from agent main loop on each 1h close) and
accumulates event-driven signals from real-time feeds.

Triggers are collected from:
  - MomentumTrigger (1h RSI/MACD/volume)
  - OrderFlowTrigger (CVD + bid/ask imbalance)
  - LiquidationTrigger (Phase 6 — perp event)
  - FundingExtremeTrigger (Phase 6 — perp event)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from data_snapshot import DataSnapshot
from decision import TriggerSignal
from triggers.momentum import MomentumTrigger
from triggers.orderflow import OrderFlowTrigger

_log = logging.getLogger(__name__)


class TriggerEngine:
    """Collects TriggerSignals from all registered triggers.

    Args:
        symbol: Trading pair used to initialise per-symbol triggers.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._momentum = MomentumTrigger(symbol=symbol)
        self._orderflow = OrderFlowTrigger(symbol=symbol)
        # Phase 6: LiquidationTrigger and FundingExtremeTrigger added here

    def collect(
        self,
        snapshot: DataSnapshot,
        orderbook_data: Optional[dict[str, Any]] = None,
    ) -> list[TriggerSignal]:
        """Collect all active TriggerSignals for this candle/event cycle.

        Args:
            snapshot: Current DataSnapshot. df_1h is used for momentum triggers.
            orderbook_data: Optional dict from intelligence/orderbook.py. None = skip.

        Returns:
            List of all active (non-expired) TriggerSignals from all triggers.
        """
        if snapshot.df_1h is None:
            return []

        signals: list[TriggerSignal] = []

        try:
            signals.extend(self._momentum.evaluate(snapshot.df_1h))
        except Exception as e:
            _log.warning("MomentumTrigger failed: %s", e)

        try:
            signals.extend(self._orderflow.evaluate(orderbook_data))
        except Exception as e:
            _log.warning("OrderFlowTrigger failed: %s", e)

        return signals
```

**Step 4: Run all tests**

```bash
./venv/bin/python -m pytest tests/test_trigger_engine.py tests/test_triggers.py -v
./venv/bin/python -m ruff check --fix triggers/orderflow.py trigger_engine.py
```

**Step 5: Commit**

```bash
git add triggers/orderflow.py trigger_engine.py tests/test_trigger_engine.py
git commit -m "feat: Phase 9 P4 — OrderFlowTrigger + TriggerEngine (collects all spot triggers)"
```

---

### Task 10: Create `risk_supervisor.py` + decision rejection logging

**Files:**
- Create: `risk_supervisor.py`
- Create: `tests/test_risk_supervisor.py`

**Step 1: Write the failing tests**

```python
# tests/test_risk_supervisor.py
import pytest
from risk_supervisor import RiskSupervisor


class TestRiskSupervisor:
    def test_initially_allows_trading(self):
        sup = RiskSupervisor()
        assert sup.trading_enabled is True

    def test_disable_on_daily_drawdown(self):
        sup = RiskSupervisor(max_daily_loss_pct=3.0)
        sup.record_pnl(-350.0, capital=10000.0)  # -3.5% > 3.0%
        assert sup.trading_enabled is False

    def test_disable_on_consecutive_losses(self):
        sup = RiskSupervisor(max_consecutive_losses=4)
        for _ in range(4):
            sup.record_trade_result(won=False)
        assert sup.trading_enabled is False

    def test_does_not_disable_on_wins(self):
        sup = RiskSupervisor()
        for _ in range(10):
            sup.record_trade_result(won=True)
        assert sup.trading_enabled is True

    def test_risk_mode_escalates_to_cautious(self):
        sup = RiskSupervisor(max_consecutive_losses=4)
        for _ in range(2):
            sup.record_trade_result(won=False)
        assert sup.risk_mode == "cautious"

    def test_risk_mode_escalates_to_defensive(self):
        sup = RiskSupervisor(max_consecutive_losses=4)
        for _ in range(3):
            sup.record_trade_result(won=False)
        assert sup.risk_mode == "defensive"

    def test_manual_enable(self):
        sup = RiskSupervisor(max_consecutive_losses=2)
        sup.record_trade_result(won=False)
        sup.record_trade_result(won=False)
        assert sup.trading_enabled is False
        sup.enable_trading()
        assert sup.trading_enabled is True

    def test_reset_daily_on_new_day(self):
        sup = RiskSupervisor(max_daily_loss_pct=2.0)
        sup.record_pnl(-250.0, capital=10000.0)  # -2.5%
        assert sup.trading_enabled is False
        sup.reset_daily()
        assert sup.trading_enabled is True
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_risk_supervisor.py -v
```

**Step 3: Implement `risk_supervisor.py`**

```python
"""RiskSupervisor — cross-cutting trading safety governor.

Watches drawdown, consecutive losses, and API health.
Has one power: disable_new_trades().
Does NOT adjust position sizes, entry prices, scores, or context.

The ContextEngine reads risk_mode from the supervisor via set_risk_mode() injection.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


class RiskSupervisor:
    """Safety governor that monitors trading health and can halt new trades.

    Args:
        max_daily_loss_pct: Halt trading if daily loss exceeds this % of capital.
        max_consecutive_losses: Halt trading after this many consecutive losses.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,
        max_consecutive_losses: int = 4,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses

        self._trading_enabled: bool = True
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._risk_mode: str = "normal"

    @property
    def trading_enabled(self) -> bool:
        return self._trading_enabled

    @property
    def risk_mode(self) -> str:
        return self._risk_mode

    def record_pnl(self, pnl: float, capital: float) -> None:
        """Record a PnL event. Halts trading if daily loss limit is breached.

        Args:
            pnl: Realised PnL for the trade (negative = loss).
            capital: Current total capital (for % calculation).
        """
        self._daily_pnl += pnl
        if capital > 0 and abs(self._daily_pnl) / capital * 100 >= self.max_daily_loss_pct:
            if self._daily_pnl < 0:
                _log.warning(
                    "Daily loss limit reached (%.2f%%). Halting new trades.",
                    abs(self._daily_pnl) / capital * 100,
                )
                self._trading_enabled = False
                self._risk_mode = "defensive"

    def record_trade_result(self, won: bool) -> None:
        """Record a trade outcome. Escalates risk_mode on consecutive losses.

        Args:
            won: True if the trade was profitable.
        """
        if won:
            self._consecutive_losses = 0
            if self._risk_mode != "normal":
                self._risk_mode = "normal"
                _log.info("Consecutive losses cleared — risk_mode reset to normal.")
        else:
            self._consecutive_losses += 1
            _log.info("Consecutive losses: %d", self._consecutive_losses)

            if self._consecutive_losses >= self.max_consecutive_losses:
                self._trading_enabled = False
                self._risk_mode = "defensive"
                _log.warning("Max consecutive losses reached. Halting new trades.")
            elif self._consecutive_losses >= self.max_consecutive_losses - 1:
                self._risk_mode = "defensive"
            elif self._consecutive_losses >= max(1, self.max_consecutive_losses // 2):
                self._risk_mode = "cautious"

    def enable_trading(self) -> None:
        """Re-enable trading (called manually via Telegram or after cooldown)."""
        self._trading_enabled = True
        self._consecutive_losses = 0
        self._risk_mode = "normal"
        _log.info("Trading re-enabled manually.")

    def reset_daily(self) -> None:
        """Reset daily PnL counter (called at start of each trading day)."""
        self._daily_pnl = 0.0
        if self._trading_enabled is False and self._consecutive_losses == 0:
            # Re-enable if halt was purely from daily loss (not consecutive losses)
            self._trading_enabled = True
            self._risk_mode = "normal"
        _log.debug("Daily PnL reset.")
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_risk_supervisor.py -v
./venv/bin/python -m ruff check --fix risk_supervisor.py
```

**Step 5: Commit**

```bash
git add risk_supervisor.py tests/test_risk_supervisor.py
git commit -m "feat: Phase 9 P4 — RiskSupervisor (kill switch: drawdown + consecutive losses)"
```

---

## Phase 5 — Spot Integration

> Goal: Wire context + trigger + decision into the agent main loop.
> Spot paper trading only. Every rejection is logged.

---

### Task 11: Update `config.py` with new env vars

**Files:**
- Modify: `config.py`
- Modify: `tests/test_config.py` (add Phase 9 config tests)

**Step 1: Add new config fields to `config.py`**

Locate the existing `Config` class and add after existing fields:

```python
# Phase 9: Context + Trigger
CONTEXT_INTERVAL_SECONDS: int = int(os.getenv("CONTEXT_INTERVAL_SECONDS", "900"))  # 15min
SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.50"))
MAX_PERP_LEVERAGE: float = float(os.getenv("MAX_PERP_LEVERAGE", "2.0"))
MAX_PERP_POSITIONS: int = int(os.getenv("MAX_PERP_POSITIONS", "2"))
MAX_SPOT_POSITIONS: int = int(os.getenv("MAX_SPOT_POSITIONS", "3"))
RISK_MAX_DAILY_LOSS_PCT: float = float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", "3.0"))
RISK_MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "4"))
```

Also add to `.env.example`:
```
# Phase 9 — Context + Trigger
CONTEXT_INTERVAL_SECONDS=900
SCORE_THRESHOLD=0.50
MAX_PERP_LEVERAGE=2.0
MAX_PERP_POSITIONS=2
MAX_SPOT_POSITIONS=3
RISK_MAX_DAILY_LOSS_PCT=3.0
RISK_MAX_CONSECUTIVE_LOSSES=4
```

**Step 2: Write config tests**

Append to `tests/test_config.py`:

```python
def test_context_interval_has_default():
    assert Config.CONTEXT_INTERVAL_SECONDS == 900

def test_score_threshold_has_default():
    assert Config.SCORE_THRESHOLD == 0.50

def test_max_perp_leverage_has_default():
    assert Config.MAX_PERP_LEVERAGE == 2.0
```

**Step 3: Run tests**

```bash
./venv/bin/python -m pytest tests/test_config.py -v -k "context or score or perp"
./venv/bin/python -m ruff check --fix config.py
```

**Step 4: Commit**

```bash
git add config.py .env.example tests/test_config.py
git commit -m "feat: Phase 9 P5 — config vars for context interval, score threshold, perp limits"
```

---

### Task 12: Create decision rejection logger

**Files:**
- Create: `decision_logger.py`
- Create: `tests/test_decision_logger.py`

**Step 1: Write the failing tests**

```python
# tests/test_decision_logger.py
from datetime import UTC, datetime, timedelta
from decision import ContextState, Decision, TriggerSignal
from decision_logger import DecisionLogger


def make_context():
    return ContextState(
        context_id="2026-03-03T14:15Z", swing_bias="bullish",
        allowed_directions=["long"], volatility_regime="normal",
        funding_pressure="neutral", whale_flow="neutral", oi_trend="neutral",
        key_levels={"support": 90000.0, "resistance": 95000.0, "poc": 92000.0},
        risk_mode="normal", confidence=0.75, tradeable=True,
        valid_until=datetime.now(UTC) + timedelta(minutes=15),
        updated_at=datetime.now(UTC),
    )


class TestDecisionLogger:
    def test_log_rejection_stores_entry(self):
        logger = DecisionLogger()
        d = Decision(action="reject", reason="context_not_tradeable")
        logger.log(context=make_context(), triggers=[], decision=d, symbol="BTC/USDT")
        assert len(logger.recent(10)) == 1

    def test_log_trade_stores_entry(self):
        logger = DecisionLogger()
        d = Decision(action="trade", direction="long", route="spot", score=0.6, reason="ok")
        logger.log(context=make_context(), triggers=[], decision=d, symbol="BTC/USDT")
        entries = logger.recent(10)
        assert entries[0]["action"] == "trade"

    def test_recent_returns_latest_n(self):
        logger = DecisionLogger(max_history=5)
        d = Decision(action="reject", reason="test")
        for _ in range(10):
            logger.log(context=make_context(), triggers=[], decision=d, symbol="BTC/USDT")
        assert len(logger.recent(5)) == 5

    def test_rejection_rate(self):
        logger = DecisionLogger()
        d_reject = Decision(action="reject", reason="test")
        d_trade = Decision(action="trade", direction="long", route="spot", score=0.6, reason="ok")
        ctx = make_context()
        for _ in range(3):
            logger.log(ctx, [], d_reject, "BTC/USDT")
        logger.log(ctx, [], d_trade, "BTC/USDT")
        assert logger.rejection_rate() == 0.75
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_decision_logger.py -v
```

**Step 3: Implement `decision_logger.py`**

```python
"""DecisionLogger — records every decision (trade or reject) for audit and tuning.

This is the primary truth-building mechanism during paper trading.
Reviewing why trades were rejected reveals whether the system is too tight
or too loose — more valuable than any backtest during early phases.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

from decision import ContextState, Decision, TriggerSignal

_log = logging.getLogger(__name__)

_DEFAULT_MAX_HISTORY = 1000


class DecisionLogger:
    """Records all decisions with full context and trigger snapshots.

    Args:
        max_history: Maximum number of entries to keep in memory.
    """

    def __init__(self, max_history: int = _DEFAULT_MAX_HISTORY) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)

    def log(
        self,
        context: ContextState,
        triggers: list[TriggerSignal],
        decision: Decision,
        symbol: str,
    ) -> None:
        """Record a decision with full context and trigger snapshot.

        Args:
            context: The ContextState that was evaluated.
            triggers: The TriggerSignals that were presented.
            decision: The Decision that was made.
            symbol: Trading pair this decision was for.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "action": decision.action,
            "reason": decision.reason,
            "direction": decision.direction,
            "route": decision.route,
            "score": decision.score,
            "context_id": context.context_id,
            "swing_bias": context.swing_bias,
            "tradeable": context.tradeable,
            "allowed_directions": context.allowed_directions,
            "context_confidence": context.confidence,
            "risk_mode": context.risk_mode,
            "trigger_count": len(triggers),
            "trigger_sources": [t.source for t in triggers],
        }
        self._history.append(entry)

        if decision.action == "trade":
            _log.info("[DECISION] TRADE %s %s via %s (score=%.2f)",
                      symbol, decision.direction, decision.route, decision.score or 0)
        else:
            _log.debug("[DECISION] REJECT %s — %s", symbol, decision.reason)

    def recent(self, n: int) -> list[dict[str, Any]]:
        """Return the most recent n decision log entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of decision dicts, most recent last.
        """
        return list(self._history)[-n:]

    def rejection_rate(self) -> float:
        """Fraction of decisions that were rejections (0.0–1.0).

        Returns:
            0.0 if no decisions logged yet.
        """
        if not self._history:
            return 0.0
        rejections = sum(1 for e in self._history if e["action"] == "reject")
        return rejections / len(self._history)
```

**Step 4: Run tests**

```bash
./venv/bin/python -m pytest tests/test_decision_logger.py -v
./venv/bin/python -m ruff check --fix decision_logger.py
```

**Step 5: Commit**

```bash
git add decision_logger.py tests/test_decision_logger.py
git commit -m "feat: Phase 9 P5 — DecisionLogger (every reject/trade logged with full context)"
```

---

### Task 13: Update `agent.py` main loop for Phase 9 architecture

**Files:**
- Modify: `agent.py`

> **Note**: This is the largest change. Read `agent.py` fully before editing.
> The goal is to add a parallel Phase 9 code path that runs alongside the old
> one during transition — controlled by a `USE_PHASE9_PIPELINE` env var.
> This avoids breaking the existing paper trading session.

**Step 1: Add Phase 9 imports to `agent.py`**

Find the imports section and add:

```python
from context_engine import ContextEngine
from decision import evaluate
from decision_logger import DecisionLogger
from multi_timeframe_fetcher import MultiTimeframeFetcher
from risk_supervisor import RiskSupervisor
from trigger_engine import TriggerEngine
```

**Step 2: Add Phase 9 components to `Agent.__init__`**

Locate `Agent.__init__` and add after existing component initialization:

```python
# Phase 9: Context + Trigger pipeline (enabled via USE_PHASE9_PIPELINE=true in .env)
self._use_phase9 = os.getenv("USE_PHASE9_PIPELINE", "false").lower() == "true"
if self._use_phase9:
    self._risk_supervisor = RiskSupervisor(
        max_daily_loss_pct=Config.RISK_MAX_DAILY_LOSS_PCT,
        max_consecutive_losses=Config.RISK_MAX_CONSECUTIVE_LOSSES,
    )
    self._context_engine = ContextEngine()
    self._trigger_engines: dict[str, TriggerEngine] = {
        pair: TriggerEngine(symbol=pair) for pair in Config.TRADING_PAIRS
    }
    self._mtf_fetcher = MultiTimeframeFetcher(exchange=self.fetcher.exchange)
    self._decision_logger = DecisionLogger()
    self._last_context_time: dict[str, float] = {}
    self._log.info("Phase 9 Context+Trigger pipeline enabled.")
```

**Step 3: Add `_run_phase9_cycle` method**

Add this method to the `Agent` class (before the main `_run_cycle` method):

```python
def _run_phase9_cycle(self, symbol: str) -> None:
    """Run one Phase 9 context+trigger evaluation cycle for a symbol.

    Logs every decision (trade or reject). Does not execute trades yet
    until Phase 9 is fully validated (USE_PHASE9_EXECUTE=true).

    Args:
        symbol: Trading pair to evaluate.
    """
    import time
    now = time.time()

    # Fetch multi-timeframe snapshot
    snapshot = self._mtf_fetcher.fetch(symbol)

    # Rebuild context every 15min or if never built
    last_ctx_time = self._last_context_time.get(symbol, 0)
    if now - last_ctx_time >= Config.CONTEXT_INTERVAL_SECONDS:
        # Get real-time data for context analyzers
        funding_rate = None
        net_whale_flow = None
        oi_change_pct = None
        price_change_pct = None

        try:
            if self.intelligence:
                intel = self._last_intelligence
                if intel:
                    for sig in intel.get("signals", []):
                        if sig.get("source") == "funding_oi":
                            funding_rate = sig.get("data", {}).get("funding_rate")
                        if sig.get("source") == "whale_tracker":
                            net_whale_flow = sig.get("data", {}).get("net_flow_usd")
        except Exception as e:
            self._log.debug("Phase 9 context data fetch failed: %s", e)

        # Sync risk mode from supervisor
        self._context_engine.set_risk_mode(self._risk_supervisor.risk_mode)

        self._current_context = self._context_engine.build(
            snapshot,
            funding_rate=funding_rate,
            net_whale_flow=net_whale_flow,
            oi_change_pct=oi_change_pct,
            price_change_pct=price_change_pct,
        )
        self._last_context_time[symbol] = now
        self._log.debug(
            "Phase 9 context [%s]: bias=%s tradeable=%s confidence=%.2f",
            symbol, self._current_context.swing_bias,
            self._current_context.tradeable, self._current_context.confidence,
        )

    # Collect triggers
    orderbook_data = None
    try:
        if self._last_intelligence:
            for sig in self._last_intelligence.get("signals", []):
                if sig.get("source") == "order_book":
                    orderbook_data = sig.get("data")
                    break
    except Exception:
        pass

    trigger_eng = self._trigger_engines[symbol]
    triggers = trigger_eng.collect(snapshot, orderbook_data=orderbook_data)

    # Evaluate
    context = getattr(self, "_current_context", None)
    if context is None:
        return

    decision = evaluate(context, triggers)
    self._decision_logger.log(
        context=context, triggers=triggers, decision=decision, symbol=symbol
    )

    # Phase 9 execution gate (enabled separately via USE_PHASE9_EXECUTE=true)
    if decision.action == "trade" and os.getenv("USE_PHASE9_EXECUTE", "false").lower() == "true":
        self._log.info(
            "Phase 9 TRADE signal: %s %s via %s (score=%.2f)",
            symbol, decision.direction, decision.route, decision.score or 0,
        )
        # TODO: wire to executor (Task 14)
```

**Step 4: Call `_run_phase9_cycle` from `_run_cycle`**

Find the per-pair loop inside `_run_cycle` and add the Phase 9 call:

```python
# Inside the per-pair loop, after existing strategy processing:
if self._use_phase9:
    try:
        self._run_phase9_cycle(symbol)
    except Exception as e:
        self._log.error("Phase 9 cycle failed for %s: %s", symbol, e)
```

**Step 5: Run full test suite**

```bash
./venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
./venv/bin/python -m ruff check --fix agent.py
```
Expected: Existing tests still pass. New phase 9 code is guarded by env var.

**Step 6: Commit**

```bash
git add agent.py
git commit -m "feat: Phase 9 P5 — Phase 9 pipeline wired into agent.py (USE_PHASE9_PIPELINE=true)"
```

---

## Phase 6 — Perp Execution

> Goal: Add liquidation + funding extreme triggers. Add perp executor.
> Perp paper trading only. Enable with USE_PHASE9_PERP=true.

---

### Task 14: Create `triggers/liquidation.py` and `triggers/funding_extreme.py`

**Files:**
- Create: `triggers/liquidation.py`
- Create: `triggers/funding_extreme.py`
- Modify: `tests/test_triggers.py` (add perp trigger tests)
- Modify: `trigger_engine.py` (add perp triggers, gated by USE_PHASE9_PERP)

**Step 1: Write the failing tests**

Append to `tests/test_triggers.py`:

```python
from triggers.liquidation import LiquidationTrigger
from triggers.funding_extreme import FundingExtremeTrigger


class TestLiquidationTrigger:
    def test_long_liquidation_cascade_produces_short_signal(self):
        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 50_000_000, "direction": "long"})
        assert any(s.direction == "short" for s in signals)

    def test_short_liquidation_cascade_produces_long_signal(self):
        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 50_000_000, "direction": "short"})
        assert any(s.direction == "long" for s in signals)

    def test_small_liquidation_ignored(self):
        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 100_000, "direction": "long"})
        assert signals == []

    def test_signals_have_high_urgency(self):
        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 50_000_000, "direction": "long"})
        assert all(s.urgency == "high" for s in signals)

    def test_none_data_returns_empty(self):
        trigger = LiquidationTrigger(symbol="BTC/USDT")
        assert trigger.evaluate(None) == []


class TestFundingExtremeTrigger:
    def test_extreme_positive_funding_produces_short(self):
        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=0.0012)  # 0.12% > extreme
        assert any(s.direction == "short" for s in signals)

    def test_extreme_negative_funding_produces_long(self):
        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=-0.0006)
        assert any(s.direction == "long" for s in signals)

    def test_normal_funding_produces_no_signal(self):
        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=0.0001)
        assert signals == []

    def test_signals_have_high_urgency(self):
        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=0.0012)
        assert all(s.urgency == "high" for s in signals)
```

**Step 2: Run to verify fail**

```bash
./venv/bin/python -m pytest tests/test_triggers.py -k "Liquidation or FundingExtreme" -v
```

**Step 3: Implement both trigger files**

```python
# triggers/liquidation.py
"""LiquidationTrigger — detects liquidation cascades and fires perp event signals.

Large liquidation volumes create momentum in the opposite direction:
  - Long liquidations → bearish cascade → short signal
  - Short liquidations → short squeeze → long signal
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_SIGNAL_TTL_MINUTES = 15
_MIN_LIQ_USD = 10_000_000  # $10M minimum to be significant


class LiquidationTrigger:
    """Fires high-urgency TriggerSignals on significant liquidation events.

    Args:
        symbol: Trading pair. "BTC" cascades set symbol_scope="market".
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._base = symbol.split("/")[0]

    def evaluate(self, liq_data: Optional[dict[str, Any]]) -> list[TriggerSignal]:
        """Evaluate liquidation data and return cascade triggers.

        Args:
            liq_data: Dict with keys ``liq_volume_usd`` (float) and ``direction``
                ("long" = longs being liquidated = bearish; "short" = bullish).
                None returns empty list.

        Returns:
            List of high-urgency TriggerSignal (empty if below threshold).
        """
        if liq_data is None:
            return []

        volume = liq_data.get("liq_volume_usd", 0)
        direction = liq_data.get("direction", "")

        if volume < _MIN_LIQ_USD or direction not in ("long", "short"):
            return []

        # Long liquidations = bearish signal; short liquidations = bullish
        signal_direction = "short" if direction == "long" else "long"
        strength = min(0.6 + (volume / 100_000_000) * 0.3, 0.95)

        # BTC liquidation cascades are market-wide
        scope = "market" if self._base == "BTC" else self._base

        return [TriggerSignal(
            trigger_id=str(uuid.uuid4()),
            source="liquidation",
            direction=signal_direction,
            strength=strength,
            urgency="high",
            symbol_scope=scope,
            reason=f"${volume/1e6:.0f}M {direction} liquidation cascade",
            expires_at=datetime.now(UTC) + timedelta(minutes=_SIGNAL_TTL_MINUTES),
            raw_data={"liq_volume_usd": volume, "liquidated_direction": direction},
        )]
```

```python
# triggers/funding_extreme.py
"""FundingExtremeTrigger — fires when funding rate reaches contrarian extremes.

Extreme funding = crowded trade = mean-reversion setup:
  - Rate > 0.10%/8h (long crowded) → fade the crowd → short signal
  - Rate < -0.05%/8h (short crowded) → fade the crowd → long signal
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Optional

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_SIGNAL_TTL_MINUTES = 60
_LONG_EXTREME = 0.0010   # 0.10% per 8h
_SHORT_EXTREME = -0.0005  # -0.05% per 8h


class FundingExtremeTrigger:
    """Fires high-urgency TriggerSignals when funding reaches contrarian extremes.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(self, funding_rate: Optional[float]) -> list[TriggerSignal]:
        """Evaluate current funding rate and return extreme signals.

        Args:
            funding_rate: Current 8h funding rate as decimal. None = no signal.

        Returns:
            List of high-urgency TriggerSignal (empty if not at extreme).
        """
        if funding_rate is None:
            return []

        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)

        if funding_rate >= _LONG_EXTREME:
            strength = min(0.55 + (funding_rate - _LONG_EXTREME) * 100, 0.90)
            return [TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="funding_extreme",
                direction="short",
                strength=strength,
                urgency="high",
                symbol_scope=self._symbol_scope,
                reason=f"Funding extreme long {funding_rate*100:.3f}%/8h — fade crowd",
                expires_at=expiry,
                raw_data={"funding_rate": funding_rate},
            )]

        if funding_rate <= _SHORT_EXTREME:
            strength = min(0.55 + abs(funding_rate - _SHORT_EXTREME) * 100, 0.90)
            return [TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="funding_extreme",
                direction="long",
                strength=strength,
                urgency="high",
                symbol_scope=self._symbol_scope,
                reason=f"Funding extreme short {funding_rate*100:.3f}%/8h — fade crowd",
                expires_at=expiry,
                raw_data={"funding_rate": funding_rate},
            )]

        return []
```

**Step 4: Update `trigger_engine.py` to include perp triggers**

```python
# In TriggerEngine.__init__, after existing triggers:
self._use_perp = os.getenv("USE_PHASE9_PERP", "false").lower() == "true"
if self._use_perp:
    from triggers.liquidation import LiquidationTrigger
    from triggers.funding_extreme import FundingExtremeTrigger
    self._liquidation = LiquidationTrigger(symbol=symbol)
    self._funding_extreme = FundingExtremeTrigger(symbol=symbol)

# In TriggerEngine.collect(), after existing triggers:
if self._use_perp:
    try:
        liq_data = None  # TODO: wire real-time liquidation feed
        signals.extend(self._liquidation.evaluate(liq_data))
    except Exception as e:
        _log.warning("LiquidationTrigger failed: %s", e)
    try:
        funding_rate = None  # TODO: wire from intelligence/funding_oi.py
        signals.extend(self._funding_extreme.evaluate(funding_rate))
    except Exception as e:
        _log.warning("FundingExtremeTrigger failed: %s", e)
```

Add `import os` to top of `trigger_engine.py` if not present.

**Step 5: Run tests**

```bash
./venv/bin/python -m pytest tests/test_triggers.py -v
./venv/bin/python -m ruff check --fix triggers/liquidation.py triggers/funding_extreme.py trigger_engine.py
```

**Step 6: Commit**

```bash
git add triggers/liquidation.py triggers/funding_extreme.py trigger_engine.py tests/test_triggers.py
git commit -m "feat: Phase 9 P6 — LiquidationTrigger + FundingExtremeTrigger (perp event triggers)"
```

---

### Task 15: Run full test suite + final lint check

**Step 1: Run all tests**

```bash
./venv/bin/python -m pytest tests/ -v 2>&1 | tail -30
```
Expected: All pre-existing tests pass. New tests pass.

**Step 2: Run ruff on all new files**

```bash
./venv/bin/python -m ruff check --fix \
  decision.py data_snapshot.py multi_timeframe_fetcher.py \
  context_engine.py context/ \
  trigger_engine.py triggers/ \
  risk_supervisor.py decision_logger.py
```
Expected: 0 errors.

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: Phase 9 — final lint pass, all tests green"
```

---

## Activation Checklist

To enable Phase 9 in paper trading mode, add to `.env`:

```
USE_PHASE9_PIPELINE=true    # Enable context+trigger pipeline (logging only)
USE_PHASE9_EXECUTE=true     # Enable spot trade execution via Phase 9
USE_PHASE9_PERP=true        # Enable perp triggers (after spot is validated)
```

**Recommended sequence:**
1. `USE_PHASE9_PIPELINE=true` only → watch rejection logs for 48h
2. Add `USE_PHASE9_EXECUTE=true` → spot paper trading for 2 weeks
3. Review `DecisionLogger.rejection_rate()` — if >90%, loosen `SCORE_THRESHOLD`
4. Add `USE_PHASE9_PERP=true` → perp paper trading only after spot is validated

---

## Files Summary

| File | Status | Phase |
|------|--------|-------|
| `decision.py` | New | P1 |
| `data_snapshot.py` | New | P2 |
| `multi_timeframe_fetcher.py` | New | P2 |
| `context/__init__.py` | New | P3 |
| `context/swing.py` | New | P3 |
| `context/funding.py` | New | P3 |
| `context/whale_flow.py` | New | P3 |
| `context/oi_trend.py` | New | P3 |
| `context_engine.py` | New | P3 |
| `triggers/__init__.py` | New | P4 |
| `triggers/momentum.py` | New | P4 |
| `triggers/orderflow.py` | New | P4 |
| `trigger_engine.py` | New | P4 |
| `risk_supervisor.py` | New | P4 |
| `decision_logger.py` | New | P5 |
| `agent.py` | Modified | P5 |
| `config.py` | Modified | P5 |
| `triggers/liquidation.py` | New | P6 |
| `triggers/funding_extreme.py` | New | P6 |
| **`perp_executor.py`** | **TODO** | P6+ |
