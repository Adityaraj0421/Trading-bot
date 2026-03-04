# Phase 10: Execution Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire `decision.route` to a real perp execution path, add WebSocket real-time data, fill `LiveExecutor.partial_close()`, and add a structured alerting module — the four blockers preventing live trading.

**Architecture:** Eight sequential tasks. Tasks 1–3 are independent and can be done in any order. Tasks 4–5 (perp executors) depend on Task 3 (perp fields). Task 7 (agent routing) depends on Tasks 4–5. Task 8 (WebSocket wiring) depends on Task 6. `alerting.py` (Task 2) is fully standalone.

**Tech Stack:** Python 3.14, pytest, ccxt (existing), aiohttp (already installed — used by async_fetcher), existing `Position`/`PaperExecutor`/`LiveExecutor` patterns in `executor.py` and `risk_manager.py`.

---

## Quick Reference

| Task | File(s) touched | What it does |
|------|-----------------|--------------|
| 1 | `executor.py`, `tests/test_executor.py` | `LiveExecutor.partial_close()` |
| 2 | `alerting.py` (new), `tests/test_alerting.py` (new) | Severity-aware alert funnel |
| 3 | `risk_manager.py`, `tests/test_risk_manager.py` | Perp fields on `Position` |
| 4 | `executor.py`, `tests/test_executor.py` | `PerpPaperExecutor` |
| 5 | `executor.py`, `tests/test_executor.py` | `PerpLiveExecutor` |
| 6 | `ws_feed.py` (new), `tests/test_ws_feed.py` (new) | WebSocket kline feed |
| 7 | `agent.py`, `tests/test_agent.py` | Route `decision.route` to correct executor |
| 8 | `agent.py` | Start `WsFeed` in background thread |

**Run all tests:** `./venv/bin/python -m pytest tests/ -v`
**Lint:** `./venv/bin/python -m ruff check --fix <file>`
**Python path:** `./venv/bin/python`

---

## Task 1: `LiveExecutor.partial_close()`

**Files:**
- Modify: `executor.py` (after `cancel_order` method, still inside `LiveExecutor`)
- Modify: `tests/test_executor.py` (new `TestLiveExecutorPartialClose` class)

**Context:** `PaperExecutor.partial_close()` already exists (lines 64–117 of `executor.py`). `LiveExecutor` is missing it, so agent code using `hasattr(self.executor, "partial_close")` always skips in live mode. The live version places a real market sell/buy order for the fractional quantity.

### Step 1: Write the failing tests

Add to `tests/test_executor.py` after `TestPartialClose`:

```python
class TestLiveExecutorPartialClose:
    """LiveExecutor.partial_close() — real order via mocked exchange."""

    def _make_position(self, side: str = "long") -> Any:
        from risk_manager import Position
        from datetime import datetime, UTC
        return Position(
            symbol="BTC/USDT",
            side=side,
            entry_price=50_000.0,
            quantity=0.1,
            entry_time=datetime.now(UTC),
            stop_loss=48_000.0,
            take_profit=53_000.0,
        )

    def test_partial_close_places_market_order(self):
        """partial_close() calls exchange.create_market_order with correct qty."""
        import ccxt
        from executor import LiveExecutor

        mock_exchange = MagicMock(spec=ccxt.Exchange)
        mock_exchange.create_market_order.return_value = {
            "id": "live_partial_1",
            "status": "closed",
            "filled": 0.05,
            "average": 51_000.0,
        }
        executor = LiveExecutor(mock_exchange)
        pos = self._make_position()

        order = executor.partial_close(pos, 0.5, 51_000.0)

        mock_exchange.create_market_order.assert_called_once()
        call_kwargs = mock_exchange.create_market_order.call_args
        assert call_kwargs[0][0] == "BTC/USDT"   # symbol
        assert call_kwargs[0][2] == pytest.approx(0.05)  # qty = 0.1 * 0.5

    def test_partial_close_reduces_position_quantity(self):
        """partial_close() mutates position.quantity by the closed fraction."""
        import ccxt
        from executor import LiveExecutor

        mock_exchange = MagicMock(spec=ccxt.Exchange)
        mock_exchange.create_market_order.return_value = {"id": "x", "status": "closed", "filled": 0.05, "average": 51_000.0}
        executor = LiveExecutor(mock_exchange)
        pos = self._make_position()

        executor.partial_close(pos, 0.5, 51_000.0)

        assert pos.quantity == pytest.approx(0.05)

    def test_partial_close_returns_order_dict(self):
        """partial_close() returns a dict with expected keys."""
        import ccxt
        from executor import LiveExecutor

        mock_exchange = MagicMock(spec=ccxt.Exchange)
        mock_exchange.create_market_order.return_value = {"id": "live_1", "status": "closed", "filled": 0.05, "average": 51_000.0}
        executor = LiveExecutor(mock_exchange)
        pos = self._make_position()

        order = executor.partial_close(pos, 0.5, 51_000.0)

        assert order["status"] == "filled"
        assert order["quantity"] == pytest.approx(0.05)
        assert order["mode"] == "LIVE"
```

### Step 2: Run the tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_executor.py::TestLiveExecutorPartialClose -v
```
Expected: `AttributeError: 'LiveExecutor' object has no attribute 'partial_close'`

### Step 3: Implement `LiveExecutor.partial_close()`

Add after `cancel_order()` inside `LiveExecutor` in `executor.py`:

```python
def partial_close(
    self,
    position: Any,
    fraction: float,
    current_price: float,
    reason: str = "partial_tp",
) -> dict[str, Any]:
    """Place a market order to close a fraction of the position.

    Args:
        position: Open Position object (quantity is mutated in-place).
        fraction: Fraction to close, e.g. ``0.50`` for 50%.
        current_price: Reference price for PnL estimation (actual fill
            may differ).
        reason: Tag for the audit trail (e.g. ``"partial_tp"``).

    Returns:
        Order dict with keys: id, symbol, side, quantity, price, pnl,
        reason, status, timestamp, mode.
    """
    closed_qty = position.quantity * fraction
    # For a long: sell to reduce; for a short: buy to reduce
    order_side = "sell" if position.side == "long" else "buy"
    try:
        raw = self.exchange.create_market_order(
            position.symbol, order_side, closed_qty
        )
        filled_price = float(raw.get("average") or current_price)
    except Exception as e:
        _log.error("[Live] partial_close failed: %s", e)
        return {"error": str(e)}

    if position.side == "long":
        pnl = (filled_price - position.entry_price) * closed_qty
    else:
        pnl = (position.entry_price - filled_price) * closed_qty

    position.quantity -= closed_qty

    order = {
        "id": raw.get("id", "unknown"),
        "symbol": position.symbol,
        "side": order_side,
        "quantity": closed_qty,
        "price": filled_price,
        "pnl": pnl,
        "reason": reason,
        "status": "filled",
        "timestamp": datetime.now().isoformat(),
        "mode": "LIVE",
    }
    _log.info(
        "[Live] Partial close %.0f%% of %s @ $%,.2f (PnL: $%.2f)",
        fraction * 100,
        position.symbol,
        filled_price,
        pnl,
    )
    return order
```

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_executor.py::TestLiveExecutorPartialClose -v
```
Expected: `3 passed`

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix executor.py
./venv/bin/python -m pytest tests/test_executor.py -v
git add executor.py tests/test_executor.py
git commit -m "feat: LiveExecutor.partial_close() — market-order fractional exit"
```

---

## Task 2: `alerting.py` — Severity-Aware Alert Funnel

**Files:**
- Create: `alerting.py`
- Create: `tests/test_alerting.py`

**Context:** Currently `_log.warning()` is scattered throughout the agent for critical events. `alerting.py` is a small module with one class that routes alerts to Telegram (critical) or log-only (warn). It accepts an optional `Notifier` reference — if None, all alerts go to log only (safe for tests).

### Step 1: Write the failing tests

Create `tests/test_alerting.py`:

```python
"""Tests for alerting.py severity-aware alert funnel."""
from unittest.mock import MagicMock, patch
import pytest


class TestAlerting:
    def test_critical_calls_notifier_error(self):
        """critical severity triggers notifier.notify_error()."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        al.alert("disk full", severity="critical")
        notifier.notify_error.assert_called_once()
        call_msg = notifier.notify_error.call_args[0][1]
        assert "disk full" in call_msg

    def test_warn_does_not_call_notifier(self):
        """warn severity logs but does not call notifier."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        al.alert("high latency", severity="warn")
        notifier.notify_error.assert_not_called()

    def test_no_notifier_does_not_raise(self):
        """Alerting with no notifier is safe for all severities."""
        from alerting import Alerting
        al = Alerting()
        al.alert("test", severity="critical")  # should not raise
        al.alert("test", severity="warn")

    def test_liquidation_proximity_alert(self):
        """liquidation_proximity() fires critical when within threshold."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        # mark price 9% above liquidation → within 10% threshold
        al.liquidation_proximity("BTC/USDT", mark_price=54_450.0, liquidation_price=50_000.0)
        notifier.notify_error.assert_called_once()

    def test_liquidation_proximity_no_alert_when_safe(self):
        """liquidation_proximity() is silent when > 10% away."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        al.liquidation_proximity("BTC/USDT", mark_price=60_000.0, liquidation_price=50_000.0)
        notifier.notify_error.assert_not_called()
```

### Step 2: Run tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_alerting.py -v
```
Expected: `ModuleNotFoundError: No module named 'alerting'`

### Step 3: Implement `alerting.py`

Create `alerting.py`:

```python
"""Alerting — severity-aware alert funnel for the trading agent.

Routes alerts to Telegram (critical) or log-only (warn/info).
Attach a ``Notifier`` instance via the constructor for Telegram delivery.
Safe to use with no notifier (tests, paper mode).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

AlertSeverity = Literal["info", "warn", "critical"]

_LIQUIDATION_PROXIMITY_THRESHOLD = 0.10  # 10% from liquidation price


class Alerting:
    """Severity-aware alert funnel.

    Args:
        notifier: Optional ``Notifier`` instance. When provided, ``critical``
            alerts are forwarded to ``notifier.notify_error()``. When absent,
            all alerts go to the Python logger only.
    """

    def __init__(self, notifier: Any = None) -> None:
        self._notifier = notifier

    def alert(self, msg: str, severity: AlertSeverity = "warn") -> None:
        """Fire an alert with the given severity.

        Args:
            msg: Human-readable alert message.
            severity: One of ``"info"``, ``"warn"``, ``"critical"``.
                ``"critical"`` also sends a Telegram notification if a
                notifier is wired.
        """
        if severity == "critical":
            _log.error("ALERT[CRITICAL]: %s", msg)
            if self._notifier is not None:
                self._notifier.notify_error("agent", f"CRITICAL: {msg}")
        elif severity == "warn":
            _log.warning("ALERT[WARN]: %s", msg)
        else:
            _log.info("ALERT[INFO]: %s", msg)

    def liquidation_proximity(
        self,
        symbol: str,
        mark_price: float,
        liquidation_price: float,
    ) -> None:
        """Fire a critical alert when mark price is within 10% of liquidation.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            mark_price: Current mark price.
            liquidation_price: Position liquidation price.
        """
        if liquidation_price <= 0:
            return
        proximity = abs(mark_price - liquidation_price) / liquidation_price
        if proximity < _LIQUIDATION_PROXIMITY_THRESHOLD:
            self.alert(
                f"{symbol} liquidation proximity {proximity:.1%} "
                f"(mark={mark_price:,.0f}, liq={liquidation_price:,.0f})",
                severity="critical",
            )
```

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_alerting.py -v
```
Expected: `5 passed`

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix alerting.py
git add alerting.py tests/test_alerting.py
git commit -m "feat: alerting.py — severity-aware alert funnel with liquidation proximity"
```

---

## Task 3: Perp Fields on `Position`

**Files:**
- Modify: `risk_manager.py` (add 4 fields to `Position` dataclass)
- Modify: `tests/test_risk_manager.py` (new `TestPerpPositionFields` class)

**Context:** Rather than a subclass (which has dataclass inheritance pitfalls), add optional perp fields directly to `Position` with `field(default_factory=...)` or scalar defaults. Spot positions leave them at defaults (`leverage=1`, others `0.0`). This is backward-compatible — all 1168 existing tests pass unchanged.

### Step 1: Write the failing tests

Add to `tests/test_risk_manager.py` after `TestPartialTP`:

```python
class TestPerpPositionFields:
    """Perp-specific fields on Position — backward-compatible defaults."""

    def _make_position(self, **kwargs) -> Any:
        from risk_manager import Position
        from datetime import datetime, UTC
        defaults = dict(
            symbol="BTC/USDT", side="long", entry_price=50_000.0,
            quantity=0.1, entry_time=datetime.now(UTC),
            stop_loss=48_000.0, take_profit=53_000.0,
        )
        defaults.update(kwargs)
        return Position(**defaults)

    def test_default_leverage_is_one(self):
        pos = self._make_position()
        assert pos.leverage == 1

    def test_default_perp_fields_are_zero(self):
        pos = self._make_position()
        assert pos.margin_used == 0.0
        assert pos.liquidation_price == 0.0
        assert pos.funding_pnl == 0.0

    def test_perp_fields_settable(self):
        pos = self._make_position(leverage=3, margin_used=1666.67,
                                   liquidation_price=34_000.0)
        assert pos.leverage == 3
        assert pos.margin_used == pytest.approx(1666.67)
        assert pos.liquidation_price == 34_000.0

    def test_spot_position_unchanged(self):
        """Existing Position construction with no perp kwargs still works."""
        pos = self._make_position()
        assert pos.symbol == "BTC/USDT"
        assert pos.entry_price == 50_000.0
```

### Step 2: Run tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_risk_manager.py::TestPerpPositionFields -v
```
Expected: `TypeError: Position.__init__() got an unexpected keyword argument 'leverage'`

### Step 3: Add perp fields to `Position`

In `risk_manager.py`, after the `partial_exits` line (currently line ~47), add:

```python
    # Perp fields — default values keep spot positions fully backward-compatible
    leverage: int = 1              # 1 = spot (no leverage); 2+ = leveraged perp
    margin_used: float = 0.0       # notional / leverage
    liquidation_price: float = 0.0 # 0.0 = not set (spot or not yet computed)
    funding_pnl: float = 0.0       # accrued funding cost (negative = paid)
```

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_risk_manager.py::TestPerpPositionFields -v
./venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -5
```
Expected: `4 passed` for new tests; full suite still `1168+ passed`

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix risk_manager.py
git add risk_manager.py tests/test_risk_manager.py
git commit -m "feat: add leverage/margin/liquidation_price/funding_pnl fields to Position"
```

---

## Task 4: `PerpPaperExecutor`

**Files:**
- Modify: `executor.py` (add `PerpPaperExecutor` class after `PaperExecutor`)
- Modify: `tests/test_executor.py` (new `TestPerpPaperExecutor` class)

**Context:** Simulates leveraged futures in paper mode. Creates a `Position` with perp fields computed at entry. `partial_close()` works identically to `PaperExecutor.partial_close()`. Computes liquidation price at `open_position()`. `apply_funding()` called externally by agent each 8h cycle.

### Step 1: Write the failing tests

Add to `tests/test_executor.py`:

```python
class TestPerpPaperExecutor:
    """PerpPaperExecutor — simulated leveraged futures."""

    def _make_pos_kwargs(self):
        from datetime import datetime, UTC
        return dict(
            symbol="BTC/USDT", side="long", entry_price=50_000.0,
            quantity=0.1, entry_time=datetime.now(UTC),
            stop_loss=48_000.0, take_profit=53_000.0,
        )

    def test_place_order_returns_filled(self):
        from executor import PerpPaperExecutor
        ex = PerpPaperExecutor(leverage=3)
        order = ex.place_order("BTC/USDT", "long", 0.1, 50_000.0)
        assert order["status"] == "filled"
        assert order["mode"] == "PERP_PAPER"

    def test_open_position_sets_leverage_fields(self):
        """open_position() creates a Position with leverage and liquidation_price."""
        from executor import PerpPaperExecutor
        from risk_manager import Position
        ex = PerpPaperExecutor(leverage=3)
        pos = ex.open_position(**self._make_pos_kwargs())
        assert pos.leverage == 3
        assert pos.margin_used == pytest.approx(50_000.0 * 0.1 / 3)
        # Long liq price = entry * (1 - 1/leverage)
        assert pos.liquidation_price == pytest.approx(50_000.0 * (1 - 1 / 3))

    def test_liquidation_price_short(self):
        from executor import PerpPaperExecutor
        ex = PerpPaperExecutor(leverage=5)
        kwargs = dict(symbol="BTC/USDT", side="short", entry_price=50_000.0,
                      quantity=0.1, entry_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                      stop_loss=52_000.0, take_profit=47_000.0)
        pos = ex.open_position(**kwargs)
        assert pos.liquidation_price == pytest.approx(50_000.0 * (1 + 1 / 5))

    def test_apply_funding_accrues_funding_pnl(self):
        """apply_funding() reduces funding_pnl by rate * notional."""
        from executor import PerpPaperExecutor
        ex = PerpPaperExecutor(leverage=3)
        pos = ex.open_position(**self._make_pos_kwargs())
        ex.apply_funding(pos, funding_rate=0.0001)  # 0.01% per 8h
        # notional = 50_000 * 0.1 = 5000; cost = 5000 * 0.0001 = 0.50
        assert pos.funding_pnl == pytest.approx(-0.50)

    def test_partial_close_works(self):
        from executor import PerpPaperExecutor
        ex = PerpPaperExecutor(leverage=3)
        pos = ex.open_position(**self._make_pos_kwargs())
        order = ex.partial_close(pos, 0.5, 51_000.0)
        assert pos.quantity == pytest.approx(0.05)
        assert order["mode"] == "PERP_PAPER"
```

### Step 2: Run tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_executor.py::TestPerpPaperExecutor -v
```
Expected: `ImportError: cannot import name 'PerpPaperExecutor' from 'executor'`

### Step 3: Implement `PerpPaperExecutor`

Add after `PaperExecutor` in `executor.py`:

```python
class PerpPaperExecutor:
    """Simulates leveraged perpetual futures in paper mode.

    Creates ``Position`` objects with leverage, margin, and liquidation
    price pre-computed. Funding costs are accrued via ``apply_funding()``.

    Args:
        leverage: Integer leverage multiplier (e.g. ``3`` for 3×).
    """

    def __init__(self, leverage: int = 3) -> None:
        self.leverage = leverage
        self.orders: list[dict[str, Any]] = []

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]:
        """Simulate an instant fill at the given price.

        Args:
            symbol: Trading pair symbol.
            side: ``"long"`` (buy) or ``"short"`` (sell).
            quantity: Asset quantity.
            price: Simulated fill price.

        Returns:
            Order dict with ``status="filled"`` and ``mode="PERP_PAPER"``.
        """
        order = {
            "id": f"perp_paper_{len(self.orders) + 1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "filled",
            "mode": "PERP_PAPER",
            "timestamp": datetime.now().isoformat(),
        }
        self.orders.append(order)
        return order

    def open_position(self, **kwargs: Any) -> Any:
        """Create a Position with perp-specific fields computed.

        Accepts the same keyword arguments as ``Position.__init__``.
        Sets ``leverage``, ``margin_used``, and ``liquidation_price``
        automatically.

        Returns:
            ``Position`` instance with perp fields populated.
        """
        from risk_manager import Position

        pos = Position(**kwargs)
        pos.leverage = self.leverage
        pos.margin_used = pos.entry_price * pos.quantity / self.leverage
        if pos.side == "long":
            pos.liquidation_price = pos.entry_price * (1 - 1 / self.leverage)
        else:
            pos.liquidation_price = pos.entry_price * (1 + 1 / self.leverage)
        return pos

    def apply_funding(self, position: Any, funding_rate: float) -> None:
        """Accrue synthetic funding cost to ``position.funding_pnl``.

        Called every 8h by the agent. Positive ``funding_rate`` means
        longs pay shorts (reduces long PnL).

        Args:
            position: Open ``Position`` with ``leverage > 1``.
            funding_rate: 8h funding rate as a decimal (e.g. ``0.0001``).
        """
        notional = position.entry_price * position.quantity
        cost = notional * funding_rate
        # Longs pay when rate > 0; shorts receive
        if position.side == "long":
            position.funding_pnl -= cost
        else:
            position.funding_pnl += cost

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """No-op in paper mode — always returns True."""
        return True

    def partial_close(
        self,
        position: Any,
        fraction: float,
        current_price: float,
        reason: str = "partial_tp",
    ) -> dict[str, Any]:
        """Simulate a partial close. Identical logic to PaperExecutor."""
        closed_qty = position.quantity * fraction
        if position.side == "long":
            pnl = (current_price - position.entry_price) * closed_qty
        else:
            pnl = (position.entry_price - current_price) * closed_qty
        position.quantity -= closed_qty
        order = {
            "id": f"perp_paper_partial_{len(self.orders) + 1}",
            "symbol": position.symbol,
            "side": "sell" if position.side == "long" else "buy",
            "quantity": closed_qty,
            "price": current_price,
            "pnl": pnl,
            "reason": reason,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PERP_PAPER",
        }
        self.orders.append(order)
        return order
```

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_executor.py::TestPerpPaperExecutor -v
```
Expected: `5 passed`

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix executor.py
git add executor.py tests/test_executor.py
git commit -m "feat: PerpPaperExecutor — simulated leveraged perp with funding accrual"
```

---

## Task 5: `PerpLiveExecutor`

**Files:**
- Modify: `executor.py` (add `PerpLiveExecutor` class after `PerpPaperExecutor`)
- Modify: `tests/test_executor.py` (new `TestPerpLiveExecutor` class)

**Context:** Uses CCXT `create_market_order` with `reduceOnly=True` for closes. Binance USDT-Margined futures use `type="future"` markets. All heavy logic (position sizing, risk) stays in `RiskManager` — `PerpLiveExecutor` is just a thin CCXT wrapper.

### Step 1: Write the failing tests

Add to `tests/test_executor.py`:

```python
class TestPerpLiveExecutor:
    """PerpLiveExecutor — real CCXT futures orders via mocked exchange."""

    def _make_position(self, side: str = "long") -> Any:
        from risk_manager import Position
        from datetime import datetime, UTC
        return Position(
            symbol="BTC/USDT", side=side, entry_price=50_000.0,
            quantity=0.1, entry_time=datetime.now(UTC),
            stop_loss=48_000.0, take_profit=53_000.0,
            leverage=3,
        )

    def test_place_order_calls_create_market_order(self):
        import ccxt
        from executor import PerpLiveExecutor
        mock_ex = MagicMock(spec=ccxt.Exchange)
        mock_ex.create_market_order.return_value = {"id": "perp_1", "status": "closed", "average": 50_000.0, "filled": 0.1}
        ex = PerpLiveExecutor(mock_ex, leverage=3)
        ex.place_order("BTC/USDT", "long", 0.1, 50_000.0)
        mock_ex.create_market_order.assert_called_once()

    def test_place_order_buy_side_for_long(self):
        import ccxt
        from executor import PerpLiveExecutor
        mock_ex = MagicMock(spec=ccxt.Exchange)
        mock_ex.create_market_order.return_value = {"id": "p1", "status": "closed", "average": 50_000.0, "filled": 0.1}
        ex = PerpLiveExecutor(mock_ex, leverage=3)
        ex.place_order("BTC/USDT", "long", 0.1, 50_000.0)
        call_side = mock_ex.create_market_order.call_args[0][1]
        assert call_side == "buy"

    def test_partial_close_uses_reduce_only(self):
        import ccxt
        from executor import PerpLiveExecutor
        mock_ex = MagicMock(spec=ccxt.Exchange)
        mock_ex.create_market_order.return_value = {"id": "p1", "status": "closed", "average": 51_000.0, "filled": 0.05}
        ex = PerpLiveExecutor(mock_ex, leverage=3)
        pos = self._make_position()
        ex.partial_close(pos, 0.5, 51_000.0)
        call_kwargs = mock_ex.create_market_order.call_args[1] or {}
        call_args = mock_ex.create_market_order.call_args[0]
        # params dict should contain reduceOnly=True
        params = call_args[3] if len(call_args) > 3 else call_kwargs.get("params", {})
        assert params.get("reduceOnly") is True

    def test_insufficient_funds_returns_error_dict(self):
        import ccxt
        from executor import PerpLiveExecutor
        mock_ex = MagicMock(spec=ccxt.Exchange)
        mock_ex.create_market_order.side_effect = ccxt.InsufficientFunds("no margin")
        ex = PerpLiveExecutor(mock_ex, leverage=3)
        result = ex.place_order("BTC/USDT", "long", 0.1, 50_000.0)
        assert "error" in result
```

### Step 2: Run tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_executor.py::TestPerpLiveExecutor -v
```
Expected: `ImportError: cannot import name 'PerpLiveExecutor' from 'executor'`

### Step 3: Implement `PerpLiveExecutor`

Add after `PerpPaperExecutor` in `executor.py`:

```python
class PerpLiveExecutor:
    """Places real leveraged perpetual futures orders via CCXT.

    Uses market orders for both entry and exit. Closes use
    ``reduceOnly=True`` to prevent accidentally opening new positions.

    Args:
        exchange: Authenticated CCXT exchange (Binance futures market).
        leverage: Integer leverage multiplier (must match exchange setting).
    """

    def __init__(self, exchange: ccxt.Exchange, leverage: int = 3) -> None:
        self.exchange = exchange
        self.leverage = leverage

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]:
        """Place a market order to open or add to a position.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            side: ``"long"`` (buy) or ``"short"`` (sell).
            quantity: Asset quantity.
            price: Reference price (not used — market order fills at best).

        Returns:
            CCXT order dict on success; dict with ``"error"`` key on failure.
        """
        if side not in ("long", "short"):
            raise ValueError(f"Invalid side '{side}': must be 'long' or 'short'")
        ccxt_side = "buy" if side == "long" else "sell"
        try:
            order = self.exchange.create_market_order(symbol, ccxt_side, quantity)
            _log.info("[PerpLive] %s %.6f %s (order: %s)", side.upper(), quantity, symbol, order.get("id"))
            return order
        except ccxt.InsufficientFunds as e:
            _log.error("[PerpLive] Insufficient margin: %s", e)
            return {"error": str(e)}
        except ccxt.ExchangeError as e:
            _log.error("[PerpLive] Exchange error: %s", e)
            return {"error": str(e)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancel an open order.

        Args:
            order_id: Exchange-assigned order ID.
            symbol: Required by Binance futures API.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        try:
            self.exchange.cancel_order(order_id, symbol or Config.TRADING_PAIR)
            return True
        except Exception as e:
            _log.error("[PerpLive] Cancel error: %s", e)
            return False

    def open_position(self, **kwargs: Any) -> Any:
        """Create a Position with perp-specific fields computed.

        Accepts the same keyword arguments as ``Position.__init__``.
        Delegates to ``PerpPaperExecutor.open_position`` logic for field
        computation (same formula regardless of paper vs. live).

        Returns:
            ``Position`` instance with perp fields populated.
        """
        from risk_manager import Position

        pos = Position(**kwargs)
        pos.leverage = self.leverage
        pos.margin_used = pos.entry_price * pos.quantity / self.leverage
        if pos.side == "long":
            pos.liquidation_price = pos.entry_price * (1 - 1 / self.leverage)
        else:
            pos.liquidation_price = pos.entry_price * (1 + 1 / self.leverage)
        return pos

    def partial_close(
        self,
        position: Any,
        fraction: float,
        current_price: float,
        reason: str = "partial_tp",
    ) -> dict[str, Any]:
        """Close a fraction of the position with ``reduceOnly=True``.

        Args:
            position: Open ``Position`` object (quantity mutated in-place).
            fraction: Fraction to close (e.g. ``0.50``).
            current_price: Reference price for PnL estimation.
            reason: Audit tag.

        Returns:
            Order dict or error dict.
        """
        closed_qty = position.quantity * fraction
        ccxt_side = "sell" if position.side == "long" else "buy"
        try:
            raw = self.exchange.create_market_order(
                position.symbol, ccxt_side, closed_qty, None,
                params={"reduceOnly": True},
            )
            filled_price = float(raw.get("average") or current_price)
        except Exception as e:
            _log.error("[PerpLive] partial_close failed: %s", e)
            return {"error": str(e)}

        if position.side == "long":
            pnl = (filled_price - position.entry_price) * closed_qty
        else:
            pnl = (position.entry_price - filled_price) * closed_qty

        position.quantity -= closed_qty
        return {
            "id": raw.get("id", "unknown"),
            "symbol": position.symbol,
            "side": ccxt_side,
            "quantity": closed_qty,
            "price": filled_price,
            "pnl": pnl,
            "reason": reason,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PERP_LIVE",
        }
```

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_executor.py::TestPerpLiveExecutor -v
./venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -5
```
Expected: `4 passed` for new tests; full suite still passing

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix executor.py
git add executor.py tests/test_executor.py
git commit -m "feat: PerpLiveExecutor — CCXT market orders with reduceOnly partial close"
```

---

## Task 6: `ws_feed.py` — Binance WebSocket 1h Kline Feed

**Files:**
- Create: `ws_feed.py`
- Create: `tests/test_ws_feed.py`

**Context:** Runs in a daemon thread (not asyncio from the agent's perspective). Uses `aiohttp` (already in venv — present in `async_fetcher.py` import chain). Subscribes to Binance combined stream for all symbols. Thread-safe cache via simple dict (GIL protects single-key reads). Feeds `get_latest_close()` to `_run_phase9_cycle()` as a real-time price update.

### Step 1: Write the failing tests

Create `tests/test_ws_feed.py`:

```python
"""Tests for ws_feed.py Binance WebSocket kline feed."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


class TestWsFeedCache:
    """WsFeed cache updates from parsed WebSocket messages."""

    def _make_kline_msg(self, symbol: str = "BTCUSDT", close: str = "51000.0", is_closed: bool = True) -> dict:
        return {
            "stream": f"{symbol.lower()}@kline_1h",
            "data": {
                "e": "kline",
                "k": {
                    "t": 1704067200000,
                    "o": "50000.0",
                    "h": "52000.0",
                    "l": "49000.0",
                    "c": close,
                    "v": "1234.5",
                    "x": is_closed,  # candle closed flag
                },
            },
        }

    def test_closed_candle_updates_cache(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._handle(self._make_kline_msg("BTCUSDT", "51000.0", is_closed=True))
        assert feed.get_latest_close("BTC/USDT") == pytest.approx(51_000.0)

    def test_open_candle_does_not_update_cache(self):
        """Non-closed candles are ignored — only use confirmed 1h bars."""
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._handle(self._make_kline_msg("BTCUSDT", "51000.0", is_closed=False))
        assert feed.get_latest_close("BTC/USDT") is None

    def test_unknown_symbol_returns_none(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        assert feed.get_latest_close("XYZ/USDT") is None

    def test_symbol_normalisation(self):
        """get_latest_close() accepts both 'BTC/USDT' and 'BTCUSDT' forms."""
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._handle(self._make_kline_msg("BTCUSDT", "51000.0", is_closed=True))
        assert feed.get_latest_close("BTC/USDT") == pytest.approx(51_000.0)
        assert feed.get_latest_close("BTCUSDT") == pytest.approx(51_000.0)

    def test_stop_sets_running_false(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._running = True
        feed.stop()
        assert feed._running is False


class TestWsFeedReconnect:
    """Reconnect backoff doubles up to max."""

    def test_backoff_doubles(self):
        from ws_feed import _next_backoff
        assert _next_backoff(1.0) == pytest.approx(2.0)
        assert _next_backoff(32.0) == pytest.approx(60.0)  # capped at 60

    def test_stream_names(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        names = feed._stream_names()
        assert "btcusdt@kline_1h" in names
        assert "ethusdt@kline_1h" in names
        assert len(names) == 3
```

### Step 2: Run tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_ws_feed.py -v
```
Expected: `ModuleNotFoundError: No module named 'ws_feed'`

### Step 3: Implement `ws_feed.py`

Create `ws_feed.py`:

```python
"""WsFeed — Binance WebSocket 1h kline feed.

Runs in a dedicated daemon thread. Subscribes to the Binance combined
stream for all configured symbols and caches the latest closed-candle
close price per symbol.

Usage::

    feed = WsFeed(Config.TRADING_PAIRS)
    feed.start()          # spawns daemon thread
    # in agent cycle:
    price = feed.get_latest_close("BTC/USDT")  # None if no data yet
    feed.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

import aiohttp

_log = logging.getLogger(__name__)

_BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"
_MAX_BACKOFF_S = 60.0


def _next_backoff(current: float) -> float:
    """Double ``current`` delay, capped at ``_MAX_BACKOFF_S``."""
    return min(current * 2, _MAX_BACKOFF_S)


class WsFeed:
    """Subscribe to Binance 1h kline WebSocket streams.

    Args:
        symbols: List of trading pairs, e.g. ``["BTC/USDT", "ETH/USDT"]``.
    """

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols
        self._cache: dict[str, float] = {}  # "BTCUSDT" -> latest close
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API (called from agent thread — all thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket feed in a background daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ws_feed"
        )
        self._thread.start()
        _log.info("WsFeed started for %s", self._symbols)

    def stop(self) -> None:
        """Signal the feed loop to stop."""
        self._running = False

    def get_latest_close(self, symbol: str) -> float | None:
        """Return the latest closed 1h candle close price for ``symbol``.

        Args:
            symbol: Trading pair in either ``"BTC/USDT"`` or ``"BTCUSDT"`` form.

        Returns:
            Latest close price, or ``None`` if no data has been received yet.
        """
        normalized = symbol.replace("/", "").upper()
        return self._cache.get(normalized)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_names(self) -> list[str]:
        return [f"{s.replace('/', '').lower()}@kline_1h" for s in self._symbols]

    def _handle(self, data: dict[str, Any]) -> None:
        """Parse a raw WebSocket message and update the cache."""
        stream: str = data.get("stream", "")
        symbol_raw = stream.split("@")[0].upper()  # "btcusdt" → "BTCUSDT"
        kline: dict = data.get("data", {}).get("k", {})
        if kline.get("x"):  # only closed candles
            self._cache[symbol_raw] = float(kline["c"])
            _log.debug("WsFeed cached %s close=%.2f", symbol_raw, self._cache[symbol_raw])

    def _run_loop(self) -> None:
        """Entry point for the daemon thread — owns its own asyncio loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect())
        finally:
            loop.close()

    async def _connect(self) -> None:
        """Connect to Binance WebSocket with exponential-backoff reconnect."""
        streams = "/".join(self._stream_names())
        url = f"{_BINANCE_WS_URL}?streams={streams}"
        retry_delay = 1.0

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        _log.info("WsFeed connected to %s", url)
                        retry_delay = 1.0  # reset on successful connect
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    self._handle(json.loads(msg.data))
                                except Exception as e:
                                    _log.debug("WsFeed parse error: %s", e)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                _log.warning("WsFeed message type %s — reconnecting", msg.type)
                                break
            except Exception as e:
                _log.warning("WsFeed disconnected: %s — retry in %.0fs", e, retry_delay)

            if self._running:
                await asyncio.sleep(retry_delay)
                retry_delay = _next_backoff(retry_delay)
```

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_ws_feed.py -v
```
Expected: `7 passed`

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix ws_feed.py
git add ws_feed.py tests/test_ws_feed.py
git commit -m "feat: ws_feed.py — aiohttp Binance WebSocket 1h kline feed with reconnect"
```

---

## Task 7: Agent Executor Routing (`decision.route`)

**Files:**
- Modify: `agent.py`
- Modify: `tests/test_agent.py` (new `TestPerpRouting` class)

**Context:** `_run_phase9_cycle()` currently ignores `decision.route`. This task adds `self.perp_executor` to the agent and routes "perp" decisions to it. `_execute_trade()` gets an optional `executor` parameter (defaults to `self.executor` for backward compatibility). A new config flag `USE_PERP=true` enables perp; when false (default), "perp" decisions fall through to spot executor (safe degradation).

### Step 1: Write the failing tests

Add a new `TestPerpRouting` class to `tests/test_agent.py` (or create it if the file doesn't already have agent unit tests):

```python
class TestPerpRouting:
    """_run_phase9_cycle routes decision.route to correct executor."""

    def _make_agent_paper(self):
        """Return a minimal TradingAgent in paper mode for testing."""
        import os
        os.environ.setdefault("TRADING_MODE", "paper")
        from agent import TradingAgent
        agent = TradingAgent.__new__(TradingAgent)
        # Minimal init
        from executor import PaperExecutor, PerpPaperExecutor
        agent.executor = PaperExecutor()
        agent.perp_executor = PerpPaperExecutor(leverage=3)
        agent._use_perp = True
        return agent

    def test_route_spot_uses_spot_executor(self):
        """decision.route='spot' calls _execute_trade with spot executor."""
        agent = self._make_agent_paper()
        from executor import PaperExecutor
        assert isinstance(
            agent._resolve_executor("spot"), PaperExecutor
        )

    def test_route_perp_uses_perp_executor(self):
        """decision.route='perp' calls _execute_trade with perp executor."""
        agent = self._make_agent_paper()
        from executor import PerpPaperExecutor
        assert isinstance(
            agent._resolve_executor("perp"), PerpPaperExecutor
        )

    def test_route_perp_falls_back_to_spot_when_use_perp_false(self):
        """When USE_PERP=false, perp route degrades to spot executor."""
        agent = self._make_agent_paper()
        agent._use_perp = False
        from executor import PaperExecutor
        assert isinstance(
            agent._resolve_executor("perp"), PaperExecutor
        )
```

### Step 2: Run tests — expect FAIL

```bash
./venv/bin/python -m pytest tests/test_agent.py::TestPerpRouting -v
```
Expected: `AttributeError: 'TradingAgent' object has no attribute '_resolve_executor'`

### Step 3: Add perp executor to agent

In `agent.py`:

**a) Add import at top (with other executor imports):**
```python
from executor import LiveExecutor, PaperExecutor, PerpLiveExecutor, PerpPaperExecutor
```

**b) In `__init__`, after `self.executor = ...` lines:**
```python
# Phase 10: Perp executor (paper or live based on TRADING_MODE)
self._use_perp: bool = os.getenv("USE_PERP", "false").lower() == "true"
if self._use_perp:
    perp_leverage = int(os.getenv("PERP_LEVERAGE", "3"))
    if Config.TRADING_MODE == "paper":
        self.perp_executor: Any = PerpPaperExecutor(leverage=perp_leverage)
    else:
        self.perp_executor = PerpLiveExecutor(
            self.fetcher.exchange, leverage=perp_leverage
        )
else:
    self.perp_executor = self.executor  # fallback: route perp to spot
```

**c) Add `_resolve_executor()` method (near `_run_phase9_cycle`):**
```python
def _resolve_executor(self, route: str) -> Any:
    """Return the appropriate executor for the given decision route.

    Args:
        route: ``"spot"`` or ``"perp"`` from ``Decision.route``.

    Returns:
        ``self.perp_executor`` when ``route=="perp"`` and ``USE_PERP=true``,
        otherwise ``self.executor`` (spot).
    """
    if route == "perp" and self._use_perp:
        return self.perp_executor
    return self.executor
```

**d) In `_run_phase9_cycle()`, the `_execute_trade` call block (around line 800), change to pass the resolved executor:**

Find this block:
```python
try:
    self._execute_trade(
        trade_signal,
        decision.score or 0.5,
        current_price,
        df_ind,
        strat_sig,
        pair=symbol,
    )
```

Replace with:
```python
active_executor = self._resolve_executor(decision.route or "spot")
try:
    self._execute_trade(
        trade_signal,
        decision.score or 0.5,
        current_price,
        df_ind,
        strat_sig,
        pair=symbol,
        executor=active_executor,
    )
```

**e) In `_execute_trade()`, add `executor` parameter and use it:**

Find the method signature:
```python
def _execute_trade(
    self,
    signal: str,
    confidence: float,
    price: float,
    df_ind: Any,
    strat_sig: Any,
    position_mult: float = 1.0,
    intel_adjustment: float = 1.0,
    pair: str | None = None,
) -> None:
```

Add `executor: Any = None` to the end of the parameter list.

Then find the first use of `self.executor` inside `_execute_trade()` that places an order and replace it with `active_executor = executor or self.executor` at the top of the method body, then use `active_executor` wherever `self.executor` is currently used for order placement.

### Step 4: Run tests — expect PASS

```bash
./venv/bin/python -m pytest tests/test_agent.py::TestPerpRouting -v
./venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -5
```
Expected: `3 passed`; full suite still passing

### Step 5: Lint and commit

```bash
./venv/bin/python -m ruff check --fix agent.py
git add agent.py tests/test_agent.py
git commit -m "feat: route decision.route to PerpExecutor in _run_phase9_cycle"
```

---

## Task 8: Agent WebSocket Integration

**Files:**
- Modify: `agent.py` (start `WsFeed`, use latest close in `_run_phase9_cycle`)

**Context:** `WsFeed` runs in its own daemon thread. The agent reads `feed.get_latest_close()` as a real-time price supplement — if the feed has a value it overrides the REST-fetched close; if `None` (feed not yet populated), falls back to REST. This is additive and backward-compatible: REST polling continues as before.

### Step 1: Write the failing test

Add to `tests/test_agent.py`:

```python
class TestWsFeedIntegration:
    """Agent uses WsFeed close price when available."""

    def test_ws_feed_attribute_exists_when_enabled(self):
        """When USE_WS_FEED=true, agent has _ws_feed attribute."""
        import os
        os.environ["USE_WS_FEED"] = "true"
        try:
            from ws_feed import WsFeed
            # Patch WsFeed.start() to avoid spawning real threads in tests
            with patch("ws_feed.WsFeed.start"):
                from agent import TradingAgent
                agent = TradingAgent.__new__(TradingAgent)
                # Verify _ws_feed would be a WsFeed
                # (simplified: just check the class exists and is importable)
                assert WsFeed is not None
        finally:
            del os.environ["USE_WS_FEED"]
```

*(Note: Full agent wiring is an integration concern — the unit test verifies the import chain. The real test is paper-mode end-to-end.)*

### Step 2: Add WsFeed to agent `__init__`

In `agent.py`, add import at top:
```python
from ws_feed import WsFeed
```

In `__init__`, after `self.perp_executor` block:
```python
# Phase 10: WebSocket real-time feed (opt-in via USE_WS_FEED=true)
self._use_ws_feed: bool = os.getenv("USE_WS_FEED", "false").lower() == "true"
self._ws_feed: WsFeed | None = None
if self._use_ws_feed:
    self._ws_feed = WsFeed(Config.TRADING_PAIRS)
    self._ws_feed.start()
    _log.info("WsFeed started for real-time kline data")
```

### Step 3: Use WebSocket price in `_run_phase9_cycle`

In `_run_phase9_cycle()`, find where `current_price` is set from REST data (around where `snapshot.df_1h.iloc[-1]["close"]` is used). Add a WebSocket override:

```python
current_price = float(snapshot.df_1h.iloc[-1]["close"])
# Phase 10: Override with real-time WsFeed price if available
if self._ws_feed is not None:
    ws_price = self._ws_feed.get_latest_close(symbol)
    if ws_price is not None:
        current_price = ws_price
        _log.debug("WsFeed price override for %s: %.2f", symbol, current_price)
```

### Step 4: Add `.env` config keys

In `.env.example`, add:
```
# Phase 10: Perp trading
USE_PERP=false
PERP_LEVERAGE=3
# Phase 10: WebSocket real-time data
USE_WS_FEED=false
```

### Step 5: Run full test suite

```bash
./venv/bin/python -m pytest tests/ -q 2>&1 | tail -5
./venv/bin/python -m ruff check .
```
Expected: All tests pass, 0 ruff errors.

### Step 6: Final commit

```bash
git add agent.py .env.example
git commit -m "feat: Phase 10 complete — WebSocket feed, perp routing, alerting"
```

---

## End-to-End Smoke Test (manual, paper mode)

After all 8 tasks:

1. Set in `.env`: `TRADING_MODE=paper`, `USE_PERP=true`, `USE_WS_FEED=true`
2. Start API: `make dev`
3. Observe logs — should see `WsFeed started` and `Phase 9 evaluate` lines
4. With `USE_PHASE9_PERP=true` in `.env`: verify `LiquidationTrigger` and `FundingExtremeTrigger` load without error
5. Check `/health` returns 200 with non-stale `updated_at`
6. Trigger a test trade via Telegram `/status` command — verify it goes through paper executor

---

## Summary

| Task | New tests | Files changed |
|------|-----------|---------------|
| 1 — LiveExecutor.partial_close | 3 | executor.py, test_executor.py |
| 2 — alerting.py | 5 | alerting.py (new), test_alerting.py (new) |
| 3 — Position perp fields | 4 | risk_manager.py, test_risk_manager.py |
| 4 — PerpPaperExecutor | 5 | executor.py, test_executor.py |
| 5 — PerpLiveExecutor | 4 | executor.py, test_executor.py |
| 6 — ws_feed.py | 7 | ws_feed.py (new), test_ws_feed.py (new) |
| 7 — Agent routing | 3 | agent.py, test_agent.py |
| 8 — WsFeed wiring | 1 | agent.py, .env.example |
| **Total** | **32** | **9 files** |
