# Design: Crypto Checklist Gap Implementation
**Date:** 2026-03-04
**Status:** Approved
**Branch target:** main

---

## Overview

Phase 9 is the primary execution pipeline. This design closes seven gaps identified against
a crypto-specific trading checklist. All changes are additive to Phase 9 — Phase 8 code
(strategies.py, decision_engine.py, backtester) is untouched.

The seven gaps, grouped by layer:

| # | Gap | Layer |
|---|-----|-------|
| 1 | `volatility_regime` hardcoded "normal" | Context |
| 2 | No session/weekend liquidity penalty | Context |
| 3 | `prev_day_high`/`prev_day_low` absent from key_levels | Context |
| 4 | Funding extreme not blocking trades | Decision |
| 5 | No BTC dominance gate for SOL | Decision (agent.py) |
| 6 | No liquidity sweep entry trigger | Trigger |
| 7 | No partial TP at structural levels | Risk / Executor |

---

## Section 1: Context Layer

### 1a. VolatilityAnalyzer (`context/volatility.py`)

Computes 14-bar ATR on the 1h DataFrame, expresses as % of close, compares to a 30-bar
rolling mean of ATR%, and emits the existing `VolatilityRegime` literal.

| ATR% vs rolling mean | Regime |
|---|---|
| < 50% | `low` |
| 50–150% | `normal` |
| 150–250% | `elevated` |
| > 250% | `extreme` |

**Wiring:** `ContextEngine.build()` receives `snapshot.df_1h` (already available via
`MultiTimeframeFetcher`). `VolatilityAnalyzer.analyze(df_1h)` result replaces the
hardcoded `"normal"` on `context_engine.py:99`.

**Downstream:** `ContextState.volatility_regime` is now live. `evaluate()` does not
immediately gate on it, but `RiskSupervisor` can in future. The field is logged to
`phase9_decisions.jsonl` via `DecisionLogger`.

---

### 1b. SessionAnalyzer (`context/session.py`)

Maps UTC datetime to a session label and confidence multiplier. Applied inside
`ContextEngine.build()` by scaling `swing["confidence"]` before it is stored.

| UTC window | Session | Multiplier |
|---|---|---|
| 13:00–22:00 | US | 1.00 |
| 07:00–15:00 | EU | 0.90 |
| 00:00–08:00 | Asia | 0.75 |
| Sat/Sun (any) | Weekend | 0.60 |
| Overlapping windows | max of applicable | — |

If `confidence × multiplier < 0.30`, `ContextEngine.build()` sets `tradeable = False`.
This prevents entries during thin Asian / weekend sessions without a dedicated gate.

---

### 1c. `prev_day_high` / `prev_day_low` in key_levels

`SwingAnalyzer.analyze()` already receives the 4h DataFrame. Group the DataFrame by UTC
date, take yesterday's row: `high.max()` → `pdh`, `low.min()` → `pdl`.

These are added to the returned `key_levels` dict alongside existing `support`,
`resistance`, and `poc`. No `ContextState` schema change — `key_levels: dict` is
already untyped.

**Edge case:** If fewer than two distinct UTC dates exist in `df`, skip and omit `pdh`/`pdl`
from the dict. Consumers check `key_levels.get("pdh")` defensively.

---

## Section 2: Decision Layer

### 2a. Funding Extreme Gate (`decision.py`)

Inserted at **Step 1.5** in `evaluate()` — after the `tradeable` / `allowed_directions`
hard gate, before trigger filtering:

```python
# Funding extreme: block trading the crowded side
if context.funding_pressure == "long_crowded_extreme":
    allowed = [d for d in context.allowed_directions if d != "long"]
    if not allowed:
        return Decision(action="reject", reason="funding_extreme_blocks_direction")
if context.funding_pressure == "short_crowded_extreme":
    allowed = [d for d in context.allowed_directions if d != "short"]
    if not allowed:
        return Decision(action="reject", reason="funding_extreme_blocks_direction")
```

`mild` crowding is not a hard block. `evaluate()` continues using the modified `allowed`
list for trigger filtering in Step 2.

---

### 2b. BTC Dominance Gate (`agent.py`)

One new attribute on `TradingAgent`:

```python
self._btc_swing_bias: str = "neutral"  # updated each time BTC completes context build
```

In `_run_phase9_cycle()`, after building BTC context, store:
```python
if symbol == "BTC/USDT":
    self._btc_swing_bias = context.swing_bias
```

For SOL entries only, before calling `evaluate()`:
```python
if symbol == "SOL/USDT":
    if self._btc_swing_bias == "bearish" and trigger.direction == "long":
        # log btc_dominance_block, skip evaluate()
        continue
    if self._btc_swing_bias == "bullish" and trigger.direction == "short":
        continue
```

ETH is not gated — large-cap divergence is valid. BTC is never gated by itself.

---

## Section 3: LiquiditySweepTrigger (`triggers/liquidity_sweep.py`)

### Detection

On each 1h bar, over the last 20 bars:

1. **Equal highs** — find bars whose `high` is within 0.3% of another bar's `high`
   (min 2 instances). The zone maximum is the highest of the cluster.
2. **Equal lows** — mirror logic on `low`.
3. **Sweep detection:**
   - Long (bullish sweep): `bar.low < equal_lows_zone AND bar.close > equal_lows_zone`
   - Short (bearish sweep): `bar.high > equal_highs_zone AND bar.close < equal_highs_zone`

### Signal

```python
TriggerSignal(
    source="liquidity_sweep",
    urgency="normal",
    strength=0.65,          # fixed — sweeps are high-confidence by nature
    expires_at=now + 75min, # matches MomentumTrigger TTL
)
```

A sweep signal alone cannot pass `evaluate()` — consensus requires ≥2 agreeing triggers.
`LiquiditySweepTrigger` is designed to pair with `MomentumTrigger` or `OrderFlowTrigger`.

### Wiring

`TriggerEngine` instantiates `LiquiditySweepTrigger` per-symbol alongside existing
triggers. It receives `snapshot.df_1h` in `_run_phase9_cycle()`.

---

## Section 4: Partial TP at Structural Levels

### 4a. `Position` dataclass (`risk_manager.py`)

Two new fields:
```python
partial_tp_levels: list[float] = field(default_factory=list)
partial_exits: list[dict] = field(default_factory=list)
```

`partial_tp_levels` is populated at entry time in `RiskManager.open_position()` from
`key_levels`. Algorithm:

1. For longs: candidates = `[resistance, pdh]` above entry where candidate > entry + (entry - stop_loss)
   (i.e., at least 1:1 R:R from entry). Take up to 2 nearest.
2. For shorts: candidates = `[support, pdl]` below entry. Mirror logic.
3. Levels closer than 0.5× SL distance are skipped (noise).

### 4b. `RiskManager.check_partial_tp()` (`risk_manager.py`)

```python
def check_partial_tp(
    self, position: Position, current_price: float, key_levels: dict
) -> float | None:
    """Return exit fraction (0.50) if next partial level is reached, else None."""
```

- Iterates `position.partial_tp_levels`, skipping levels already in `partial_exits`
- For long: triggers if `current_price >= level`
- For short: triggers if `current_price <= level`
- Returns `0.50` (50% scale-out at each level)
- After triggering, appends to `position.partial_exits` to prevent re-fire

### 4c. `Executor.partial_close()` (`executor.py`)

```python
def partial_close(
    self, position: Position, fraction: float, current_price: float, reason: str
) -> None:
```

- **Paper mode:** reduces `position.quantity *= (1 - fraction)`, calculates partial PnL,
  records to `TradeDB` with `exit_reason=reason`, `status="partial"`
- **Live mode:** submits reduce-only market order for `quantity × fraction`

### 4d. `agent.py` — `_manage_exits()` integration

In the exit management section of `_run_pair_cycle()`, after trailing stop update and
before full TP check:

```python
if self._phase9_context.get(symbol):
    key_levels = self._phase9_context[symbol].key_levels
    fraction = self.risk_manager.check_partial_tp(position, price, key_levels)
    if fraction:
        self.executor.partial_close(position, fraction, price, "partial_tp")
```

`_phase9_context` is a new per-symbol dict storing the latest `ContextState`, populated
in `_run_phase9_cycle()`.

### 4e. `TradeDB` — no schema change

Partial exits use the existing schema with:
- `exit_reason = "partial_tp_1"` / `"partial_tp_2"`
- `status = "partial"` for the parent while open with reduced size
- `status = "closed"` only when the final portion exits

---

## Files Changed

| File | Change type |
|------|-------------|
| `context/volatility.py` | **New** |
| `context/session.py` | **New** |
| `triggers/liquidity_sweep.py` | **New** |
| `context/swing.py` | Modified — add pdh/pdl to key_levels |
| `context_engine.py` | Modified — wire VolatilityAnalyzer + SessionAnalyzer |
| `decision.py` | Modified — funding extreme gate in evaluate() |
| `risk_manager.py` | Modified — Position fields + check_partial_tp() |
| `executor.py` | Modified — partial_close() |
| `agent.py` | Modified — BTC bias attribute, partial TP in exits, _phase9_context dict |
| `trigger_engine.py` | Modified — instantiate LiquiditySweepTrigger per-symbol |

---

## Test Coverage Required

- `tests/test_context_volatility.py` — VolatilityAnalyzer (low/normal/elevated/extreme)
- `tests/test_context_session.py` — SessionAnalyzer (US/EU/Asia/Weekend multipliers, tradeable cutoff)
- `tests/test_swing.py` — pdh/pdl in key_levels (sufficient/insufficient date history)
- `tests/test_decision.py` — funding extreme gate (direction-specific rejection)
- `tests/test_triggers.py` — LiquiditySweepTrigger (sweep long, sweep short, no sweep)
- `tests/test_risk_manager.py` — check_partial_tp (trigger, skip used, 1:1 filter)
- `tests/test_executor.py` — partial_close (paper mode quantity reduction + TradeDB record)

---

## Constraints

- Phase 8 code (strategies.py, decision_engine.py, backtester) untouched
- 0 ruff errors (run `ruff check --fix` after each file)
- All existing 988 tests must continue to pass
- New test files follow existing pytest patterns (fixtures in conftest.py where needed)
