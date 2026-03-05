# PullbackTrigger Design

**Date:** 2026-03-05
**Status:** Approved
**Context:** Phase 9 logged 602 consecutive `no_valid_triggers` rejects during a sustained bullish trend. `MomentumTrigger` only fires on RSI/MACD zero-crossings (trend reversals). In a trending market the entry opportunity is a pullback within the trend, not a reversal — this trigger fills that gap.

---

## Problem

`MomentumTrigger` requires RSI to cross 50 from below (long) or above (short). In a sustained uptrend, RSI stays above 50 continuously — no crossings occur, no triggers fire, no trades execute despite a healthy bullish context.

---

## Solution

`PullbackTrigger` — detects RSI pulling back to the neutral zone (42–52) within a bullish context, then recovering. Direction is gated by `swing_bias` from `ContextEngine` (4h EMA-based), avoiding any 1h EMA trend recomputation inside the trigger.

---

## Signal Logic

### Long signal
- `swing_bias == "bullish"` (passed from context)
- RSI in [42, 52] — pulled back to neutral zone, not oversold
- `current_rsi > min(rsi[-3:-1])` — RSI is recovering (higher than 3-bar low)

### Short signal
- `swing_bias == "bearish"`
- RSI in [48, 58] — bounced into neutral zone, not overbought
- `current_rsi < max(rsi[-3:-1])` — RSI is declining again (lower than 3-bar high)

### Ranging / neutral market
- `swing_bias == "neutral"` → returns `[]` immediately (no computation)

### Strength scaling
- Long: `strength = 0.50 + (52 − rsi) / 14 × 0.22`, capped at 0.72
- Short: `strength = 0.50 + (rsi − 48) / 14 × 0.22`, capped at 0.72
- Deeper pullback = stronger conviction. Cap at 0.72 keeps pullback signals below the 0.75 threshold used by high-urgency perp triggers.

### Parameters
| Constant | Value | Rationale |
|----------|-------|-----------|
| `_MIN_BARS` | 26 | Same as MomentumTrigger (RSI needs 14+ bars) |
| `_SIGNAL_TTL_MINUTES` | 90 | ~1.5 candles; pullback entries act quickly |
| `_LONG_RSI_LO` | 42.0 | Floor — below this is oversold, not a pullback |
| `_LONG_RSI_HI` | 52.0 | Ceiling — above this, RSI hasn't pulled back |
| `_SHORT_RSI_LO` | 48.0 | Floor — below this, RSI hasn't bounced |
| `_SHORT_RSI_HI` | 58.0 | Ceiling — above this is overbought |
| `_RECOVERY_LOOKBACK` | 3 | Bars to look back for RSI dip/bounce |
| `_BASE_STRENGTH` | 0.50 | Minimum strength |
| `_MAX_STRENGTH` | 0.72 | Cap — below high-urgency perp triggers (0.75+) |

---

## Architecture

```
triggers/pullback.py          ← new file (~110 lines)
trigger_engine.py             ← wire in PullbackTrigger + swing_bias param
agent.py                      ← pass ctx.swing_bias to on_1h_close()
tests/test_triggers.py        ← new fixtures + TestPullbackTrigger class
```

### Data flow

```
agent.py: _run_phase9_cycle()
  ctx = context_engine.build(...)           # ctx.swing_bias = "bullish" / "bearish" / "neutral"
  trigger_eng.on_1h_close(
      df_1h, swing_bias=ctx.swing_bias)     # NEW kwarg
    ├── self._pullback.evaluate(df, bias)   # NEW
    ├── self._momentum.evaluate(df)         # unchanged
    └── self._sweep.evaluate(df)            # unchanged
  triggers = trigger_eng.valid_signals()
  decision = evaluate(ctx, triggers)
```

### Backward compatibility
`on_1h_close(df, swing_bias="neutral")` defaults to `"neutral"` — all existing call sites and tests that don't pass `swing_bias` continue to work unchanged (pullback trigger simply returns `[]`).

---

## Testing Plan

`tests/test_triggers.py` — new `TestPullbackTrigger` class:

1. `test_insufficient_bars` — `len(df) < 26` → `[]`
2. `test_neutral_bias_no_signal` — perfect pullback conditions but `swing_bias="neutral"` → `[]`
3. `test_long_fires_in_bullish_pullback` — bullish + RSI ~46 + recovering → 1 long signal
4. `test_long_rsi_above_zone` — RSI=55 → `[]`
5. `test_long_rsi_below_floor` — RSI=35 → `[]`
6. `test_long_rsi_not_recovering` — RSI in zone but still falling → `[]`
7. `test_short_fires_in_bearish_pullback` — bearish + RSI ~54 + declining → 1 short signal
8. `test_strength_scales_with_depth` — RSI=43 produces higher strength than RSI=50
9. `test_trigger_engine_passes_bias` — `TriggerEngine.on_1h_close(df, swing_bias="bullish")` routes to pullback trigger
10. `test_trigger_engine_default_bias_no_pullback` — default `swing_bias="neutral"` → no pullback signals

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Trend detection | Trust `swing_bias` from context | 4h EMA-based, higher quality than 1h EMA; avoids duplicating logic |
| Entry timing | Recovery required (RSI rising from 3-bar low) | Avoids catching falling knives |
| RSI zone | 42–52 long / 48–58 short | Neutral zone; shallower zones would fire in overbought/oversold |
| Strength | Depth-based | Deeper pullback = stronger recovery potential |
| Ranging market | `swing_bias="neutral"` → no signal | Context already handles ranging correctly |
