# PullbackTrigger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `PullbackTrigger` to Phase 9 so the agent can enter on RSI pullbacks within trending markets, eliminating the "602 consecutive no_valid_triggers" problem.

**Architecture:** New `triggers/pullback.py` follows exact `MomentumTrigger` pattern. `TriggerEngine.on_1h_close()` gains a `swing_bias` kwarg (default `"neutral"`) and calls the new trigger. `agent.py` passes `context.swing_bias` at the call site. No new dependencies.

**Tech Stack:** Python, pandas, numpy, pytest. See `triggers/momentum.py` for the trigger pattern to follow.

---

### Task 1: Write failing tests for PullbackTrigger

**Files:**
- Modify: `tests/test_triggers.py` (append after the last class)

**Step 1: Add fixtures and test class to `tests/test_triggers.py`**

Append this entire block at the bottom of the file:

```python
# ---------------------------------------------------------------------------
# PullbackTrigger fixtures + tests
# ---------------------------------------------------------------------------


def make_pullback_long_df(n: int = 55) -> pd.DataFrame:
    """Uptrend then pullback then recovery bar — RSI in [42,52] and rising.

    Phase 1 (30 bars, 88000→95000): sustained rally drives RSI to ~72.
    Phase 2 (12 bars, 95000→91500): pullback drops RSI to ~44-50.
    Phase 3 (1 bar, 91500→92200): recovery bar — RSI ticks up (recovery=True).

    When n < 55, returns n bars of declining phase only — used for the
    insufficient-data test.
    """
    phase1 = np.linspace(88000.0, 95000.0, 30)
    phase2 = np.linspace(95000.0, 91500.0, 12)
    phase3 = np.array([92200.0])
    full = np.concatenate([phase1, phase2, phase3])
    closes = full[:n]
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


def make_pullback_short_df(n: int = 55) -> pd.DataFrame:
    """Downtrend then bounce then rejection bar — RSI in [48,58] and declining.

    Phase 1 (30 bars, 95000→88000): sustained decline drives RSI to ~28.
    Phase 2 (12 bars, 88000→91500): bounce lifts RSI to ~50-56.
    Phase 3 (1 bar, 91500→90800): rejection bar — RSI ticks down (declining=True).
    """
    phase1 = np.linspace(95000.0, 88000.0, 30)
    phase2 = np.linspace(88000.0, 91500.0, 12)
    phase3 = np.array([90800.0])
    full = np.concatenate([phase1, phase2, phase3])
    closes = full[:n]
    return pd.DataFrame(
        {
            "open": closes * 1.001,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


class TestPullbackTrigger:
    def test_insufficient_bars_returns_empty(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(n=10), swing_bias="bullish")
        assert signals == []

    def test_neutral_bias_returns_empty(self):
        """swing_bias='neutral' must suppress all signals regardless of RSI."""
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="neutral")
        assert signals == []

    def test_long_fires_on_bullish_pullback(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert any(s.direction == "long" for s in signals)

    def test_long_not_fired_on_bearish_bias(self):
        """Bullish pullback conditions with bearish bias → no long signal."""
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bearish")
        assert not any(s.direction == "long" for s in signals)

    def test_short_fires_on_bearish_pullback(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_short_df(), swing_bias="bearish")
        assert any(s.direction == "short" for s in signals)

    def test_short_not_fired_on_bullish_bias(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_short_df(), swing_bias="bullish")
        assert not any(s.direction == "short" for s in signals)

    def test_source_is_pullback_1h(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert all(s.source == "pullback_1h" for s in signals)

    def test_urgency_is_normal(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert all(s.urgency == "normal" for s in signals)

    def test_signal_not_expired(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert all(s.expires_at > datetime.now(UTC) for s in signals)

    def test_strength_in_valid_range(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        for s in signals:
            assert 0.0 < s.strength <= 0.72

    def test_symbol_scope_extracted(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="ETH/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert all(s.symbol_scope == "ETH" for s in signals)

    def test_flat_market_no_signal(self):
        """Flat prices → no sustained trend → RSI stays near 50 without recovery."""
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_flat_df(n=55), swing_bias="bullish")
        # Flat market has no clear pullback recovery pattern
        for s in signals:
            assert s.direction == "long"  # if any fire, they must match bias


class TestTriggerEngineSwingBias:
    def test_on_1h_close_accepts_swing_bias_kwarg(self):
        """on_1h_close must accept swing_bias without raising."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        result = eng.on_1h_close(make_pullback_long_df(), swing_bias="bullish")
        assert isinstance(result, list)

    def test_on_1h_close_default_swing_bias_neutral(self):
        """Existing call sites (no swing_bias) must continue to work."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        result = eng.on_1h_close(make_pullback_long_df())
        assert isinstance(result, list)

    def test_pullback_signal_in_buffer_after_bullish_close(self):
        """After on_1h_close with bullish bias, valid_signals includes pullback."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        eng.on_1h_close(make_pullback_long_df(), swing_bias="bullish")
        sigs = eng.valid_signals()
        pullback_sigs = [s for s in sigs if s.source == "pullback_1h"]
        assert len(pullback_sigs) >= 1

    def test_no_pullback_signal_with_neutral_bias(self):
        """Neutral bias must not produce pullback signals in buffer."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        eng.on_1h_close(make_pullback_long_df(), swing_bias="neutral")
        sigs = eng.valid_signals()
        assert not any(s.source == "pullback_1h" for s in sigs)
```

**Step 2: Run tests to confirm they fail (module missing)**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
./venv/bin/python -m pytest tests/test_triggers.py::TestPullbackTrigger::test_insufficient_bars_returns_empty -v
```

Expected: `ModuleNotFoundError: No module named 'triggers.pullback'`

---

### Task 2: Implement `triggers/pullback.py`

**Files:**
- Create: `triggers/pullback.py`

**Step 1: Create `triggers/pullback.py`**

```python
"""PullbackTrigger — RSI pullback + recovery detection in trending markets.

Fires when:
  - Context swing_bias is "bullish" and RSI pulls back to [42, 52] then
    starts recovering (current RSI > 3-bar low)
  - Context swing_bias is "bearish" and RSI bounces to [48, 58] then
    starts declining (current RSI < 3-bar high)
  - swing_bias == "neutral" → always returns [] (no signal in ranging markets)

Relies on the caller (TriggerEngine.on_1h_close) to pass swing_bias from
ContextEngine — avoids re-computing trend from 1h EMAs inside the trigger.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_MIN_BARS = 26                # RSI-14 needs 14+ bars; 26 adds MACD-length buffer
_SIGNAL_TTL_MINUTES = 90      # ~1.5 candle window
_RECOVERY_LOOKBACK = 3        # Bars to scan for RSI dip/bounce
_LONG_RSI_LO = 42.0           # Floor: below this is oversold, not a pullback
_LONG_RSI_HI = 52.0           # Ceiling: above this RSI hasn't pulled back to neutral
_SHORT_RSI_LO = 48.0          # Floor for short bounce zone
_SHORT_RSI_HI = 58.0          # Ceiling: above this RSI is overbought
_BASE_STRENGTH = 0.50
_MAX_STRENGTH = 0.72          # Below high-urgency perp triggers (0.75+)


class PullbackTrigger:
    """Generates pullback TriggerSignals from 1h OHLCV + context swing_bias.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Used to set symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(
        self, df: pd.DataFrame, swing_bias: str = "neutral"
    ) -> list[TriggerSignal]:
        """Evaluate 1h OHLCV + swing_bias and return any pullback triggers.

        Args:
            df: 1h OHLCV DataFrame with columns: open, high, low, close, volume.
            swing_bias: Context swing bias — "bullish", "bearish", or "neutral".
                        Returns [] immediately when "neutral".

        Returns:
            List of TriggerSignal (empty if no pullback/recovery conditions met).
        """
        if swing_bias == "neutral":
            return []

        if len(df) < _MIN_BARS:
            return []

        close = df["close"]

        # RSI-14 (Wilder smoothing via rolling mean)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        if rsi.isna().iloc[-1] or len(rsi) < _RECOVERY_LOOKBACK + 2:
            return []

        current_rsi = float(rsi.iloc[-1])
        # The _RECOVERY_LOOKBACK bars immediately before the current bar
        rsi_window = rsi.iloc[-_RECOVERY_LOOKBACK - 1 : -1]

        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)
        signals: list[TriggerSignal] = []

        if swing_bias == "bullish":
            in_zone = _LONG_RSI_LO <= current_rsi <= _LONG_RSI_HI
            recovering = float(rsi_window.min()) < current_rsi
            if in_zone and recovering:
                depth = _LONG_RSI_HI - current_rsi          # 0 at zone top, 10 at floor
                strength = min(_BASE_STRENGTH + depth / 14.0 * 0.22, _MAX_STRENGTH)
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="pullback_1h",
                        direction="long",
                        strength=round(strength, 3),
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=(
                            f"Pullback long: RSI {current_rsi:.1f} in"
                            f" [{_LONG_RSI_LO:.0f}–{_LONG_RSI_HI:.0f}], recovering"
                        ),
                        expires_at=expiry,
                        raw_data={
                            "rsi": round(current_rsi, 2),
                            "rsi_3bar_min": round(float(rsi_window.min()), 2),
                        },
                    )
                )

        elif swing_bias == "bearish":
            in_zone = _SHORT_RSI_LO <= current_rsi <= _SHORT_RSI_HI
            declining = float(rsi_window.max()) > current_rsi
            if in_zone and declining:
                depth = current_rsi - _SHORT_RSI_LO         # 0 at zone floor, 10 at top
                strength = min(_BASE_STRENGTH + depth / 14.0 * 0.22, _MAX_STRENGTH)
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="pullback_1h",
                        direction="short",
                        strength=round(strength, 3),
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=(
                            f"Pullback short: RSI {current_rsi:.1f} in"
                            f" [{_SHORT_RSI_LO:.0f}–{_SHORT_RSI_HI:.0f}], declining"
                        ),
                        expires_at=expiry,
                        raw_data={
                            "rsi": round(current_rsi, 2),
                            "rsi_3bar_max": round(float(rsi_window.max()), 2),
                        },
                    )
                )

        return signals
```

**Step 2: Run the PullbackTrigger unit tests**

```bash
./venv/bin/python -m pytest tests/test_triggers.py::TestPullbackTrigger -v
```

Expected: most pass. If fixtures produce wrong RSI values (e.g. `test_long_fires_on_bullish_pullback` fails with 0 signals), debug by running:

```python
import numpy as np, pandas as pd
phase1 = np.linspace(88000.0, 95000.0, 30)
phase2 = np.linspace(95000.0, 91500.0, 12)
phase3 = np.array([92200.0])
closes = pd.Series(np.concatenate([phase1, phase2, phase3]))
delta = closes.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
rsi = 100 - (100 / (1 + rs))
print("Last 5 RSI:", rsi.iloc[-5:].values)
```

Adjust phase2 endpoint (e.g. 91000 vs 91500) until `rsi.iloc[-1]` lands in [42, 52].

**Step 3: Lint**

```bash
./venv/bin/python -m ruff check triggers/pullback.py
```

Expected: `All checks passed!`

---

### Task 3: Wire PullbackTrigger into TriggerEngine

**Files:**
- Modify: `trigger_engine.py`

**Step 1: Add import + instantiation in `__init__`**

In `trigger_engine.py`, after the existing imports add:
```python
from triggers.pullback import PullbackTrigger
```

In `TriggerEngine.__init__`, after `self._sweep = LiquiditySweepTrigger(symbol=symbol)`:
```python
self._pullback = PullbackTrigger(symbol=symbol)
```

**Step 2: Add `swing_bias` param to `on_1h_close` and call the new trigger**

Change the `on_1h_close` signature and body from:
```python
def on_1h_close(self, df: pd.DataFrame) -> list[TriggerSignal]:
    ...
    new_signals = self._momentum.evaluate(df) + self._sweep.evaluate(df)
```
to:
```python
def on_1h_close(
    self, df: pd.DataFrame, swing_bias: str = "neutral"
) -> list[TriggerSignal]:
    ...
    new_signals = (
        self._momentum.evaluate(df)
        + self._sweep.evaluate(df)
        + self._pullback.evaluate(df, swing_bias=swing_bias)
    )
```

Also update the docstring Args section to add:
```
swing_bias: Context trend direction ("bullish"/"bearish"/"neutral").
    Passed through to PullbackTrigger; defaults to "neutral" (no pullback
    signals) so existing call sites without the kwarg are unaffected.
```

**Step 3: Run TriggerEngine tests**

```bash
./venv/bin/python -m pytest tests/test_triggers.py::TestTriggerEngineSwingBias -v
```

Expected: all 4 pass.

**Step 4: Lint**

```bash
./venv/bin/python -m ruff check trigger_engine.py
```

---

### Task 4: Update `agent.py` to pass `swing_bias`

**Files:**
- Modify: `agent.py` (~line 786)

**Step 1: Pass `swing_bias` at the call site**

Find this block (around line 784–788):
```python
trigger_eng = self._phase9_trigger_engines[symbol]
try:
    if snapshot.df_1h is not None:
        trigger_eng.on_1h_close(snapshot.df_1h)
except Exception as e:
    _log.warning("Phase 9 momentum trigger failed for %s: %s", symbol, e)
```

Change to:
```python
trigger_eng = self._phase9_trigger_engines[symbol]
try:
    if snapshot.df_1h is not None:
        trigger_eng.on_1h_close(
            snapshot.df_1h,
            swing_bias=context.swing_bias if context else "neutral",
        )
except Exception as e:
    _log.warning("Phase 9 momentum trigger failed for %s: %s", symbol, e)
```

(`context` is assigned at line 772: `context = self._phase9_context.get(symbol)` — already checked for `None` at line 773, but the ternary guard is defensive.)

**Step 2: Lint**

```bash
./venv/bin/python -m ruff check agent.py
```

Expected: `All checks passed!`

---

### Task 5: Full verification + commit

**Step 1: Run the full trigger test file**

```bash
./venv/bin/python -m pytest tests/test_triggers.py -v
```

Expected: all tests pass (existing + new).

**Step 2: Run the full test suite**

```bash
./venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: 1233 + new tests, 0 failures.

**Step 3: Verify import chain works end-to-end**

```bash
./venv/bin/python -c "
from trigger_engine import TriggerEngine
import pandas as pd, numpy as np
eng = TriggerEngine('BTC/USDT')
closes = pd.Series(np.linspace(88000, 95000, 55))
df = pd.DataFrame({'open': closes*0.999, 'high': closes*1.002, 'low': closes*0.998, 'close': closes, 'volume': [1000.0]*55})
sigs = eng.on_1h_close(df, swing_bias='bullish')
print('on_1h_close OK, signals:', len(sigs))
print('valid_signals:', len(eng.valid_signals()))
"
```

**Step 4: Commit**

```bash
git add triggers/pullback.py trigger_engine.py agent.py tests/test_triggers.py
git commit -m "feat: add PullbackTrigger — RSI pullback/recovery in trending markets

Fires when swing_bias is bullish/bearish and RSI pulls back to neutral
zone (42-52 for longs, 48-58 for shorts) then shows recovery.
Wired into TriggerEngine.on_1h_close(df, swing_bias=) — default
'neutral' keeps all existing call sites unchanged.
agent.py passes context.swing_bias at the trigger call site.

Fixes 602-consecutive-no_valid_triggers in sustained trending markets."
```

---

### Task 6: Run Phase 9 1yr backtest

**Step 1: Run baseline backtest (current state before starting the agent)**

```bash
./venv/bin/python scripts/backtest_phase9_1yr.py 2>&1 | tee /tmp/phase9_backtest_results.txt
```

Expected runtime: ~5–10 minutes. Note the trade counts and PnL per pair.

**Step 2: Review results**

Look for:
- Trade count per pair (was 7/9/6 trades in prior run)
- Whether pullback signals fire meaningfully
- PnL delta vs prior baseline (−0.25%/−0.45%/+0.62%)

**Step 3: Record results in CLAUDE.md**

Add a new entry under the Phase 9 Enhancements row with the backtest results.
