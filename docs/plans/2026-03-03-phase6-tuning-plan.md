# Phase 6 Tuning — Backtest-Driven Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix four backtest-identified issues (RANGING over-trading, SOL trailing stop, MIN_CONFIDENCE, fee drag) and run an autonomous improve→test→backtest cycle.

**Architecture:** Surgical changes to `strategies.py` REGIME_STRATEGY_MAP, `config.py` per-pair trailing stop lookup, `backtester.py`/`risk_manager.py` consuming new config, and `.env.example`/`backtest_3yr.py` confidence raise. Each fix is independently revertable.

**Tech Stack:** Python 3.14, pytest, XGBoost/LSTM (backtester), CCXT (data fetch)

---

### Task 1: Fix RANGING regime — remove Momentum from secondaries

**Files:**
- Modify: `strategies.py` (REGIME_STRATEGY_MAP RANGING entry)
- Modify: `tests/test_strategies.py` (RANGING secondary assertions)
- Modify: `tests/test_rsi_divergence_strategy.py` (1 RANGING assertion)

**Step 1: Read current RANGING config and failing tests first**

```bash
grep -n "RANGING\|ranging" tests/test_strategies.py | head -20
grep -n "RANGING\|ranging" tests/test_rsi_divergence_strategy.py | head -10
```

**Step 2: Update `strategies.py` REGIME_STRATEGY_MAP RANGING entry**

Find this block (around line 1261):
```python
MarketRegime.RANGING: {
    # Backtesting (2021-2026) showed MeanReversion/VWAP/Grid destroyed capital
    # in crypto's persistent trends. OBV+RSI divergence performs across all regimes.
    "primary": "OBVDivergence",
    "secondary": ["RSIDivergence", "EMACrossover", "Momentum"],
    "weights": {"OBVDivergence": 0.33, "RSIDivergence": 0.27, "EMACrossover": 0.22, "Momentum": 0.18},
},
```

Replace with:
```python
MarketRegime.RANGING: {
    # Backtesting (2021-2026) showed MeanReversion/VWAP/Grid destroyed capital
    # in crypto's persistent trends. OBV+RSI divergence performs across all regimes.
    # Phase 6: Removed Momentum from secondaries — trend-follower in non-trending
    # regime generated largest losing ensemble (OBVDivergence+EMACrossover+Momentum).
    "primary": "OBVDivergence",
    "secondary": ["RSIDivergence", "EMACrossover"],
    "weights": {"OBVDivergence": 0.40, "RSIDivergence": 0.35, "EMACrossover": 0.25},
},
```

**Step 3: Update test assertions in `tests/test_strategies.py`**

Find the test that checks RANGING secondaries contain "Momentum" and update it to not include Momentum. Find the test that checks weights sum to 1.0 — it should still pass since 0.40+0.35+0.25=1.0.

Look for something like:
```python
assert "Momentum" in engine.REGIME_STRATEGY_MAP[MarketRegime.RANGING]["secondary"]
```
Remove or update that assertion.

Also check and fix any weight value assertions for RANGING.

**Step 4: Update `tests/test_rsi_divergence_strategy.py`**

Find and remove/update the line that asserts "Momentum" is in RANGING secondaries:
```python
secondaries = StrategyEngine.REGIME_STRATEGY_MAP[MarketRegime.RANGING]["secondary"]
# assert "Momentum" in secondaries  # remove this assertion
assert "OBVDivergence" not in secondaries  # keep — OBVDivergence is primary, not secondary
```

**Step 5: Run tests to verify**

```bash
./venv/bin/python -m pytest tests/test_strategies.py tests/test_rsi_divergence_strategy.py -v 2>&1 | tail -20
```
Expected: All pass.

**Step 6: Run full test suite**

```bash
./venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```
Expected: 984 passed (or very close).

**Step 7: Commit**

```bash
git add strategies.py tests/test_strategies.py tests/test_rsi_divergence_strategy.py
git commit -m "perf: remove Momentum from RANGING secondaries, rebalance weights (Phase 6)"
```

---

### Task 2: Add per-pair trailing stop to `config.py`

**Files:**
- Modify: `config.py` (add SYMBOL_TRAILING_STOP_PCT + get_trailing_stop_pct)
- Modify: `tests/test_config.py` (add tests for new method)

**Step 1: Read config.py around TRAILING_STOP_PCT**

```bash
grep -n "TRAILING_STOP_PCT\|ATR_TRAILING_MULT" config.py
```

**Step 2: Add per-pair dict and classmethod to `config.py`**

After the existing `TRAILING_STOP_PCT` line (~line 46), add:
```python
# Per-pair trailing stop overrides (Phase 6: SOL needs wider stop due to higher ATR)
SYMBOL_TRAILING_STOP_PCT: dict[str, float] = {
    "BTC/USDT": 0.025,
    "ETH/USDT": 0.025,
    "SOL/USDT": 0.040,
}

@classmethod
def get_trailing_stop_pct(cls, symbol: str) -> float:
    """Return the trailing stop percentage for a given symbol.

    Uses per-pair overrides from ``SYMBOL_TRAILING_STOP_PCT`` when
    available, falling back to the global ``TRAILING_STOP_PCT``.

    Args:
        symbol: Trading pair symbol (e.g. ``"SOL/USDT"``).

    Returns:
        Trailing stop as a decimal fraction (e.g. ``0.040`` for 4%).
    """
    return cls.SYMBOL_TRAILING_STOP_PCT.get(symbol, cls.TRAILING_STOP_PCT)
```

**Step 3: Write tests in `tests/test_config.py`**

Add a new test class or method:
```python
def test_get_trailing_stop_pct_known_symbols():
    assert Config.get_trailing_stop_pct("BTC/USDT") == 0.025
    assert Config.get_trailing_stop_pct("ETH/USDT") == 0.025
    assert Config.get_trailing_stop_pct("SOL/USDT") == 0.040

def test_get_trailing_stop_pct_unknown_falls_back():
    # Unknown symbol → falls back to TRAILING_STOP_PCT (0.015 default)
    result = Config.get_trailing_stop_pct("XRP/USDT")
    assert result == Config.TRAILING_STOP_PCT
```

**Step 4: Run config tests**

```bash
./venv/bin/python -m pytest tests/test_config.py -v 2>&1 | tail -20
```
Expected: All pass (including the new tests).

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add per-pair trailing stop config (SOL=4.0%, BTC/ETH=2.5%)"
```

---

### Task 3: Wire per-pair trailing stop into `backtester.py`

**Files:**
- Modify: `backtester.py` (default trailing_stop_pct resolution)
- Modify: `tests/test_backtester.py` (if assertions exist on default trailing stop value)

**Step 1: Read Backtester.__init__ signature**

```bash
sed -n '125,165p' backtester.py
```

**Step 2: Update `backtester.py` `__init__`**

Change the default from `trailing_stop_pct: float = 0.015` to `trailing_stop_pct: float | None = None` and resolve it in the body:

```python
def __init__(
    self,
    initial_capital: float = 10_000.0,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    symbol: str | None = None,
    trailing_stop_pct: float | None = None,   # None → resolved from config per-pair
    min_confidence: float | None = None,
    max_hold_bars: int = 100,
) -> None:
```

In the `__init__` body, add after setting `self.symbol`:
```python
self.symbol = symbol or Config.TRADING_PAIR
# Resolve trailing stop: explicit arg > per-pair config > global default
self.trailing_stop_pct = (
    trailing_stop_pct
    if trailing_stop_pct is not None
    else Config.get_trailing_stop_pct(self.symbol)
)
```

Remove the old `self.trailing_stop_pct = trailing_stop_pct` line.

**Step 3: Run backtester tests**

```bash
./venv/bin/python -m pytest tests/test_backtester.py -v 2>&1 | tail -20
```
Expected: All pass. If any test hardcodes `trailing_stop_pct=0.015`, update it to pass the value explicitly.

**Step 4: Commit**

```bash
git add backtester.py tests/test_backtester.py
git commit -m "feat: backtester resolves trailing stop per-pair from Config when not explicit"
```

---

### Task 4: Wire per-pair trailing stop into `risk_manager.py` (live agent)

**Files:**
- Modify: `risk_manager.py` (Position.__post_init__ trailing stop init)
- Modify: `tests/test_risk_manager.py` (if SOL-specific assertions needed)

**Step 1: Read Position.__post_init__**

```bash
sed -n '35,70p' risk_manager.py
```

**Step 2: Update `Position.__post_init__` to use per-pair trailing stop**

Find:
```python
if self.trailing_stop == 0.0:
    trail_pct = Config.TRAILING_STOP_PCT
```

Replace with:
```python
if self.trailing_stop == 0.0:
    trail_pct = Config.get_trailing_stop_pct(self.symbol)
```

**Step 3: Run risk manager tests**

```bash
./venv/bin/python -m pytest tests/test_risk_manager.py -v 2>&1 | tail -20
```
Expected: All pass.

**Step 4: Run full test suite**

```bash
./venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```
Expected: All 984 tests pass.

**Step 5: Commit**

```bash
git add risk_manager.py tests/test_risk_manager.py
git commit -m "feat: live agent Position uses per-pair trailing stop (SOL=4.0%)"
```

---

### Task 5: Raise MIN_CONFIDENCE in `.env.example` and `backtest_3yr.py`

**Files:**
- Modify: `.env.example`
- Modify: `scripts/backtest_3yr.py`

**Step 1: Update `.env.example`**

Change:
```
MIN_CONFIDENCE=0.6
```
To:
```
MIN_CONFIDENCE=0.72
```

**Step 2: Update `scripts/backtest_3yr.py`**

Change:
```python
MIN_CONFIDENCE = 0.68       # Raised from 0.65 — reduces fee drag at high trade frequency
```
To:
```python
MIN_CONFIDENCE = 0.72       # Phase 6: raised from 0.68 — reduces marginal trades, targets <900 total
```

**Step 3: Run full test suite (no new tests needed — no logic change)**

```bash
./venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```
Expected: All pass.

**Step 4: Commit**

```bash
git add .env.example scripts/backtest_3yr.py
git commit -m "perf: raise MIN_CONFIDENCE 0.68→0.72 in backtest + .env.example (Phase 6)"
```

---

### Task 6: Run 3-year backtest and evaluate

**Step 1: Run the backtest**

```bash
./venv/bin/python scripts/backtest_3yr.py 2>&1 | tee /tmp/backtest_3yr_phase6.log
```
Expected runtime: ~20 min.

**Step 2: Extract key metrics**

```bash
grep -E "(RESULTS|Capital|Total Return|Max Draw|Sharpe|Total Trades|Win Rate|Profit Factor|Total Fees|COMPARISON|runtime)" /tmp/backtest_3yr_phase6.log | grep -v "^$"
```

**Step 3: Compare against baseline**

Baseline:
| Pair | Return | Sharpe | Trades | Win% |
|------|--------|--------|--------|------|
| BTC | −0.45% | −0.234 | 304 | 34.2% |
| ETH | −1.43% | −0.740 | 335 | 31.6% |
| SOL | −2.47% | −0.967 | 488 | 30.1% |

Success criteria:
- Total trades ≤ 900 (was 1127)
- SOL trailing-stop exit % < 45% (was 59%)
- ≥2 of 3 pairs with improved Sharpe

**Step 4: If improved → commit results doc and tag**

```bash
git add /tmp/backtest_3yr_phase6.log  # or copy to docs/
git commit -m "docs: Phase 6 backtest results — [summary of improvement]"
```

**Step 5: If still negative — proceed to cycle 2 analysis**

Analyze which strategy ensemble is now the dominant loser, and apply the next targeted fix.

---

### Cycle 2+ (if results are still negative after Cycle 1)

**Possible next-level fixes to evaluate:**

1. **Remove EMACrossover from RANGING** — if it's still a dominant loser after removing Momentum
2. **Raise RANGING OBVDivergence minimum confidence** — add a pre-check in `StrategyEngine.get_signal()`: if regime is RANGING and OBVDivergence signal confidence < 0.60, return HOLD immediately
3. **Raise MIN_CONFIDENCE further** — try 0.75 if trade count is still >800
4. **Widen SOL take-profit** — `suggested_tp_pct` in OBVDivergenceStrategy can be increased to give winners more room

Each cycle: implement → `make test` → `backtest_3yr.py` → commit if improved.
