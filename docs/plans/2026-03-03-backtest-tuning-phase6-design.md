# Phase 6 Tuning — Backtest-Driven Fixes Design

**Date:** 2026-03-03
**Approach:** A — Targeted Micro-Tuning
**Trigger:** 3-year real-data backtest (Mar 2023–Mar 2026) showing all three pairs negative

## Problem Summary

| Issue | Evidence | Impact |
|-------|----------|--------|
| RANGING over-trading | `Ensemble(OBVDivergence, EMACrossover, Momentum)` top loser all 3 pairs | −$54/−$106/−$155 per pair |
| SOL trailing stop too tight | 59% exits via trailing stop, 6.8h avg hold | Premature exits before edge materializes |
| Low win rate (30–34%) | Break-even at current 1.5:1 payoff needs 40% WR | Every marginal trade is net-negative |
| Fee drag | $182–$291 total transaction cost per pair on $10k | Symptom of the above three |

## Baseline (3yr, Mar 2023–Mar 2026)

| Pair | Return | Sharpe | Trades | Win% | Fees+Slip |
|------|--------|--------|--------|------|-----------|
| BTC/USDT | −0.45% | −0.234 | 304 | 34.2% | $182 |
| ETH/USDT | −1.43% | −0.740 | 335 | 31.6% | $200 |
| SOL/USDT | −2.47% | −0.967 | 488 | 30.1% | $291 |

## Fix 1 — RANGING Secondary Cleanup (`strategies.py`)

Remove `Momentum` from `REGIME_STRATEGY_MAP[RANGING]` secondaries. Momentum is a
trend-follower placed in a non-trending regime — it adds noise and generated the largest
single losing ensemble across all three pairs.

```python
# Before
MarketRegime.RANGING: {
    "primary": "OBVDivergence",
    "secondary": ["RSIDivergence", "EMACrossover", "Momentum"],
    "weights": {"OBVDivergence": 0.33, "RSIDivergence": 0.27, "EMACrossover": 0.22, "Momentum": 0.18},
}

# After
MarketRegime.RANGING: {
    "primary": "OBVDivergence",
    "secondary": ["RSIDivergence", "EMACrossover"],
    "weights": {"OBVDivergence": 0.40, "RSIDivergence": 0.35, "EMACrossover": 0.25},
}
```

Test files: `tests/test_strategies.py`, `tests/test_rsi_divergence_strategy.py`

## Fix 2 — Per-Pair Trailing Stop (`config.py`, `backtester.py`, `risk_manager.py`, `backtest_3yr.py`)

SOL's hourly ATR is ~2–3× BTC's. The uniform 2.5% trailing stop acts as a noise stop on SOL.

**`config.py`** — add per-pair lookup:
```python
SYMBOL_TRAILING_STOP_PCT: dict[str, float] = {
    "BTC/USDT": 0.025,
    "ETH/USDT": 0.025,
    "SOL/USDT": 0.040,
}

@classmethod
def get_trailing_stop_pct(cls, symbol: str) -> float:
    return cls.SYMBOL_TRAILING_STOP_PCT.get(symbol, cls.TRAILING_STOP_PCT)
```

**`backtester.py`** — default `trailing_stop_pct=None`, resolve from `Config.get_trailing_stop_pct(symbol)`.

**`risk_manager.py`** `Position.__post_init__` — replace `Config.TRAILING_STOP_PCT` with `Config.get_trailing_stop_pct(self.symbol)`.

**`backtest_3yr.py`** — remove hardcoded constant, pass per-pair value from config.

## Fix 3 — MIN_CONFIDENCE Raise (`.env.example`, `backtest_3yr.py`)

Raise confidence floor from 0.68 → 0.72 in backtest and from 0.6 → 0.72 in `.env.example`.
Target: reduce trade count from ~1127 → ~700 across 3 pairs.

## Success Criteria (re-run 3yr backtest)

- Total trades reduced by ≥20% (1127 → <900)
- SOL trailing-stop exit % drops from 59% to <45%
- At least 2 of 3 pairs show improved Sharpe ratio
- All 984 tests pass

## Autonomous Improvement Cycle

After each backtest:
1. If results improve: commit, continue to next cycle
2. If results worsen: revert the last change, try alternative
3. Cycle limit: up to 3 full improvement iterations
