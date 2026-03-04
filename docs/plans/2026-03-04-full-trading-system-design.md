# Full Trading System Design
**Date:** 2026-03-04
**Status:** Approved
**Goal:** Complete the crypto trading agent into a production-grade live trading system (spot + perp, BTC/USDT, ETH/USDT, SOL/USDT on Binance)

---

## Context

The system is at Phase 9 Enhancements (1168 tests). Phase 9 is the primary execution pipeline:
`ContextEngine → TriggerEngine → evaluate() → Decision`. Three triggers are live
(Momentum, OrderFlow, LiquiditySweep). Two blockers prevent going live:

1. `decision.route` ("spot" / "perp") is computed but never read — perp execution is missing
2. No real-time data (REST polling only — high-urgency signals get stale prices)

---

## Approach: Phase-Gated Roadmap (Approach C)

Four sequential phases with a hard go-live gate after Phase 11.

| Phase | Focus | Go-live gate? |
|-------|-------|---------------|
| 10 | Execution Foundation | — |
| 11 | Validation Layer | ← GO LIVE HERE |
| 12 | Signal Enrichment | post-live |
| 13 | Execution Quality | post-live |

---

## Phase 10 — Execution Foundation

### Goals
- WebSocket real-time data (eliminates REST polling latency for high-urgency signals)
- Full perp execution path (spot + perp both routable from `decision.route`)
- `LiveExecutor.partial_close()` (closes the current `hasattr` gap)
- Structured alerting (liquidation proximity, heartbeat, critical errors)

### Architecture

```
Data Layer
  ws_feed.py — ccxt.pro WebSocket (kline.1h, depth20, aggTrades per symbol)
    ├── updates _ws_cache[symbol] with latest OHLCV + bid/ask + CVD increment
    ├── exposes get_latest(symbol) → DataSnapshot
    ├── reconnect: exponential backoff 1s→2s→4s→max 60s; REST fallback after 30s
    └── paper mode: not started (REST continues)

Decision Layer (unchanged)
  ContextEngine → TriggerEngine → evaluate() → Decision(route="spot"|"perp")

Execution Layer  ← decision.route read HERE (currently missing)
  route="spot"  → SpotExecutor (existing + partial_close added to LiveExecutor)
  route="perp"  → PerpExecutor (new)
    ├── PerpPaperExecutor: simulates leverage, liquidation_price, funding cost
    └── PerpLiveExecutor: Binance USDT-Margined futures via CCXT

Monitoring Layer (new alerting.py)
  alert(msg, severity: "info"|"warn"|"critical")
    ├── critical → Telegram + log
    ├── warn → log only (configurable via ALERT_CHANNELS)
    └── perp: liquidation proximity < 10% → CRITICAL alert
```

### New / Modified Files

| File | Change |
|------|--------|
| `ws_feed.py` | NEW — ccxt.pro WebSocket wrapper, DataSnapshot cache |
| `executor.py` | ADD PerpPaperExecutor, PerpLiveExecutor; ADD partial_close to LiveExecutor |
| `agent.py` | READ decision.route; route to perp_executor when "perp" |
| `alerting.py` | NEW — severity-aware alerting funnel |
| `risk_manager.py` | ADD PerpPosition dataclass (leverage, margin_used, liquidation_price, funding_pnl) |

### Key Design Decisions

**PerpPosition** extends `Position` with:
- `leverage: int` — from Config `PERP_LEVERAGE` (default 3×)
- `margin_used: float` — `entry_price * quantity / leverage`
- `liquidation_price: float` — `entry × (1 - 1/leverage)` for longs; `entry × (1 + 1/leverage)` for shorts
- `funding_pnl: float` — accrued every 8h using FundingAnalyzer's cached rate

**`_execute_trade()` routing:**
```python
executor = self.perp_executor if decision.route == "perp" else self.executor
```
Both executors share the same interface — `submit_order`, `partial_close`, `cancel_order`.

**Paper vs. Live symmetry:** `PerpPaperExecutor` mirrors `PaperExecutor` exactly — simulates instant fill at mid-price, synthetic funding, liquidation event if price crosses `liquidation_price`. This lets paper mode validate the full perp path before going live.

### Error Handling
- WebSocket disconnect → reconnect with backoff; REST fallback if >30s unreachable; `RiskSupervisor.on_api_error()` fires existing circuit breaker
- Perp order rejection (insufficient margin, leverage cap) → `ExecutionError` raised, logged, no retry
- Funding cost stale (>8h cache) → skipped silently, position management unaffected

### Tests
- `test_ws_feed.py` — mock WebSocket frames, verify cache update and reconnect backoff
- `test_perp_executor.py` — liquidation price formula, funding PnL accrual, partial_close quantity reduction
- `test_executor.py` — add `TestLiveExecutorPartialClose` (3 tests, same pattern as `TestPartialClose`)
- `test_agent_routing.py` — verify `_run_phase9_cycle` sends "perp" decisions to `perp_executor`

---

## Phase 11 — Validation Layer

### Goals
- Multi-cycle Phase 9 walk-forward backtest with full trigger suite
- Sharpe / Calmar / per-trigger attribution analytics
- Dashboard analytics page
- Hard go-live checklist — must pass before `TRADING_MODE=live`

### Architecture

```
scripts/backtest_phase9_wf.py
  5-fold walk-forward × 3 pairs × 2.5yr (mid-2023 → end-2025)
  fold = 5 months in-sample + 1 month out-of-sample, rolling forward
  full Phase 9 simulation: ContextEngine → TriggerEngine → evaluate() → trade
  perp route simulates leverage + funding cost
  outputs: pnl_pct, sharpe, calmar, max_drawdown, trigger_attribution per fold/pair

analytics.py (new module)
  compute_performance(trades) → PerformanceStats
  compute_trigger_attribution(trades, decisions_log_path) → list[TriggerAttribution]
  no new DB table — computed on demand from TradeDB + phase9_decisions.jsonl

api/routes/analytics.py (new route group)
  GET /analytics/performance?days=30 → PerformanceStats
  GET /analytics/triggers?days=30    → list[TriggerAttribution]

dashboard/app/analytics/page.tsx (new page)
  equity curve + drawdown overlay (Recharts AreaChart)
  per-trigger win-rate / avg-PnL bar chart
  Sharpe / Calmar / max-DD summary cards
```

### Data Schemas

```python
@dataclass
class PerformanceStats:
    sharpe: float          # annualised, risk-free=0
    calmar: float          # annualised return / max drawdown
    sortino: float         # downside-deviation Sharpe
    max_drawdown: float    # peak-to-trough fraction
    win_rate: float        # 0–1
    avg_hold_hours: float
    total_trades: int
    pnl_pct: float

@dataclass
class TriggerAttribution:
    source: str            # "momentum" | "orderflow" | "liquidity_sweep" | "ml_xgboost"
    trade_count: int
    win_rate: float
    avg_pnl_pct: float
    avg_strength: float    # mean trigger.strength for fired signals
```

### Walk-Forward Go-Live Gate (hard)
- Aggregate Sharpe ≥ 0.0 across all 3 pairs across all 5 folds
- No single pair with PnL < −1.5% per fold consistently

### Go-Live Manual Checklist (all must pass)
1. ✅ Walk-forward Sharpe ≥ 0.0 (all pairs)
2. ✅ Perp paper mode ≥ 5 days, zero unexpected liquidation events
3. ✅ `alerting.py` verified — WARN and CRITICAL confirmed delivered to Telegram
4. ✅ `RiskSupervisor` kill-switch verified to fire in paper under simulated drawdown

### Tests
- `test_analytics.py` — `compute_performance` with known trades; Sharpe formula; `compute_trigger_attribution` with mock JSONL
- `test_analytics_routes.py` — endpoints return correct schema; empty history returns zeros not 500
- Walk-forward: smoke test with `--pairs BTC/USDT --folds 1` fast path

---

## Phase 12 — Signal Enrichment (post-live)

### Goals
- 2 new triggers (Divergence, VolumeSpike) to increase consensus opportunities
- ML re-integrated as a confirming trigger (not standalone decision-maker)
- Optuna-based Phase 9 parameter tuning using Phase 11 walk-forward as objective

### New Triggers

**`triggers/divergence.py` — DivergenceTrigger**
- Bullish: price lower low + RSI higher low → long
- Bearish: price higher high + RSI lower high → short
- Lookback: 20 bars, 2 confirmed pivots required
- `strength=0.70`, `urgency="normal"`, TTL=90 min
- Wired into `TriggerEngine.on_1h_close()`

**`triggers/volume_spike.py` — VolumeSpikeТrigger**
- Fires when `volume > rolling_20_avg × 2.5` AND `bar_range > atr × 0.6`
- Direction from bar colour (close vs open)
- Deduplication: suppressed if same-direction signal fired within last 3 bars
- `strength = min(0.5 + (ratio - 2.5) × 0.1, 0.85)`
- `urgency="high"` when `ratio > 5.0` (extreme spike → perp route)
- TTL=60 min
- Wired into `TriggerEngine.on_1h_close()`

**`triggers/ml_signal.py` — MLSignalTrigger**
- Wraps existing XGBoost model: `strength = abs(proba_long - 0.5) × 2`
- Fires only when `strength >= 0.30`
- Source: `"ml_xgboost"`, TTL=90 min, `urgency="normal"`
- No-op if `_model.is_trained` is False
- Opt-in: `USE_ML_TRIGGER=true` in `.env` (off by default until walk-forward validates it)
- Wired into `TriggerEngine` (called from `_run_phase9_cycle` after model prediction)

### Phase 9 Parameter Tuning

**`scripts/tune_phase9.py`** — Optuna TPESampler, 50 trials

Parameters tuned:

| Parameter | Range | Current default |
|-----------|-------|-----------------|
| `SCORE_THRESHOLD` | 0.35 – 0.65 | 0.50 |
| Momentum TTL (min) | 30 – 120 | 75 |
| LiquiditySweep TTL (min) | 30 – 120 | 75 |
| Divergence TTL (min) | 45 – 120 | 90 |
| `MIN_TRIGGER_STRENGTH` floor | 0.40 – 0.70 | none |

- Objective: maximise mean Sharpe across BTC/ETH/SOL on Phase 11 walk-forward
- Output: best params printed; engineer commits to `config.py` manually (same pattern as Phase 8)

### Tests
- `test_triggers.py` — add `TestDivergenceTrigger` (6 tests: bullish, bearish, insufficient pivots, boundary)
- `test_triggers.py` — add `TestVolumeSpikeTrigger` (5 tests: spike fires, dedup suppresses, urgency escalation)
- `test_trigger_engine.py` — verify all 5 triggers wired into `on_1h_close`

---

## Phase 13 — Execution Quality (post-live)

### Goals
- Limit order entry for `urgency="normal"` trades (taker fee avoidance)
- Slippage tracking per trade and per trigger source
- TWAP deferred (not relevant at early capital levels)

### Limit Order Entry

**`executor.py` — `LimitOrderMixin`** applied to `LiveExecutor` and `PerpLiveExecutor`

For `urgency="normal"`:
1. Post limit order at `bid + 1 tick`, `time_in_force="GTC"`
2. If not filled within `LIMIT_ORDER_TIMEOUT_S` (default 30s) → cancel + market order
3. Record `fill_type: "limit" | "market_fallback"` in TradeDB

For `urgency="high"`: market order unchanged (speed > cost).

Enabled via `USE_LIMIT_ORDERS=true` in `.env` (off by default).
`PaperExecutor` always simulates instant fill at mid-price — no limit logic.

### Slippage Tracking

**`analytics.py` extensions:**
```python
avg_slippage_bps: float                   # basis points, positive = worse than expected
slippage_by_trigger: dict[str, float]     # source → avg bps
```

- `expected_price` = signal price at decision time (from `_run_phase9_cycle`)
- `filled_price` = actual fill from executor response
- `slippage_bps` new column in TradeDB (NULL for old rows — backward-compatible)
- Dashboard analytics page: slippage heatmap by trigger source and hour-of-day

### What Is Explicitly Deferred
- **TWAP** — only relevant when a single order moves the market (>0.5% of 24h volume)
- **Multi-exchange routing** — Binance-only for now
- **Tax / trade journal export** — not in scope

### Tests
- `test_executor.py` — `TestLimitOrderEntry`: timeout triggers market fallback; fill_type recorded
- `test_analytics.py` — slippage_bps computed correctly; None fill_price returns 0 bps gracefully

---

## Summary of All New Files

| File | Phase | Type |
|------|-------|------|
| `ws_feed.py` | 10 | New module |
| `alerting.py` | 10 | New module |
| `analytics.py` | 11 | New module |
| `api/routes/analytics.py` | 11 | New route group |
| `dashboard/app/analytics/page.tsx` | 11 | New dashboard page |
| `triggers/divergence.py` | 12 | New trigger |
| `triggers/volume_spike.py` | 12 | New trigger |
| `triggers/ml_signal.py` | 12 | New trigger |
| `scripts/backtest_phase9_wf.py` | 11 | New script |
| `scripts/tune_phase9.py` | 12 | New script |

## Summary of Modified Files

| File | Phases | Changes |
|------|--------|---------|
| `executor.py` | 10, 13 | PerpPaperExecutor, PerpLiveExecutor, LiveExecutor.partial_close, LimitOrderMixin |
| `risk_manager.py` | 10 | PerpPosition dataclass |
| `agent.py` | 10 | Read decision.route; route to perp_executor |
| `trigger_engine.py` | 12 | Wire DivergenceTrigger, VolumeSpikeТrigger, MLSignalTrigger |
| `analytics.py` | 13 | avg_slippage_bps, slippage_by_trigger |
| `trade_db.py` | 13 | slippage_bps column (NULL-safe migration) |
