# Context + Trigger Architecture — Design Document

**Date**: 2026-03-03
**Status**: Approved
**Replaces**: Phase 1–8 ensemble/regime architecture

---

## 1. Problem Statement

After 8 improvement cycles the system remains unprofitable on 3-year walk-forward backtests
(best result: BTC −0.12%, ETH −0.54%, SOL −0.47%). Root cause analysis identified three
structural problems that cannot be fixed by tuning:

1. **Lagging regime detection** — MA/HMM classifier confirms regime after the move has
   occurred. By the time `TRENDING_DOWN` is labeled, profitable short entries have passed.
   The "fix" of banning TRENDING_DOWN/RANGING trades is a workaround, not a solution.

2. **Intelligence is decorative** — 10 intelligence providers (whale, orderbook, funding,
   liquidation, cascade) produce rich signal data but are reduced to a single
   `adjustment_factor` (0.5–1.5) that shifts confidence by at most `0.15`. The intelligence
   layer is nearly invisible in the final decision.

3. **Double-lag stacking** — Lagging strategies (EMACrossover, Momentum, Ichimoku) run on
   top of a lagging regime label. Two lagging systems produce a deeply lagging signal.

---

## 2. Design Decisions

| Question | Decision |
|----------|----------|
| Trading styles | Swing (4h) + Intraday (1h) + Event-driven (realtime) |
| Conflict resolution | Hybrid: swing bias is prerequisite gate, intraday+events score, events can hard-veto |
| ML role | Minimal — deterministic signals from real data; ML only where rule-based is insufficient |
| Instruments | Spot (BTC/ETH/SOL) for swing/intraday; Perps (BTCUSDT/ETHUSDT) for event-driven |
| Codebase | Clean slate for trading brain; keep FastAPI/Next.js/TradeDB/Telegram/CCXT shell |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER (immutable snapshots with timestamps)           │
│  4h OHLCV │ 1h OHLCV │ 15m OHLCV │ WS: OrderBook/Liq/Perp │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────────┐         ┌──────────────────────┐
│  CONTEXT ENGINE  │         │   TRIGGER ENGINE     │
│  (every 15min)   │         │  (per-candle + event)│
│  versioned,      │         │  suggestions only —  │
│  monotonic,      │         │  no sizing/routing   │
│  never mutates   │         │  knowledge           │
│  → ContextState  │         │  → TriggerSignal[]   │
└────────┬─────────┘         └──────────┬───────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
             ┌────────────────────┐
             │  DECISION LAYER    │
             │                    │
             │  1. Gate (hard)    │  context says no → stop, score irrelevant
             │  2. Score (soft)   │  context.confidence × trigger.strength
             │  3. Route (fixed)  │  spot stays spot, perp stays perp
             └────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌─────────────┐       ┌──────────────────┐
   │  SPOT EXEC  │       │   PERP EXEC      │
   │ (belief,    │       │ (urgency, event- │
   │  slow exit) │       │  only, max 2x)   │
   │             │       │  blocked if spot │
   │             │       │  open same dir   │
   └─────────────┘       └──────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │  RISK/SAFETY SUPERVISOR│  ← cross-cutting governor
         │  (always watching)     │
         │                        │
         │  watches: drawdown,    │  single power: disable_new_trades()
         │  consecutive losses,   │
         │  vol shocks, API errs  │  never adjusts: scores / context /
         │                        │  position sizing / entries
         └────────────────────────┘
```

---

## 4. Four Immutable Rules

These rules must never be violated by any future change:

1. **Gate > Score** — If context says no (`tradeable=False` or `allowed_directions=[]`),
   score never runs. No exceptions.

2. **Routing is deterministic** — Spot trades never become perps. Perp trades never fall
   back to spot. Instrument choice is policy, not optimization.

3. **No hidden leverage stacking** — Perp exec is blocked if a spot trade is already open
   in the same direction on the same symbol.

4. **ContextState is immutable within its window** — If context changes, it is a new state
   with a new `context_id`. Nothing downstream mutates it.

---

## 5. Core Schemas

### 5.1 ContextState

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

SwingBias        = Literal["bullish", "bearish", "neutral"]
VolatilityRegime = Literal["low", "normal", "elevated", "extreme"]
FundingPressure  = Literal["long_crowded_mild", "long_crowded_extreme",
                            "short_crowded_mild", "short_crowded_extreme", "neutral"]
WhaleFlow        = Literal["accumulating", "distributing", "neutral"]
OITrend          = Literal["expanding_up", "expanding_down", "contracting", "neutral"]
RiskMode         = Literal["normal", "cautious", "defensive"]

@dataclass
class ContextState:
    context_id: str                  # "2026-03-03T14:15Z" — version key
    swing_bias: SwingBias
    allowed_directions: list[str]    # ["long"] | ["short"] | ["long","short"] | []
    volatility_regime: VolatilityRegime
    funding_pressure: FundingPressure
    whale_flow: WhaleFlow
    oi_trend: OITrend
    key_levels: dict                 # {"support": float, "resistance": float, "poc": float}
    risk_mode: RiskMode
    confidence: float                # 0.0–1.0 overall context clarity (logging + future ML)
    tradeable: bool                  # derived by ContextEngine — never set downstream
    valid_until: datetime            # next 15min boundary — triggers reject stale context
    updated_at: datetime
```

**Rules**:
- `tradeable` is derived by the Context Engine from internal signal agreement — no
  downstream component reasons about *why* it is False.
- `allowed_directions` is computed once per window and is read-only to all consumers.
- `confidence` is for logging and post-mortem analysis. It does not gate trades directly
  (yet) — that is `tradeable`.

### 5.2 TriggerSignal

```python
@dataclass
class TriggerSignal:
    trigger_id: str       # uuid
    source: str           # "momentum_1h" | "orderflow" | "liquidation" | "funding_extreme"
    direction: str        # "long" | "short"
    strength: float       # 0.0–1.0
    urgency: str          # "normal" | "high" (high = event-driven, faster expiry)
    symbol_scope: str     # "BTC" | "ETH" | "SOL" | "market"
                          # "market" scope propagates to all symbols (e.g. BTC liq → block alts)
    reason: str           # human-readable: "RSI crossed 50 upward + volume 1.8×"
    expires_at: datetime  # trigger is stale if not acted on
    raw_data: dict        # source-specific forensic details
```

**Rules**:
- Triggers know nothing about position size, instrument, or whether a trade will happen.
- `symbol_scope = "market"` means the trigger applies system-wide (e.g. a BTC liquidation
  cascade should block ETH/SOL longs too).

### 5.3 Decision

```python
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class Decision:
    action: Literal["trade", "reject"]
    reason: str
    direction: Optional[str] = None   # "long" | "short"
    route: Optional[str] = None       # "spot" | "perp"
    score: Optional[float] = None
```

**Rules**:
- Pure data. No methods, no logic.
- Decisions are loggable and replayable. Execution layers never interpret intent.

---

## 6. Decision Layer Contract

```python
from datetime import UTC, datetime
from itertools import groupby

SCORE_THRESHOLD = 0.50

def evaluate(context: ContextState, triggers: list[TriggerSignal]) -> Decision:
    # Step 1: Gate (hard) — if context says no, nothing else runs
    if not context.tradeable:
        return Decision(action="reject", reason="context_not_tradeable")
    if not context.allowed_directions:
        return Decision(action="reject", reason="no_allowed_directions")

    # Step 2: Filter to allowed directions + non-expired triggers
    valid = [
        t for t in triggers
        if t.direction in context.allowed_directions
        and t.expires_at > datetime.now(UTC)
    ]

    # Step 3: Directional consensus — 2+ triggers must agree on same direction
    by_dir: dict[str, list[TriggerSignal]] = {}
    for t in valid:
        by_dir.setdefault(t.direction, []).append(t)

    if not by_dir:
        return Decision(action="reject", reason="no_valid_triggers")

    best_dir, agreeing = max(by_dir.items(), key=lambda g: len(g[1]))
    if len(agreeing) < 2:
        return Decision(action="reject", reason="insufficient_directional_agreement")

    # Step 4: Score
    score = context.confidence * (sum(t.strength for t in agreeing) / len(agreeing))
    if score < SCORE_THRESHOLD:
        return Decision(action="reject", reason=f"score_below_threshold:{score:.2f}")

    # Step 5: Route (deterministic — event veto by risk_mode)
    if any(t.urgency == "high" for t in agreeing):
        if context.risk_mode == "defensive":
            return Decision(action="reject", reason="event_blocked_by_risk_mode")
        route = "perp"
    else:
        route = "spot"

    return Decision(action="trade", direction=best_dir, route=route, score=score)
```

Every rejection logs full `ContextState` + `TriggerSignal[]` + reason. This is the
primary truth-building mechanism — more valuable than backtests in early phases.

---

## 7. Context Engine

Runs every 15 minutes. Composes four analyzers into one `ContextState`.

### 7.1 SwingAnalyzer (4h data)
- **Price structure**: Higher Highs / Higher Lows = bullish; Lower Highs / Lower Lows = bearish
- **EMA alignment**: 21 / 50 / 200 EMA on 4h bars (not 1h — this is the lag fix)
- **Volume Profile**: Point of Control (POC) — is current price above or below value area?
- **Key levels**: Recent swing highs/lows as support/resistance

### 7.2 FundingAnalyzer (perp feed)
- Current funding rate: positive = longs crowded; negative = shorts crowded
- Funding trend (rising/falling/extreme)
- Contrarian signal at extremes: rate > 0.1% = `long_crowded_extreme` (bearish lean);
  rate < −0.05% = `short_crowded_extreme` (bullish lean)

### 7.3 WhaleFlowAnalyzer (upgraded from intelligence/whale_tracker.py)
- Net whale flow over last 4h: accumulation vs distribution
- Large order clustering at price levels
- On-chain inflow/outflow to exchanges

### 7.4 OITrendAnalyzer (perp feed)
- OI increasing + price up = trend continuation signal
- OI decreasing + price up = potential exhaustion
- OI diverging from price = reversal signal

### 7.5 `tradeable` Derivation
Context Engine sets `tradeable = False` when:
- Swing bias is `neutral` AND volatility regime is `extreme`
- Signal agreement across analyzers is too low (< 2/4 agree on direction)
- Risk Supervisor has called `disable_new_trades()`

---

## 8. Trigger Engine

Runs per-candle (1h close) and on real-time events.

### 8.1 MomentumTrigger (1h, `urgency="normal"`)
- RSI > 50 + rising → long; RSI < 50 + falling → short
- MACD zero-line cross (momentum shift, not just signal line)
- Volume confirmation: volume > 1.5× 20-bar average
- `symbol_scope` = specific symbol (BTC / ETH / SOL)

### 8.2 OrderFlowTrigger (realtime, `urgency="normal"`)
- CVD divergence: price new high but CVD not confirming = exhaustion signal
- Large bid/ask walls appearing or disappearing
- Bid-ask imbalance ratio spike (> 2σ)
- `symbol_scope` = specific symbol

### 8.3 LiquidationTrigger (realtime, `urgency="high"` → PERP)
- Estimated cascade level hit
- Liquidation volume spike (> 3σ above rolling average)
- Direction: cascade of longs = bearish trigger; cascade of shorts = bullish trigger
- `symbol_scope = "market"` when BTC cascade — blocks alt longs system-wide

### 8.4 FundingExtremeTrigger (15min, `urgency="high"` → PERP)
- Funding > 0.1% → short trigger (fade the crowd)
- Funding < −0.05% → long trigger (fade the crowd)
- Funding flip (sign change) → trend change signal
- `symbol_scope` = specific symbol

---

## 9. Execution Layers

### 9.1 Spot Executor (extended from executor.py)
- Expresses directional belief (swing + intraday agreement)
- Slower exits — structural stops at key_levels from ContextState
- Max 3 concurrent spot positions
- Stop-loss: key support/resistance level, not fixed ATR %

### 9.2 Perp Executor (new: perp_executor.py)
- Event-driven only — fires on `urgency="high"` triggers
- Max 2x leverage (hard cap, never configurable upward)
- Max 2 concurrent perp positions
- Fast invalidation: if trigger `expires_at` passes, exit
- **Blocked if spot position open in same direction on same symbol**

---

## 10. Risk/Safety Supervisor

Cross-cutting governor. Watches everything, controls nothing except the kill switch.

**Watches**:
- Daily drawdown > 3% → `disable_new_trades()`
- Consecutive losses > 4 → `disable_new_trades()`
- Volatility shock (ATR spike > 3σ) → `disable_new_trades()`
- Exchange/API error rate > threshold → `disable_new_trades()`

**Does NOT**:
- Adjust entry prices
- Modify scores or context
- Change position sizes
- Interpret why trades are bad

**Re-enables** after: configurable cooldown (default 2h) + manual override via Telegram.

---

## 11. File Changes

### Removed
```
regime_detector.py       → replaced by context_engine.py
strategies.py            → replaced by triggers/
backtester.py            → replaced by backtester_v2.py
meta_learner.py          → removed (no ML)
auto_optimizer.py        → removed
strategy_evolver.py      → removed
```

### Added
```
context_engine.py
context/
  __init__.py
  swing.py               # SwingAnalyzer
  funding.py             # FundingAnalyzer
  whale_flow.py          # WhaleFlowAnalyzer
  oi_trend.py            # OITrendAnalyzer

trigger_engine.py
triggers/
  __init__.py
  momentum.py            # MomentumTrigger
  orderflow.py           # OrderFlowTrigger
  liquidation.py         # LiquidationTrigger
  funding_extreme.py     # FundingExtremeTrigger

decision.py              # evaluate() + Decision dataclass
perp_executor.py         # Binance USDT-margined perp execution
data_snapshot.py         # Immutable timestamped data snapshots
risk_supervisor.py       # Cross-cutting governor
backtester_v2.py         # New backtester for context+trigger logic
```

### Kept (minimal changes)
```
agent.py                 # Simplified main loop
executor.py              # Spot only (extended)
risk_manager.py          # Extended with perp risk limits
config.py                # New env vars
api/                     # Unchanged
dashboard/               # Minimal updates
trade_db.py              # Unchanged
notifier.py              # Unchanged
telegram_bot.py          # Unchanged
```

---

## 12. Build Phases

| Phase | Scope | Gate to next phase |
|-------|-------|--------------------|
| **P1** | Freeze interfaces — implement `ContextState`, `TriggerSignal`, `Decision` schemas with full type annotations. Zero trading logic. | All 3 schemas compile; unit tests cover schema validation |
| **P2** | Data layer — immutable `DataSnapshot` class, multi-timeframe OHLCV fetcher (4h/1h/15m), perp WebSocket feed (funding, OI, liquidations) | Snapshots produce correct timestamps; no downstream data fetching |
| **P3** | Context Engine — all 4 analyzers, `ContextState` output, rejection logging | Context logs look correct over 48h of real market data |
| **P4** | Trigger Engine — `MomentumTrigger` + `OrderFlowTrigger` (spot triggers only), `Decision` layer with full rejection logging | Decision reject/accept log reviewed; no live trades yet |
| **P5** | Spot execution — wire into `executor.py`, `RiskSupervisor`, full spot paper trading | 2 weeks paper trading; rejection log reviewed for false pattern |
| **P6** | Perp execution — `LiquidationTrigger` + `FundingExtremeTrigger` + `perp_executor.py`, paper trading only | Event false-positive rate reviewed before enabling live perps |

---

## 13. What This Fixes

| Current Problem | How This Fixes It |
|-----------------|-------------------|
| Lagging regime detection | Context Engine uses 4h price structure + funding + OI — leading/coincident signals, not MA confirmation |
| Intelligence as scalar only | Whale flow, funding, OI, liquidation are first-class inputs to ContextState and TriggerSignals |
| No-trade zone workarounds | RANGING/TRENDING_DOWN no-trade zones eliminated — Context Engine simply sets `allowed_directions=[]` when uncertain |
| Double-lag stacking | Triggers fire on 1h momentum, not on confirmed regime + lagging strategy agreement |
| `_combine` math ceiling | Replaced by clean gate + directional consensus + score threshold |
| Hidden leverage stacking | Perp blocked when spot open same direction — hard rule |
