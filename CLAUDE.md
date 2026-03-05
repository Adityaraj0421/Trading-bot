# CLAUDE.md — Crypto Trading Agent

## Project Overview

Autonomous crypto trading agent (Python 3.14 + Next.js 16) that trades BTC/USDT, ETH/USDT, SOL/USDT on Binance. Runs 11 strategies with ML ensemble (XGBoost + LSTM), 10 intelligence providers, risk management, and real-time intelligence signals. Paper and live trading modes.

## Development History

| Phase | Summary |
|-------|---------|
| v1–v6 | Core trading loop, CCXT integration, paper/live modes, basic strategies |
| v7–v8 | Risk manager (Kelly sizing, drawdown gates), TradeDB persistence, Telegram bot, dashboard |
| v9.1 | 10th strategy (Ichimoku), Williams %R / CCI features (24 FEATURE_COLUMNS), pair scorer, ATR trailing + breakeven stops, 935 tests |
| Phase 4 | Code quality sweep: type hints + Google-style docstrings across all 37 production modules (37+ files, ~4,000 annotation additions, 0 ruff errors) |
| Phase 5 | 11th strategy (RSI Divergence + Stochastic), 10th intelligence provider (Fear & Greed Index), ML feature importance pruning (opt-in via `ML_FEATURE_PRUNING`), new `/model/feature-importance` API, dashboard Model page + Fear & Greed card; 984 tests |
| Phase 5 Tuning | 5-yr walk-forward backtest (BTC/ETH/SOL, 2021–2026, 1h bars, ~78 min); rebalanced `REGIME_STRATEGY_MAP` — removed MeanReversion/VWAP/Grid from RANGING (replaced with OBVDivergence), boosted Momentum across trending regimes; widened trailing stop 1.5%→2.5% |
| Phase 5 Tuning v2 | 3-yr BTC backtests diagnosed ML HOLD dominance (100% HOLD on sampled bars) + `_combine` math ceiling (0.60×conf ≤ 0.57 < threshold). Fix: `_combine` ONE_HOLD multiplier 0.60→0.75 (Option C), MIN_CONFIDENCE 0.65→0.68. `BreakoutStrategy` triple-confirmation filter (strict price exceed + volume ≥ 1.5× + bar range ≥ 0.8×ATR) reduced Breakout false positives 96%. Synthetic 3yr: 2198 trades/−10.7% → 1397 trades/−7.2% (−36% trades, −35% fee drag) |
| Phase 6 | 5-cycle RANGING regime optimization using 3-yr walk-forward backtests (`scripts/backtest_3yr.py`). Added per-pair trailing stop to `Config` (BTC/ETH=2.5%, SOL=4.0%); `risk_manager.py` now calls `Config.get_trailing_stop_pct(symbol)`. Added RANGING confidence gate in `StrategyEngine.run()` — OBVDiv primary must reach conf ≥ 0.68 or ensemble short-circuits to HOLD. Best-ever cycle results: BTC −0.08% (Cycle 5), ETH −0.88% (Cycle 2), SOL −2.15% (Cycle 4). Key finding: 2-strategy RANGING (OBV 0.60 / RSI 0.40 + gate 0.68) creates mathematical RSI-veto impossibility — RSI max veto score 0.40 × 1.0 = 0.40 < OBV min pass score 0.60 × 0.68 = 0.408. 989 tests. |
| Phase 7 | RANGING made unconditional no-trade zone in `StrategyEngine.run()` — returns HOLD before any strategy runs. 3-yr walk-forward: BTC −0.53%, ETH −1.26%, SOL **−0.97%** (SOL +1.50pp vs baseline, trade counts halved: 670/578/429→255/199/185). New dominant loser: `Ensemble(EMACrossover, Momentum)` in TRENDING_DOWN (105 SOL trades/−$108; 47 ETH trades/−$57). 988 tests. |
| Phase 8 | 10-cycle improvement loop using 3-yr walk-forward backtest (BTC/ETH/SOL). Baseline C1: −0.53%/−1.26%/−0.97%. Final C8: −0.12%/−0.54%/−0.47% (**+0.54pp avg vs baseline**). Cycles: C1 TRENDING_DOWN revamp, C2 HIGH_VOL Breakout gate, C3 direction gates + MIN_CONF 0.68, C4 ATR floor 0.6%, C5 EMA weight cut + ATR 0.8%, C6 TRENDING_DOWN no-trade zone, C7 regime-adaptive trailing stop (TRENDING_UP 2.5%/HIGH_VOL 2.0%), C8 EMACrossover removed from TRENDING_UP entirely. 963 tests. |
| Phase 9 | Context + Trigger architecture — `ContextEngine` (SwingAnalyzer/FundingAnalyzer/WhaleAnalyzer/OITrendAnalyzer) → `TriggerEngine` (MomentumTrigger/OrderFlowTrigger + optional LiquidationTrigger/FundingExtremeTrigger) → `evaluate()` → `Decision`. **Phase 9 is now the primary execution pipeline**: `_run_pair_cycle()` stripped to exit management only; `_run_phase9_cycle()` drives all trade entry via `_execute_trade()`. Phase 8 code (strategies.py, decision_engine.py, ML, RL) kept for backtest scripts, never called from agent loop. Phase 9 backtest (1yr BTC/ETH/SOL): avg −0.03% (7/9/6 trades). Decision log: `data/phase9_decisions.jsonl`. |
| Phase 9 Enhancements | Funding extreme gate in `evaluate()` (`effective_allowed` list blocks crowded-side trading); `LiquiditySweepTrigger` — equal-highs/lows cluster sweep detection (strength 0.65, 75min TTL), wired into `TriggerEngine.on_1h_close()`; Partial TP system — `Position.partial_tp_levels/partial_exits` + `RiskManager.check_partial_tp()` + `PaperExecutor.partial_close()`; BTC dominance gate blocks SOL counter-trend entries when `_btc_swing_bias` conflicts. 1168 tests. |

## Architecture

```
agent.py (main loop, runs in background thread)
    ├── data_fetcher.py           → CCXT/Binance market data
    ├── indicators.py             → 15+ technical indicators (24 FEATURE_COLUMNS for ML)
    ├── risk_manager.py           → Position sizing, stop-loss, daily limits
    ├── executor.py               → Paper/live trade execution
    ├── notifier.py               → Telegram/Discord alerts
    │
    ├── [Phase 9 — primary execution pipeline]
    ├── context_engine.py         → Builds ContextState from snapshot + intel feeds
    ├── context/swing.py          → 4h EMA 21/50/200 structure → swing_bias
    ├── context/funding.py        → Funding rate → crowding pressure
    ├── context/whale.py          → Net whale flow → sentiment
    ├── context/oi_trend.py       → OI vs price → conviction
    ├── trigger_engine.py         → Manages TriggerSignal TTL + consensus
    ├── triggers/momentum.py      → 1h RSI + MACD zero-cross + volume
    ├── triggers/orderflow.py     → CVD divergence + order imbalance
    ├── triggers/liquidity_sweep.py → Equal-highs/lows sweep detection (strength 0.65, 75min TTL)
    ├── decision.py               → evaluate() → Decision(action, direction, score)
    ├── decision_logger.py        → JSONL audit log (data/phase9_decisions.jsonl)
    ├── multi_timeframe_fetcher.py→ Fetches 1h/4h/15m DataSnapshot per symbol
    ├── risk_supervisor.py        → Kill switch (drawdown/loss streak/API errors)
    │
    └── [Phase 8 — backtest only, not called from agent loop]
        ├── decision_engine.py    → Strategy ensemble + ML signals
        ├── strategies.py         → 11 strategies + REGIME_STRATEGY_MAP
        ├── backtester.py         → Walk-forward backtester
        └── regime_detector.py   → MarketRegime classification

api/server.py (FastAPI, hosts agent in lifespan)
    ├── api/data_store.py    → Thread-safe agent↔API bridge
    ├── api/routes/          → REST endpoints
    ├── api/websocket_manager.py → Real-time push to dashboard
    └── telegram_bot.py      → Interactive Telegram commands

dashboard/ (Next.js React app, port 3000)
    ├── app/                 → Pages (trades, equity, autonomous, etc.)
    ├── components/          → Reusable UI (charts, tables, skeletons)
    └── lib/                 → API client, WebSocket hook, types

intelligence/ (11 signal modules, imported by decision_engine.py)
    ├── aggregator.py        → Orchestrates all intelligence sources (10 providers)
    ├── orderbook.py         → VPIN, CVD, spoofing detection, bid/ask walls
    ├── whale_tracker.py     → On-chain large-transaction flow
    ├── correlation.py       → Cross-asset correlation (yfinance optional)
    ├── funding_oi.py        → Perpetual funding rate + open interest
    ├── liquidation.py       → Estimated liquidation cascade levels
    ├── llm_sentiment.py     → LLM-based sentiment (Anthropic/OpenAI)
    ├── news_sentiment.py    → Reddit/RSS headline sentiment
    ├── onchain.py           → On-chain metrics + ML feature extraction
    ├── cascade_predictor.py → Cascade risk score from liquidation data
    └── fear_greed.py        → Fear & Greed Index (alternative.me, contrarian signal, 1-hour cache)
```

## Development Commands

```bash
make dev          # Start API + Dashboard (Ctrl+C stops both)
make stop         # Kill all running processes
make status       # Check if services are running
make test         # Run all Python tests
make test-coverage # Tests with coverage report
make lint         # Run ruff linter
make format       # Auto-format code with ruff
make install      # Install Python + Node deps
make build        # Build dashboard for production
make help         # Show all available commands
```

## Key Entry Points

| File | Purpose |
|------|---------|
| `agent.py` | Main trading loop (~1,766 lines). Called by api/server.py lifespan |
| `api/server.py` | FastAPI app factory, auth middleware, agent thread |
| `config.py` | All configuration from .env |
| `dashboard/app/page.tsx` | Dashboard home page |
| `pair_scorer.py` | Dynamic pair selection by ATR×volume score; rescored every N agent cycles |
| `scripts/backtest_5yr.py` | 5-year OHLCV backtest script — paginated CCXT fetch → Backtester (quarterly walk-forward). Run: `python scripts/backtest_5yr.py` (~78 min for BTC/ETH/SOL) |
| `scripts/backtest_3yr.py` | 3-year OHLCV backtest — same structure as 5yr but MIN_CONFIDENCE=0.65, RETRAIN_EVERY=2160 bars, no global TRAILING_STOP_PCT (resolved per-pair from Config). Run: `python scripts/backtest_3yr.py` (~18–21 min for BTC/ETH/SOL) |
| `api/routes/model.py` | `GET /model/feature-importance` — XGBoost feature importances + model tier/status |

## Testing

- **Framework**: pytest (tests/ directory, ~50 test files, 1168 tests)
- **Run**: `make test` or `./venv/bin/python -m pytest tests/ -v`
- **Linting**: `make lint` (ruff) — should report 0 errors
- **API tests**: `tests/test_api.py` — uses TestClient with auth header injection
- **Config tests**: `tests/test_config.py` — validates all config loading
- **Coverage**: `make test-coverage` (generates htmlcov/ report)

**Important**: The test client fixture in `test_api.py` automatically includes the API auth key when `API_AUTH_KEY` is set in .env. If you add new API tests, use the existing `client` fixture.

**Known env-dependent failure**: `test_config.py::TestConfigDefaults::test_default_min_confidence` fails when `.env` has `MIN_CONFIDENCE` ≠ 0.35. Not a code bug — adjust `.env` or expect 934 passing locally.

## Key Patterns

1. **DataStore bridge**: Agent thread writes to `api/data_store.py` (thread-safe with `threading.Lock`). API reads from it. Broadcast callback pushes to WebSocket clients.

2. **SWR + WebSocket hybrid**: Dashboard uses SWR for initial load + 30s fallback polling. WebSocket injects real-time updates into SWR cache via `mutate()`. **Critical**: SWR cache keys in page components must exactly match the keys used in `useWebSocket.ts` mutations — e.g. trades page must use `"/trades"` (not `"/trades-full"`) or real-time updates are silently discarded.

3. **Trade confirmation flow**: Agent thread calls `telegram_bot.request_confirmation()` → sends inline keyboard → blocks on `threading.Event.wait(60s)` → webhook callback sets decision.

4. **Auth middleware**: `API_AUTH_KEY` in .env enables auth. Header: `X-API-Key`. Public paths: `/health`, `/docs`, `/telegram/webhook`. Tests auto-inject the key. **Security**: All secret comparisons use `secrets.compare_digest()` — WebSocket auth, Telegram webhook secret. Never use plain `!=` for API key comparison.

5. **Two-tier data system**: `DataStore` (in-memory, resets on restart) is the fast real-time bridge. `TradeDB` (SQLite, `data/trades.db`) is the persistent store. Routes prefer `TradeDB` via `store.get_trade_db()` when available, falling back to the in-memory log — so trade history survives restarts. The `/status` endpoint and Telegram bot commands enrich session-only snapshots with TradeDB cumulative stats (total_pnl, total_trades, win_rate). **Important**: `TradeDB.get_total_stats()` returns `win_rate` as a percentage (e.g. 62.5); the snapshot and dashboard use a 0-1 fraction (e.g. 0.625) — always divide by 100 when merging. TradeDB uses field names `pnl_net`/`strategy_name` while the in-memory trade log uses `pnl`/`strategy`.

6. **API query limits**: All `limit` query parameters have both lower and upper bounds (e.g. `Query(ge=1, le=1000)`). When adding new endpoints with pagination, always include `le=` to prevent memory exhaustion.

7. **Health endpoint staleness**: `/health` returns 503 "degraded" if `updated_at` in the DataStore snapshot is >300s old. Monitoring and load balancers can use this to detect a hung agent.

8. **Backtest concurrency guard**: `api/routes/backtest.py` uses `_state_lock = threading.Lock()` around the check-then-create of the background thread. Any new concurrent-singleton patterns should follow the same lock-protected check-and-set approach.

9. **Notification logging**: `notifier.py:_send_all()` is the single funnel for all alert channels. It pushes to `DataStore.append_notification()` for the `/notifications` API endpoint. The DataStore reference is wired via `notifier.set_data_store()` called from `agent.py:set_data_store()`.

10. **Autonomous event push**: `agent.py` uses a high-water mark (`_event_hwm`) to push only NEW events from `decision.event_log` to DataStore each cycle. The event_log deque is session-only (not persisted to SQLite), so events reset on restart — only the `total_autonomous_decisions` counter survives via agent state.

## Environment

- **Required**: `.env` file (copy from `.env.example`)
- **Python venv**: `./venv/` (activate with `source venv/bin/activate`)
- **Node deps**: `dashboard/node_modules/` (install with `cd dashboard && npm install`)
- **Ports**: API on 8000, Dashboard on 3000
- **Mode**: Set `TRADING_MODE=paper` for paper trading, `live` for real trading

## Exception Hierarchy

All domain exceptions inherit from `TradingError` (in `exceptions.py`):
- `DataFetchError` — failed to fetch market data
- `ExecutionError` — failed to execute/cancel an order
- `ValidationError` — config or input validation failure
- `ModelError` — ML model training or prediction failure

## Code Quality

- **Linter/Formatter**: ruff (config in `pyproject.toml`)
- **Type hints**: All public methods are annotated (Python 3.10+ union syntax with `from __future__ import annotations` for deferred evaluation)
- **Docstrings**: Google-style on all public methods and classes
- **Logging**: All modules use `_log = logging.getLogger(__name__)` — no stray `print()` in production code (except CLI output in backtester/agent banner)

## Common Gotchas

- `.env` must exist — config.py reads from it via python-dotenv
- `API_AUTH_KEY` affects both API access and tests — the test fixture handles this
- Dashboard `.env.local` has `NEXT_PUBLIC_API_KEY` — must match `API_AUTH_KEY`
- Agent runs as daemon thread inside FastAPI lifespan — not a separate process
- Telegram webhook URL changes with every ngrok restart — update `.env` accordingly
- `venv/` must be activated or on PATH for `make` commands to find Python packages
- **Stale agent state**: delete `data/agent_state.json`, `data/agent_state_model.pkl`, `data/agent_state_autonomous.json` to reset capital/PnL to `INITIAL_CAPITAL` from `.env`
- **Orphaned open trades** after a crash: handled automatically by `_reconcile_trade_db()` on startup (marks stale DB open trades as `abandoned`). Manual fallback: `sqlite3 data/trades.db "UPDATE trades SET status='abandoned', exit_reason='orphaned_on_restart' WHERE status='open';"` — `data/trades.db` is the source of truth; `DataStore._trade_log` is session-only
- **`demo_data.py`**: exports `generate_ohlcv()` — the old name `generate_demo_ohlcv` was removed. `decision_engine.py` uses deferred imports inside method bodies (not top-level), so import errors there only surface when strategy evolution runs (hundreds of cycles in), not at startup
- **Multi-pair PnL**: `self._last_prices` (dict, per-pair, initialized in `__init__`) is the correct source — skip positions whose symbol isn't yet in the dict rather than falling back to `self._last_price` (scalar, whichever pair ran last). Equity snapshot code in `agent.py` uses `if p.symbol in self._last_prices` guard.
- **Orphan reconciliation float precision**: `_reconcile_trade_db()` rounds `entry_price` to 2 decimal places before matching in-memory positions against DB records — prevents false-positive orphan marking from float imprecision between the DB INSERT and the Position object.
- **SL/TP validation**: `risk_manager.py:calculate_stop_take()` validates that stop-loss is strictly below entry (long) or above entry (short) after regime-adaptive multipliers. Falls back to 2%/3% defaults and logs a warning — doesn't silently produce self-triggering levels.
- **Kelly fraction edge case**: `_kelly_fraction()` returns `min(0.01, MAX_POSITION_PCT)` (not `MAX_POSITION_PCT`) when a strategy has no losing trades — avoids aggressive sizing from incomplete win/loss data.
- **Uvicorn `--reload` breaks DataStore wiring**: Hot-reload creates a new DataStore but the agent thread keeps its old reference. `set_data_store()` only runs during initial lifespan startup. After code changes, do a full `make stop && make dev` restart — don't rely on uvicorn auto-reload for changes that affect agent↔API wiring.
- **Dashboard rendering nested API data**: Intelligence signals contain nested dicts/arrays (imbalances, bid_walls, top_headlines). Always use type checks (`typeof val === "object"`, `Array.isArray(val)`) before rendering — `String(obj)` produces `[object Object]`.
- **LSTM model shape invalidation**: When `FEATURE_COLUMNS` count changes (e.g. adding new indicators), delete `data/agent_state_model.pkl` before restarting — LSTM input layer shape is fixed at train time and will crash on mismatch. **Note**: ML feature pruning (`ML_FEATURE_PRUNING=true`) is safe — `prune_and_retrain()` saves the reduced `feature_cols` in `agent_state_model.pkl`, so restarts after pruning load the correct shape automatically.
- **ML feature pruning**: `ML_FEATURE_PRUNING=true` / `ML_TOP_FEATURES=14` in `.env` activates opt-in feature pruning after each training cycle. The agent reduces FEATURE_COLUMNS from 24 to the top-k most important XGBoost features. `feature_cols` is persisted in the model state file — deleting `data/agent_state_model.pkl` resets back to all 24 features. Pruning fires on every training cycle; floor is 5 features minimum.
- **HOLD signal confidence**: Strategies return `confidence=0.3` for HOLD (not `0.0`). Only the data-guard (insufficient rows, e.g. `len(df) < 3`) returns `confidence=0.0`. Test assertions on HOLD should use `< 0.5`, not `== 0.0`.
- **Test count assertions**: When adding strategies or FEATURE_COLUMNS, update count assertions in: `test_indicators.py` (feature cols), `test_strategies.py` (strategy count), `test_evolution_integration.py` (strategy count), `test_model.py` ×2 (feature cols), `test_strategy_evolver.py` (PARAM_BOUNDS count), `test_intelligence.py` (provider count).
- **ruff import sort (I001)**: After adding new imports, run `./venv/bin/python -m ruff check --fix <file>` — ruff auto-fixes import order silently.
- **`datetime.UTC` alias (UP017)**: Use `from datetime import UTC` and `datetime.now(UTC)` — never `datetime.now(timezone.utc)`. ruff UP017 flags the latter; auto-fix with `ruff check --fix`.
- **TYPE_CHECKING for annotation-only imports**: When `from __future__ import annotations` is present, ruff F821 still flags annotation names not in module scope. Fix: `if TYPE_CHECKING: import pandas as pd`. Used in `decision_engine.py:_fetch_fitness_data` — pandas is a deferred in-method import.
- **ruff UP037 + F821 interaction**: `ruff --fix` removes string quotes from annotations (UP037). If the type isn't importable at module scope, this creates F821. After any `--fix` run, re-check for F821 errors.
- **REGIME_STRATEGY_MAP — never re-add MeanReversion to RANGING**: 5-year backtest (2021–2026) proved MeanReversion/VWAP/Grid destroy capital in crypto's persistent trending regimes. SOL alone spent $2,822 in fees on 5,274 trades (28% of capital). RANGING regime is now OBVDivergence-primary. Any weight change must pass `test_weights_sum_to_one` and show positive Sharpe on at least 1yr of walk-forward data before merging.
- **Backtest trailing stop vs. ATR**: The 1.5% default trailing stop sits inside normal BTC/ETH hourly ATR on quiet days (0.5–1.0%) and well inside volatile days (2–4%). This generated 60%+ of exits as premature noise stops. The production `Backtester` default remains 1.5% (no change); `scripts/backtest_5yr.py` overrides to 2.5% for research. If changing the default, update both and rerun the 5yr backtest.
- **`scripts/backtest_5yr.py` Binance rate-limits**: After a long run (~80 min), Binance may return HTTP 429. Add `time.sleep()` between pairs or use `generate_ohlcv()` from `demo_data.py` for quick validation. Do not use `n_bars=` kwarg — the signature is `generate_ohlcv(symbol, periods, ...)` (positional or keyword `periods=`).
- **`_combine` ONE_HOLD multiplier — current value 0.75**: `backtester.py:_combine()` applies a penalty when one side is HOLD (ML uncertain). At 0.75, a strategy needs conf ≥ 0.91 to clear `min_confidence=0.68`. History: 0.60 = structurally blocked all trades (max reachable 0.57 < 0.65 threshold); 0.80 = overtraded (2198 trades/3yr, −10.7%); 0.75 + 0.68 threshold = balanced (1397 trades, −7.2%). Do not lower below 0.65 or raise above 0.80 without re-running the 3yr BTC benchmark.
- **BreakoutStrategy triple-confirmation filter**: Three gates required for a BUY/SELL signal — (1) price must strictly *exceed* the N-bar high/low (not just touch within 0.2%); (2) `volume_ratio >= volume_mult` (1.5× default, same gate for both BB-break and N-bar-high paths); (3) `bar_range >= atr * atr_mult` (0.8× default) — rejects noise wicks. Before this fix, the `at_high` arm used `close >= recent_high * 0.998` + soft `vol_ratio > 1.2`, generating 77 false Breakout trades per 3yr period. After: 3 trades. The `atr_mult` is evolvable (bounds 0.5–1.5 in `strategy_evolver.py`).
- **Synthetic demo data vs. real data for backtesting**: `generate_ohlcv()` produces random OHLCV with no fixed seed — results differ every run and lack real BTC's trend persistence, volatility clustering, and regime duration. Use synthetic data only to check directional improvements (fewer trades, less drag). The authoritative evaluation is `scripts/backtest_5yr.py` against real Binance data. Never tune `MIN_CONFIDENCE` or `_combine` multiplier based on a single synthetic run.
- **MIN_CONFIDENCE 0.68 in backtest scripts**: `scripts/backtest_5yr.py` uses `MIN_CONFIDENCE = 0.68` (raised from 0.65 in Phase 5 Tuning v2). The live agent reads `MIN_CONFIDENCE` from `.env` (default 0.35 in `.env.example`). These are independent — changing `scripts/backtest_5yr.py` does not affect the live agent. If raising the live agent threshold, update `.env` and `test_config.py` expected default.
- **Per-pair trailing stop (Phase 6)**: `Config.get_trailing_stop_pct(symbol)` in `config.py` returns pair-specific stops — `SYMBOL_TRAILING_STOP_PCT = {"BTC/USDT": 0.025, "ETH/USDT": 0.025, "SOL/USDT": 0.040}`. SOL needs 4% due to ~2-3× higher hourly ATR. `risk_manager.py` calls `Config.get_trailing_stop_pct(self.symbol)`; `scripts/backtest_3yr.py` omits `trailing_stop_pct` and resolves per-pair from Config. Do NOT add a global `TRAILING_STOP_PCT` constant in backtest scripts — it shadows the per-pair logic.
- **RANGING confidence gate (Phase 6)**: `StrategyEngine.run()` in `strategies.py` contains an early-exit gate immediately after the primary signal runs: when `regime == MarketRegime.RANGING` and OBVDivergence `confidence < 0.68`, the engine returns HOLD without running the ensemble. Threshold history: 0.55 (Cycle 4) → 0.68 (Cycle 5). The gate is intentionally placed *before* the short-circuit check (conf > 0.80) so even high-conf RANGING signals below 0.68 are caught. If changing the threshold, re-run the 3yr backtest before merging.
- **RANGING no-trade zone (Phase 7)**: `StrategyEngine.run()` returns HOLD immediately for `MarketRegime.RANGING` before any strategy is invoked (`last_signals = {}`). This is intentional — 3yr walk-forward backtesting showed every strategy combination in RANGING produces negative returns (fee drag 0.15% round-trip > signal edge). Do NOT add RANGING trade logic back without a full 3yr backtest showing positive Sharpe. The RANGING entry in `REGIME_STRATEGY_MAP` is kept for reference and as the default fallback in `.get()` for unknown regimes.
- **RANGING 2-strategy weight math trap (Phase 6)**: With `weights = {OBVDivergence: 0.60, RSIDivergence: 0.40}` and RANGING gate at conf ≥ 0.68: minimum OBV pass score = 0.60 × 0.68 = 0.408; maximum RSI veto score = 0.40 × 1.0 = 0.40. Since 0.408 > 0.40, RSI can **never** veto an OBV BUY that passes the gate — the ensemble degenerates to "OBVDivergence always wins." This caused Ensemble(OBVDivergence) explosion (ETH 381 trades/−$390, SOL 409 trades/−$290 in 3yr). **Phase 7 fix**: restore 3-strategy RANGING (OBV 0.40 / RSI 0.35 / EMA 0.25) with gate at 0.68, OR use equal OBV 0.50 / RSI 0.50. General rule: for a 2-strategy ensemble with gate G on primary, ensure `W_secondary × 1.0 > W_primary × G` to preserve veto ability.
- **TRENDING_DOWN no-trade zone (Phase 8, Cycle 6)**: `StrategyEngine.run()` returns HOLD immediately for `MarketRegime.TRENDING_DOWN` — identical pattern to RANGING no-trade zone. Direction gates (C3) already suppressed BUY; remaining SELL (short) signals fired late into confirmed downtrends near bounce zones. The TRENDING_DOWN entry in `REGIME_STRATEGY_MAP` is kept for reference and as config data. Do NOT add trade logic back without a 3yr backtest showing positive Sharpe on short entries.
- **EMACrossover not in TRENDING_UP (Phase 8, Cycle 8)**: EMACrossover (9/21 EMA cross) was the 3rd lagging MA confirmation in TRENDING_UP alongside Momentum (SMA20/50) and Ichimoku (TK-cross). Triple-MA consensus fires when the trend is nearly complete — 20-40% WR, −$12 to −$62 drag across all 3yr pairs. TRENDING_UP `REGIME_STRATEGY_MAP` now has 4 strategies: Momentum(0.38), Ichimoku(0.27), OBVDivergence(0.15), RSIDivergence(0.20). EMACrossover remains in HIGH_VOLATILITY (Breakout + EMA + Momentum combo = top winner). Do NOT re-add EMACrossover to TRENDING_UP without a 3yr backtest showing improvement.
- **Regime-adaptive trailing stop (Phase 8, Cycle 7)**: `Backtester.REGIME_TRAILING_STOPS` class constant in `backtester.py` maps regime strings to trailing stop percentages: `trending_up=0.025` (2.5%), `high_volatility=0.020` (2.0%), others default to `self.trailing_stop_pct` (Config value, 1.5%). TRENDING_UP gets a wider stop so trend trades can breathe past normal hourly ATR (0.6-1.0%). `_get_trailing_stop_pct(regime)` helper is called from both `_open_position` (initial trailing level) and `_check_positions` (trailing ratchet update). The live agent uses `Config.get_trailing_stop_pct(symbol)` independently — backtester adaptive stops do not affect live trading.
- **ATR_FLOOR in StrategyEngine (Phase 8, Cycles 4-5)**: `StrategyEngine.ATR_FLOOR = 0.008` (0.8%). Gate fires before strategies run in TRENDING_UP only (TRENDING_DOWN is now a no-trade zone). Uses `df.iloc[-1].get("atr_pct", 1.0)` — defaults to 1.0 (pass) when `atr_pct` column is absent. Do not lower below 0.6% — 3yr backtest showed no improvement below this value for BTC/ETH.
- **Phase 8 3yr backtest final**: `scripts/backtest_3yr.py` with `MIN_CONFIDENCE=0.68`. C1 baseline (Phase 7 state): BTC −0.53%, ETH −1.26%, SOL −0.97%. C8 final: BTC −0.12%, ETH −0.54%, SOL −0.47% (+0.54pp avg improvement, 963 tests). Persistent loser (Ensemble(M,I,E)) eliminated. Dominant winner: `Ensemble(Breakout, EMACrossover, Momentum)` = HIGH_VOLATILITY regime Breakout-primary combo.
- **Phase 9 is now the primary pipeline**: `_run_pair_cycle()` no longer calls strategy_engine/ML/RL/combine/MTF. It only fetches data, computes indicators, manages position exits, detects regime, and caches `_last_df_ind[pair]`. All trade entry comes from `_run_phase9_cycle()`. Phase 8 code stays in repo for `scripts/backtest_3yr.py` and `scripts/backtest_5yr.py` — never call it from agent.py.
- **Phase 9 execution bridge**: `_run_phase9_cycle()` calls `_execute_trade()` when `decision.action == "trade"` using a stub `StrategySignal(strategy_name="phase9", suggested_sl_pct=Config.STOP_LOSS_PCT, suggested_tp_pct=Config.TAKE_PROFIT_PCT)`. ATR for position sizing comes from `self._last_df_ind.get(symbol)` (cached by `_run_pair_cycle`). All risk management (Kelly sizing, trailing stop, Telegram confirmation, TradeDB) is unchanged.
- **Phase 9 decision log**: `PHASE9_DECISION_LOG_PATH=data/phase9_decisions.jsonl` in `.env` enables JSONL audit log (every evaluate call, trade or reject). Unset = Python `_log.info()` only. `DecisionLogger` creates the file on first write.
- **Phase 9 fires selectively**: `evaluate()` requires ≥2 agreeing TriggerSignals or 1 high-urgency signal. `MomentumTrigger` fires only on RSI/MACD zero-crossings — not every bar. Normal market (no crossover) = `reject/no_valid_triggers`. This is correct behaviour; Phase 8 was firing on every small signal.
- **Phase 9 context.build() signature**: `ContextEngine.build(snapshot, funding_rate, net_whale_flow, oi_change_pct, price_change_pct)` — all four market-data args are required (pass `None` when unavailable). SwingAnalyzer uses `snapshot.df_4h`; other analyzers use the scalar args.
- **USE_PHASE9_PIPELINE flag**: Config default changed to `"true"`. The `.env` line `USE_PHASE9_PIPELINE=true` was removed (redundant). To disable Phase 9 for debugging, add `USE_PHASE9_PIPELINE=false` to `.env`.
- **Phase 9 1yr backtest**: `scripts/backtest_phase9_1yr.py` — fetches 1yr 1h+4h OHLCV, bar-by-bar simulation with 75min trigger TTL carry. Baseline (MomentumTrigger+Sweep only): BTC −0.25% (7 trades), ETH −0.45% (9 trades), SOL +0.62% (6 trades), avg −0.03%.
- **Phase 9 backtest with PullbackTrigger (2026-03-05)**: BTC −6.07% (109 trades), ETH −10.17% (151 trades), SOL −3.52% (116 trades), avg −6.59%. Trade count exploded ×15 because `PullbackTrigger` (urgency="normal") pairs with `LiquiditySweepTrigger` to satisfy the ≥2 consensus requirement. Win rates 26-40% with 0.20% round-trip fees = negative expectancy. **Root cause**: Sweep fires on almost any wick; pullback fires on almost any RSI neutral-zone touch in a trending market → too many low-quality consensus pairs. Needs tuning before production: options include (1) require ≥1 MomentumTrigger in consensus, (2) raise strength gate for pullback-only + sweep pairs, (3) add volume confirmation to PullbackTrigger, (4) raise `SCORE_THRESHOLD` in `decision.py`.
- **Funding extreme gate reason string**: `evaluate()` returns `reason="funding_extreme_blocks_direction"` (exact). Test assertions must use `==` not `in` — project convention is exact string matching for all Decision reason codes.
- **`LiquiditySweepTrigger` window exclusion**: Cluster window MUST use `df.iloc[:-1].tail(20)`, NOT `df.tail(20)`. Excluding the current bar is required — otherwise `current["low"] < zone_low` is structurally impossible because the current bar's low IS the minimum in the window, making sweep detection never fire.
- **`field(default_factory=list)` for mutable Position fields**: `Position.partial_tp_levels` and `partial_exits` use `field(default_factory=list)` — never `default=[]` in a dataclass. A bare `[]` default is shared across all instances (Python gotcha), causing cross-position contamination.
- **`partial_close()` only on PaperExecutor**: `LiveExecutor` does not implement `partial_close()`. Always guard with `hasattr(self.executor, "partial_close")` before calling in agent code — omitting the guard causes `AttributeError` in live mode at runtime.
- **Partial TP loop order in `_run_pair_cycle()`**: The partial TP loop (`check_partial_tp` → `partial_close`) must execute BEFORE `self.risk.check_positions()`. If `check_positions()` runs first and triggers a full-TP exit, the position is closed and the partial TP loop has nothing to act on. Always: partial TP → full position check.
- **`_btc_swing_bias` on TradingAgent**: `self._btc_swing_bias: str = "neutral"` is updated every BTC/USDT context refresh in `_run_phase9_cycle()`. SOL entries are gated: if `btc_swing_bias == "bearish"` and trigger is `"long"` (or vice-versa), `_run_phase9_cycle()` returns early before `evaluate()`. ETH has no gate — ETH can diverge from BTC. Initialised to `"neutral"` so first SOL cycle before any BTC context is never blocked.
- **`tradeable` numpy bool in ContextState**: `ContextEngine.build()` computes `tradeable` from pandas comparisons which yield `numpy.bool_`. Always wrap: `tradeable = bool(expr)`. Without the wrap, `assert ctx.tradeable is True` fails even when the value is truthy, because `numpy.bool_(True) is True` is `False` in Python's identity check.
- **`PAIR_POOL` must match `TRADING_PAIRS` in `.env`**: `_phase9_trigger_engines` is initialized only for `Config.TRADING_PAIRS` at startup. If `PAIR_POOL` includes extra pairs (default includes AVAX/BNB), PairScorer can promote them to `_active_pairs` and `_run_phase9_cycle()` hits `KeyError`. Fix: set `PAIR_POOL=BTC/USDT,ETH/USDT,SOL/USDT` in `.env` to match `TRADING_PAIRS`. A defensive guard in `_run_phase9_cycle()` now logs a warning and skips unknown pairs instead of crashing.
- **Phase 9 zero-trade kill chain**: Three simultaneous blockers produce 0 trades: (1) `SwingAnalyzer` required ≥2 EMA conditions for bias — neutral/ranging markets produced `allowed_directions=[]`, hard-blocking all trades at `no_allowed_directions`; (2) `ENABLE_ORDERBOOK=false` disables `OrderFlowTrigger` so only `MomentumTrigger` fires — `on_orderflow_update()` is never called from `_run_phase9_cycle()`; (3) original `evaluate()` required ≥2 agreeing triggers — structurally impossible with 1 active source. Fixed in Phase 9 tuning (see below).
- **Phase 9 current thresholds (post-tuning, commit b16ef00)**: `SCORE_THRESHOLD=0.40` in `decision.py` (was 0.50). `SwingAnalyzer` (`context/swing.py`) now has 4 tiers: ≥2-count strong bias `confidence=0.5+(count/3)*0.4`, 1-count weak bias `confidence=0.60`, 0-0/1-1 tie neutral `confidence=0.45` with `allowed=["long","short"]`. Single trigger bypass: `strength ≥ 0.75` allows trading without a second confirming source. Signal math for EU session weak-bias: `0.60×0.90×0.80=0.432 ≥ 0.40` ✓. Asia session weak-bias: `0.60×0.75×0.80=0.36 < 0.40` ✗ (correctly blocked during low-liquidity hours).
- **`test_decision_engine.py` UTC bug (fixed)**: `test_defensive_on_daily_loss` was failing because `engine._daily_reset_date = datetime.now()` (naive IST) was compared against `datetime.now(UTC)` (UTC-aware) inside `_check_safety()`, causing `.date()` mismatch after UTC midnight. Fixed: use `datetime.now(UTC)` in test setup. All 1211 tests now pass.
- **`USE_PHASE9_PERP=true` (now active in `.env`)**: Loads `LiquidationTrigger` (fires >$10M cascade, strength 0.60–0.95, `urgency="high"`, 15min TTL; long liquidations → short signal) and `FundingExtremeTrigger` (fires when funding >0.10%/8h → short, or <-0.05%/8h → long; strength 0.55–0.90, 60min TTL). Both are event-driven — in normal markets they don't fire (`funding_pressure=neutral`). When they do fire, `urgency="high"` routes confirmed trades to perp executor (blocked in `defensive` risk_mode). Loaded at `TriggerEngine.__init__()` via `os.getenv()` — requires full agent restart to toggle.
- **`GET /phase9/decisions`**: New endpoint in `api/routes/phase9.py` reads the last N entries from `PHASE9_DECISION_LOG_PATH` JSONL. Uses `_tail_jsonl(path, n)` — reads max 2000 lines (memory guard) then returns last n, newest-first. Returns `{"decisions":[], "count":0, "log_path":null}` when path not set. Dashboard page at `/decisions` (`dashboard/app/decisions/page.tsx`), auto-refreshes every 15 s. SWR key includes `?limit=N` so each limit value is a separate cache entry.
- **Adding new API route modules**: (1) Create `api/routes/newname.py` with `def create_router(store: DataStore) -> APIRouter:` factory, prefix, tags; (2) add `newname` to the import line in `api/server.py`; (3) add `app.include_router(newname.create_router(data_store))` in `create_app()`; (4) add method to `dashboard/lib/api.ts`; (5) verify: `./venv/bin/python -c "from api.routes.newname import create_router; print('ok')"` and `[r.path for r in create_app().routes]` shows the new path.
- **`providers_enabled` count in `/system/modules`**: The fallback block in `api/routes/status.py` originally only summed 5 of 10 intelligence providers. LLMSentimentProvider, CascadePredictor, FearGreedProvider have **no** `ENABLE_` flag and always run when the aggregator is initialized. `ENABLE_FUNDING_OI`/`ENABLE_LIQUIDATION` flags exist in `Config` and appear in `any_intelligence_enabled()`, but the providers' `get_signal()` methods don't check them. Fixed to sum all 10 (with `True` for the three always-on providers). With current `.env`, shows 9/10 (OrderBook disabled for Indian IPs).
- **`update_system_modules()` never called from `agent.py`**: `DataStore.update_system_modules()` is defined but the agent never calls it, so `/system/modules` always uses the static Config-based fallback in `api/routes/status.py`. Don't rely on the live data path for module status — it's always the fallback.
- **Testing file-reading routes with `tmp_path` + `patch.object`**: For routes that read from a Config-driven file path, use `tmp_path` for temp files and `with patch.object(Config, "ATTR_NAME", str(path)):` to override the class attribute per-test without touching `.env`. See `tests/test_phase9_route.py` (`_write_jsonl` helper, `TestTailJsonl` pure unit tests, `TestGetDecisionsEndpoint` endpoint tests).
