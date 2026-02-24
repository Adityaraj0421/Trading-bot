# CLAUDE.md — Crypto Trading Agent

## Project Overview

Autonomous crypto trading agent (Python 3.14 + Next.js 16) that trades BTC/USDT, ETH/USDT, SOL/USDT on Binance. Runs 10 strategies with ML ensemble, risk management, and real-time intelligence signals. Paper and live trading modes.

## Development History

| Phase | Summary |
|-------|---------|
| v1–v6 | Core trading loop, CCXT integration, paper/live modes, basic strategies |
| v7–v8 | Risk manager (Kelly sizing, drawdown gates), TradeDB persistence, Telegram bot, dashboard |
| v9.1 | 10th strategy (Ichimoku), Williams %R / CCI features (24 FEATURE_COLUMNS), pair scorer, ATR trailing + breakeven stops, 935 tests |
| Phase 4 | Code quality sweep: type hints + Google-style docstrings across all 37 production modules (37+ files, ~4,000 annotation additions, 0 ruff errors) |

## Architecture

```
agent.py (main loop, runs in background thread)
    ├── data_fetcher.py      → CCXT/Binance market data
    ├── indicators.py        → 15+ technical indicators (24 FEATURE_COLUMNS for ML)
    ├── decision_engine.py   → Strategy ensemble + ML signals
    ├── risk_manager.py      → Position sizing, stop-loss, daily limits
    ├── executor.py          → Paper/live trade execution
    └── notifier.py          → Telegram/Discord alerts

api/server.py (FastAPI, hosts agent in lifespan)
    ├── api/data_store.py    → Thread-safe agent↔API bridge
    ├── api/routes/          → REST endpoints
    ├── api/websocket_manager.py → Real-time push to dashboard
    └── telegram_bot.py      → Interactive Telegram commands

dashboard/ (Next.js React app, port 3000)
    ├── app/                 → Pages (trades, equity, autonomous, etc.)
    ├── components/          → Reusable UI (charts, tables, skeletons)
    └── lib/                 → API client, WebSocket hook, types

intelligence/ (10 signal modules, imported by decision_engine.py)
    ├── aggregator.py        → Orchestrates all intelligence sources
    ├── orderbook.py         → VPIN, CVD, spoofing detection, bid/ask walls
    ├── whale_tracker.py     → On-chain large-transaction flow
    ├── correlation.py       → Cross-asset correlation (yfinance optional)
    ├── funding_oi.py        → Perpetual funding rate + open interest
    ├── liquidation.py       → Estimated liquidation cascade levels
    ├── llm_sentiment.py     → LLM-based sentiment (Anthropic/OpenAI)
    ├── news_sentiment.py    → Reddit/RSS headline sentiment
    ├── onchain.py           → On-chain metrics + ML feature extraction
    └── cascade_predictor.py → Cascade risk score from liquidation data
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

## Testing

- **Framework**: pytest (tests/ directory, 44 test files, 935 tests)
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
- **LSTM model shape invalidation**: When `FEATURE_COLUMNS` count changes (e.g. adding new indicators), delete `data/agent_state_model.pkl` before restarting — LSTM input layer shape is fixed at train time and will crash on mismatch.
- **HOLD signal confidence**: Strategies return `confidence=0.3` for HOLD (not `0.0`). Only the data-guard (insufficient rows, e.g. `len(df) < 3`) returns `confidence=0.0`. Test assertions on HOLD should use `< 0.5`, not `== 0.0`.
- **Test count assertions**: When adding strategies or FEATURE_COLUMNS, update count assertions in: `test_indicators.py` (feature cols), `test_strategies.py` (strategy count), `test_evolution_integration.py` (strategy count), `test_model.py` ×2 (feature cols), `test_strategy_evolver.py` (PARAM_BOUNDS count).
- **ruff import sort (I001)**: After adding new imports, run `./venv/bin/python -m ruff check --fix <file>` — ruff auto-fixes import order silently.
