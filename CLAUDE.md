# CLAUDE.md — Crypto Trading Agent

## Project Overview

Autonomous crypto trading agent (Python 3.14 + Next.js 16) that trades BTC/USDT, ETH/USDT, SOL/USDT on Binance. Runs 9 strategies with ML ensemble, risk management, and real-time intelligence signals. Paper and live trading modes.

## Architecture

```
agent.py (main loop, runs in background thread)
    ├── data_fetcher.py      → CCXT/Binance market data
    ├── indicators.py        → 15+ technical indicators
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
| `agent.py` | Main trading loop (~1.4K lines). Called by api/server.py lifespan |
| `api/server.py` | FastAPI app factory, auth middleware, agent thread |
| `config.py` | All configuration from .env |
| `dashboard/app/page.tsx` | Dashboard home page |

## Testing

- **Framework**: pytest (tests/ directory, 42 test files, 890 tests)
- **Run**: `make test` or `./venv/bin/python -m pytest tests/ -v`
- **Linting**: `make lint` (ruff) — should report 0 errors
- **API tests**: `tests/test_api.py` — uses TestClient with auth header injection
- **Config tests**: `tests/test_config.py` — validates all config loading
- **Coverage**: `make test-coverage` (generates htmlcov/ report)

**Important**: The test client fixture in `test_api.py` automatically includes the API auth key when `API_AUTH_KEY` is set in .env. If you add new API tests, use the existing `client` fixture.

## Key Patterns

1. **DataStore bridge**: Agent thread writes to `api/data_store.py` (thread-safe with `threading.Lock`). API reads from it. Broadcast callback pushes to WebSocket clients.

2. **SWR + WebSocket hybrid**: Dashboard uses SWR for initial load + 30s fallback polling. WebSocket injects real-time updates into SWR cache via `mutate()`.

3. **Trade confirmation flow**: Agent thread calls `telegram_bot.request_confirmation()` → sends inline keyboard → blocks on `threading.Event.wait(60s)` → webhook callback sets decision.

4. **Auth middleware**: `API_AUTH_KEY` in .env enables auth. Header: `X-API-Key`. Public paths: `/health`, `/docs`, `/telegram/webhook`. Tests auto-inject the key.

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
- **Type hints**: All public methods are annotated (Python 3.12+ syntax)
- **Docstrings**: Google-style on all public methods and classes
- **Logging**: All modules use `_log = logging.getLogger(__name__)` — no stray `print()` in production code (except CLI output in backtester/agent banner)

## Common Gotchas

- `.env` must exist — config.py reads from it via python-dotenv
- `API_AUTH_KEY` affects both API access and tests — the test fixture handles this
- Dashboard `.env.local` has `NEXT_PUBLIC_API_KEY` — must match `API_AUTH_KEY`
- Agent runs as daemon thread inside FastAPI lifespan — not a separate process
- Telegram webhook URL changes with every ngrok restart — update `.env` accordingly
- `venv/` must be activated or on PATH for `make` commands to find Python packages
