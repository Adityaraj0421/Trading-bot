# Architecture — Crypto Trading Agent

## System Overview

The system is a three-tier application: a Python trading agent, a FastAPI REST/WebSocket server, and a Next.js React dashboard.

```
┌──────────────────────────────────────────────────────────────┐
│                       User / Browser                         │
│                    http://localhost:3000                      │
└───────────────────────────┬──────────────────────────────────┘
                            │  SWR polling + WebSocket
┌───────────────────────────▼──────────────────────────────────┐
│                  Next.js Dashboard (port 3000)                │
│  app/           Pages (trades, equity, autonomous, ...)       │
│  components/    ErrorBanner, PageSkeleton, EquityChart, ...   │
│  lib/           api.ts (fetch client), useWebSocket.ts        │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTP REST + WebSocket /ws
┌───────────────────────────▼──────────────────────────────────┐
│                  FastAPI Server (port 8000)                    │
│  api/server.py        App factory, auth middleware, lifespan  │
│  api/data_store.py    Thread-safe agent↔API bridge            │
│  api/routes/          REST endpoints (9 route modules)        │
│  api/websocket_manager.py  Broadcast to connected clients     │
└───────────────────────────┬──────────────────────────────────┘
                            │  Daemon thread (started in lifespan)
┌───────────────────────────▼──────────────────────────────────┐
│                  Trading Agent (agent.py)                      │
│  DataFetcher → Indicators → Strategies → DecisionEngine       │
│  → RiskManager → Executor → StateManager                      │
│                                                                │
│  Intelligence modules, arbitrage scanner, self-healer,         │
│  strategy evolver, meta-learner, notifier, trade DB            │
└──────────────────────────────────────────────────────────────┘
        │                        │                    │
   Binance API             Telegram Bot          SQLite DB
   (CCXT + WS)           (webhook + cmds)      (trades.db)
```

## Agent Thread Architecture

The agent runs in a single daemon thread, started by `api/server.py`'s lifespan handler. Each iteration (default: 300s) executes this pipeline:

```
1. Fetch market data          → data_fetcher.py (CCXT candles + WebSocket tick)
2. Compute indicators         → indicators.py (RSI, MACD, Bollinger, ATR, ...)
3. Check market regime        → regime_detector.py (HMM-based)
4. Run strategy ensemble      → strategies.py (9 strategies × 3 pairs)
5. Gather intelligence        → intelligence/aggregator.py (9 providers)
6. ML model prediction        → model.py (scikit-learn ensemble)
7. Decision engine            → decision_engine.py (combine signals + safeguards)
8. Risk management            → risk_manager.py (position sizing, limits)
9. Execute trades             → executor.py (paper or live)
10. Update state + notify     → state_manager.py, notifier.py, trade_db.py
```

## Module Map

### Core Trading (`/`)

| Module | Purpose |
|--------|---------|
| `agent.py` | Main trading loop, orchestrates all modules |
| `config.py` | Loads all configuration from `.env` |
| `data_fetcher.py` | Market data via CCXT (candles, tickers) |
| `indicators.py` | Technical indicators (RSI, MACD, Bollinger, ATR, etc.) |
| `model.py` | ML model training and signal prediction |
| `strategies.py` | 9 trading strategies (mean reversion, momentum, breakout, etc.) |
| `risk_manager.py` | Position sizing, stop-loss, take-profit, daily limits |
| `executor.py` | PaperExecutor and LiveExecutor (CCXT order placement) |
| `portfolio.py` | Multi-pair portfolio management |
| `regime_detector.py` | HMM-based market regime detection |
| `sentiment.py` | Legacy sentiment analysis |
| `state_manager.py` | JSON state persistence between restarts |
| `logger.py` | Structured logging setup |

### Intelligence (`/intelligence/`)

All providers produce a score (-1.0 to 1.0) and a direction (bullish/bearish/neutral). The aggregator combines them into a single `adjustment_factor` (0.5 to 1.5).

| Module | Source |
|--------|--------|
| `aggregator.py` | Combines all providers into one signal |
| `onchain.py` | On-chain transaction metrics |
| `orderbook.py` | Exchange order book depth analysis |
| `correlation.py` | Cross-asset correlation tracking |
| `whale_tracker.py` | Large transaction monitoring |
| `news_sentiment.py` | News headline NLP scoring |
| `llm_sentiment.py` | LLM-based sentiment analysis |
| `funding_oi.py` | Funding rates + open interest |
| `liquidation.py` | Liquidation cascade detection |
| `cascade_predictor.py` | Prediction of cascading liquidations |

### Arbitrage (`/arbitrage/`)

| Module | Purpose |
|--------|---------|
| `opportunity_detector.py` | Scans for price discrepancies |
| `triangular_arbitrage.py` | Three-leg triangular arbitrage scanning |
| `funding_arbitrage.py` | Funding rate arbitrage detection |
| `execution_engine.py` | Executes arbitrage trades |
| `price_monitor.py` | Real-time price tracking |
| `fee_calculator.py` | Fee estimation for net profit |
| `latency_tracker.py` | Exchange latency monitoring |

### Risk Simulation (`/risk_simulation/`)

| Module | Purpose |
|--------|---------|
| `monte_carlo.py` | Monte Carlo portfolio simulation |
| `var_calculator.py` | Value at Risk calculations |
| `scenarios.py` | Stress test scenario definitions |
| `visualizer.py` | Risk visualization helpers |

### Autonomous Modules (`/`)

| Module | Purpose |
|--------|---------|
| `decision_engine.py` | Combines ML + strategy signals, kill switch, alerts |
| `self_healer.py` | Circuit breaker, error recovery |
| `strategy_evolver.py` | Genetic algorithm for strategy parameter evolution |
| `meta_learner.py` | Learns which strategies work in which regimes |
| `auto_optimizer.py` | Automatic hyperparameter optimization |
| `rl_ensemble.py` | Reinforcement learning ensemble weighting |

### Production Infrastructure (`/`)

| Module | Purpose |
|--------|---------|
| `websocket_streamer.py` | Connects to Binance WebSocket for live prices |
| `notifier.py` | Multi-channel notifications (Telegram, Discord, Email) |
| `telegram_bot.py` | Interactive Telegram bot (commands + trade confirmations) |
| `trade_db.py` | SQLite database for trade history and analytics |
| `graceful_shutdown.py` | Signal handling + rate limiter |
| `backtest_runner.py` | Multi-pair, multi-scenario backtesting |
| `scenarios.py` | Backtest scenario definitions |

### API (`/api/`)

| Module | Purpose |
|--------|---------|
| `server.py` | FastAPI app factory, auth middleware, lifespan |
| `data_store.py` | Thread-safe in-memory bridge (agent writes, API reads) |
| `websocket_manager.py` | Manages WebSocket connections and broadcasts |
| `routes/status.py` | Health, status, config, system modules |
| `routes/trading.py` | Trades, positions, equity, PnL |
| `routes/autonomous.py` | Autonomous mode, kill switch, alerts |
| `routes/backtest.py` | Run and view backtests |
| `routes/intelligence.py` | Intelligence signals and providers |
| `routes/arbitrage.py` | Arbitrage opportunities and fees |
| `routes/risk.py` | Monte Carlo, stress tests |
| `routes/telegram.py` | Webhook receiver, bot status |
| `routes/websocket.py` | WebSocket endpoint (`/ws`) |

### Dashboard (`/dashboard/`)

Next.js 16 React app with Tailwind CSS.

| Path | Purpose |
|------|---------|
| `app/page.tsx` | Home (overview metrics, equity chart) |
| `app/trades/page.tsx` | Trade history table |
| `app/equity/page.tsx` | Full equity chart |
| `app/autonomous/page.tsx` | Autonomous mode status + events |
| `app/intelligence/page.tsx` | Intelligence signal display |
| `app/notifications/page.tsx` | Notification log |
| `app/backtest/page.tsx` | Backtest results |
| `app/arbitrage/page.tsx` | Arbitrage opportunities |
| `app/risk/page.tsx` | Risk simulation results |
| `components/ErrorBanner.tsx` | Reusable error display |
| `components/PageSkeleton.tsx` | Loading skeleton states |
| `components/EquityChart.tsx` | Recharts equity chart |
| `lib/api.ts` | Fetch wrapper with auth |
| `lib/useWebSocket.ts` | WebSocket hook with SWR cache injection |

## Data Flow

### Agent → Dashboard (real-time)

```
Agent thread
    │ calls DataStore.update_snapshot(data)
    ▼
DataStore (thread-safe, via Lock)
    │ calls broadcast_callback("snapshot", data)
    ▼
WebSocketManager.broadcast_sync()
    │ schedules on FastAPI event loop (asyncio.run_coroutine_threadsafe)
    ▼
WebSocket /ws endpoint
    │ sends JSON to all connected clients
    ▼
Dashboard useWebSocket hook
    │ receives message, updates SWR cache via mutate()
    ▼
React components re-render
```

### Telegram Trade Confirmation

```
Agent → telegram_bot.request_confirmation(trade)
    │ sends inline keyboard (Approve / Reject)
    │ blocks on threading.Event.wait(timeout=60s)
    ▼
User taps button in Telegram
    │
    ▼
Telegram → ngrok → /telegram/webhook
    │ route handler calls telegram_bot.handle_update(body)
    │ callback_query handler sets event + decision
    ▼
Agent unblocks → proceeds or cancels trade
```

## Thread Model

The application uses three main threads:

1. **Main thread**: FastAPI/uvicorn event loop (async)
2. **Agent thread**: Daemon thread running `TradingAgent.run()` (synchronous, blocking)
3. **Backtest/Monte Carlo threads**: On-demand daemon threads for long-running computations

The `DataStore` class is the synchronization point between threads, using a single `threading.Lock` for all read/write operations. The `WebSocketManager.broadcast_sync()` bridges the synchronous agent thread to the async FastAPI event loop via `asyncio.run_coroutine_threadsafe()`.

## Deployment Options

| Method | Command | Use Case |
|--------|---------|----------|
| Development | `make dev` | Local development with hot reload |
| Overnight | `make start` / `make stop` | Background headless runs |
| Docker | `docker compose up` | Containerized deployment |
| Manual | `python -m api.server` | Direct server start |
