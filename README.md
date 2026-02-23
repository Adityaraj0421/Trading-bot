# Crypto Trading Agent v7

An autonomous, AI-powered crypto trading bot with a real-time web dashboard. Trades BTC/USDT, ETH/USDT, and SOL/USDT on Binance using a 9-strategy ensemble, ML models, risk management, and live intelligence signals.

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Configure
cp .env.example .env
# Edit .env — paper trading works without exchange API keys

# 3. Start API + Dashboard
make dev
# API: http://localhost:8000
# Dashboard: http://localhost:3000
```

## Architecture

```
agent.py              Main trading loop (runs as daemon thread in FastAPI lifespan)
├── data_fetcher.py   CCXT/Binance OHLCV + ticker data, demo fallback
├── indicators.py     15+ technical indicators (RSI, MACD, BBands, ATR, etc.)
├── decision_engine.py Strategy ensemble, ML signals, strategy evolution
├── risk_manager.py   Position sizing (Kelly/ATR), stop-loss, daily limits
├── executor.py       Paper simulator / live order placement (CCXT)
├── notifier.py       Telegram + Discord + email alerts
└── trade_db.py       SQLite persistence for trades and equity history

api/server.py         FastAPI app — hosts agent thread, auth middleware
├── api/data_store.py Thread-safe agent↔API bridge (in-memory + broadcast)
├── api/routes/       REST endpoints (trading, status, backtest, autonomous…)
├── api/websocket_manager.py Real-time WebSocket push to dashboard
└── telegram_bot.py   Interactive Telegram commands + trade confirmation

dashboard/            Next.js 16 React app (port 3000)
├── app/              Pages: overview, trades, equity, autonomous, backtest…
├── components/       Charts, tables, skeletons, error banners
└── lib/              API client, SWR+WebSocket hook, TypeScript types
```

## Trading Strategies

The agent runs 9 strategies in a weighted ensemble, selected by market regime:

| Strategy | Best Regime | Approach |
|----------|------------|----------|
| Momentum | Trending | MA alignment + MACD crossovers |
| MeanReversion | Ranging | Bollinger + RSI oversold/overbought |
| Breakout | Ranging→Trending | Volatility expansion after BB squeeze |
| Grid | Low-vol Ranging | Oscillation between grid levels |
| Scalping | Any | Quick reversals on pin bars + RSI extremes |
| Sentiment | Any | Contrarian at Fear/Greed extremes |
| MLStrategy | Any | Pure ML signal (XGBoost/LSTM ensemble) |
| TrendFollowing | Trending | Higher-timeframe confirmation |
| Arbitrage | Any | Cross-pair spread capture |

## Regime Detection

The agent detects 5 market regimes and adapts strategy weights dynamically:
- **Trending Up / Down** — Momentum dominant
- **Ranging** — MeanReversion dominant
- **High Volatility** — Breakout + Scalping
- **Accumulation** — Long-biased, tight sizing

Detection uses HMM + trend analysis + ATR volatility clustering.

## Intelligence Signals

Optional external signals that adjust confidence scores:

| Signal | Source |
|--------|--------|
| On-chain data | Whale alerts, exchange flows |
| Order book | Bid/ask imbalance, spread |
| Correlation | Cross-pair correlation risk |
| Whale tracking | Large wallet movements |
| News NLP | LLM sentiment (OpenAI/Anthropic) |

## Configuration (`.env`)

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `EXCHANGE_ID` | `binance` | CCXT exchange ID |
| `TRADING_PAIRS` | `BTC/USDT,ETH/USDT,SOL/USDT` | Comma-separated pairs |
| `TIMEFRAME` | `1h` | Candle timeframe |
| `INITIAL_CAPITAL` | `1000` | Starting capital (USDT) |
| `MAX_POSITION_PCT` | `0.05` | Max 5% per trade |
| `STOP_LOSS_PCT` | `0.02` | Default stop-loss |
| `TAKE_PROFIT_PCT` | `0.05` | Default take-profit |
| `MIN_CONFIDENCE` | `0.35` | Minimum signal confidence to trade |
| `API_AUTH_KEY` | _(empty)_ | Enables REST + WebSocket auth |
| `TELEGRAM_BOT_TOKEN` | _(empty)_ | Enables Telegram alerts |
| `ENABLE_TRADE_DB` | `true` | SQLite trade persistence |

## Risk Management

- Position sizing via Kelly Criterion (capped at `MAX_POSITION_PCT`)
- Regime-adaptive SL/TP multipliers (wider in trending, tighter in ranging)
- SL/TP validation guard — prevents self-triggering zero-distance stops
- Tiered drawdown protocol (reduce sizing → pause → halt at thresholds)
- Volatility targeting: adjusts size to target 15% annualized portfolio vol
- Correlation-adjusted sizing for multi-pair portfolios
- Daily loss limit (auto-stops trading when breached)

## Development

```bash
make dev            # Start API (port 8000) + Dashboard (port 3000)
make stop           # Kill all processes
make test           # Run 890 pytest tests
make test-coverage  # Tests + HTML coverage report (htmlcov/)
make lint           # ruff check (must be 0 errors)
make format         # ruff format
make build          # Build dashboard for production
make help           # Show all targets
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check — 503 if agent stale >5min |
| `GET /status` | Full agent snapshot (cycle, prices, PnL, positions) |
| `GET /trades` | Trade history (SQLite + in-memory fallback) |
| `GET /equity` | Equity curve data points |
| `GET /positions` | Currently open positions |
| `GET /pnl-summary` | Win rate, avg win/loss, total PnL |
| `GET /autonomous/events` | Autonomous mode event log |
| `POST /backtest/run` | Trigger background backtest |
| `GET /config` | Current agent configuration |
| `WS /ws` | Real-time WebSocket stream (auth via first message) |

Auth: set `X-API-Key` header when `API_AUTH_KEY` is configured.

## Persistence

- **`data/trades.db`** — SQLite: all trades, equity snapshots, open positions
- **`data/agent_state.json`** — Agent state (capital, cycle count, strategy weights)
- **`data/agent_state_model.pkl`** — Trained ML model weights
- **`data/agent_state_autonomous.json`** — Autonomous mode state

Reset state: delete those files — agent restarts with `INITIAL_CAPITAL` from `.env`.

## Disclaimer

For educational and research purposes only. Crypto trading involves substantial risk of loss. Always start with paper trading (`TRADING_MODE=paper`). Never invest more than you can afford to lose. This is not financial advice.
