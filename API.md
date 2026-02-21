# API Reference — Crypto Trading Agent

Base URL: `http://localhost:8000`

## Authentication

When `API_AUTH_KEY` is set in `.env`, all endpoints except public paths require the header:

```
X-API-Key: <your-key>
```

Public paths (no auth required): `/health`, `/docs`, `/openapi.json`, `/redoc`, `/telegram/webhook`

WebSocket connections authenticate via first message after connecting (see WebSocket section below)

---

## Status Endpoints

### `GET /health`

Health check (no auth required).

```json
{
  "status": "healthy",
  "agent_running": true,
  "last_update": "2025-01-15T10:30:00.123456"
}
```

### `GET /status`

Full agent snapshot. Returns `{"status": "waiting"}` if the agent hasn't completed its first cycle yet. Otherwise returns the full snapshot dict with current prices, positions, signals, regime, etc.

### `GET /config`

Current configuration values (read from `.env`).

```json
{
  "exchange": "binance",
  "pair": "BTC/USDT",
  "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  "timeframe": "1h",
  "mode": "paper",
  "initial_capital": 1000,
  "max_position_pct": 0.02,
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0.05,
  "trailing_stop_pct": 0.015,
  "fee_pct": 0.001,
  "slippage_pct": 0.0005,
  "agent_interval_seconds": 300,
  "enable_websocket": true,
  "enable_trade_db": true,
  "notifications_enabled": true,
  "intelligence_enabled": true,
  "max_requests_per_minute": 1200,
  "max_orders_per_minute": 10
}
```

### `GET /system/modules`

Status of all system modules (WebSocket, notifications, trade DB, rate limiter, intelligence).

### `GET /system/rate-limiter`

Rate limiter usage statistics.

### `GET /notifications`

Recent notification log entries.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 100 | Max entries to return |

```json
{
  "notifications": [
    {"type": "trade", "channel": "telegram", "status": "sent", "timestamp": "..."}
  ],
  "count": 1
}
```

---

## Trading Endpoints

### `GET /trades`

Trade history.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 100 | Max trades to return (0 = all) |

```json
{
  "trades": [
    {
      "symbol": "BTC/USDT",
      "side": "long",
      "entry_price": 42000.0,
      "exit_price": 43050.0,
      "quantity": 0.001,
      "pnl_net": 0.95,
      "strategy": "MomentumStrategy",
      "entry_time": "2025-01-15T08:00:00",
      "exit_time": "2025-01-15T12:00:00"
    }
  ],
  "total": 1
}
```

### `GET /positions`

Currently open positions.

```json
{
  "positions": [
    {"symbol": "ETH/USDT", "side": "long", "entry_price": 2500.0, "quantity": 0.1}
  ],
  "count": 1
}
```

### `GET /equity`

Equity curve history.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 0 | Max points (0 = all) |

```json
{
  "equity": [
    {"equity": 1000.0, "timestamp": "2025-01-15T08:00:00"}
  ],
  "total_points": 1
}
```

### `GET /pnl-summary`

PnL breakdown by pair and by strategy, plus cumulative PnL curve.

```json
{
  "by_pair": {
    "BTC/USDT": {"trades": 5, "pnl": 12.50, "wins": 3, "losses": 2}
  },
  "by_strategy": {
    "MomentumStrategy": {"trades": 3, "pnl": 8.20, "wins": 2, "losses": 1}
  },
  "cumulative_pnl": [
    {"pnl": 2.50, "timestamp": "...", "symbol": "BTC/USDT"}
  ],
  "total_closed": 5
}
```

---

## Autonomous Endpoints

### `GET /autonomous/status`

Autonomous trading mode status (decision engine state, current regime, strategy weights).

### `GET /autonomous/events`

Event log for autonomous operations.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 50 | Max events to return |

### `POST /autonomous/halt`

Emergency kill switch — halts all trading immediately.

```json
// Request body (optional)
{ "reason": "Manual kill switch via API" }

// Response
{ "status": "halted", "reason": "Manual kill switch via API" }
```

### `POST /autonomous/resume`

Resume trading after an emergency halt.

```json
{ "status": "resumed" }
```

### `POST /autonomous/force-close`

Force close all open positions immediately.

```json
{ "status": "force_close_signaled" }
```

### `GET /autonomous/alerts`

Get decision engine alerts.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `unacknowledged` | bool | false | Only return unacknowledged alerts |

### `POST /autonomous/alerts/acknowledge`

Acknowledge all alerts.

---

## Backtest Endpoints

### `POST /backtest/run`

Run a backtest in a background thread. Only one backtest can run at a time.

```json
// Request body (all optional)
{
  "pair": "BTC/USDT",
  "scenario": "bull_run",
  "timeframe": "1h",
  "periods": 500,
  "mode": "all_pairs"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pair` | string | Single pair to backtest |
| `scenario` | string | Named scenario (see `/backtest/scenarios`) |
| `periods` | int | Number of bars (1-10000, default 500) |
| `mode` | string | `all_pairs`, `all_scenarios`, or `all_timeframes` |

### `GET /backtest/results`

Get all backtest results.

### `POST /backtest/clear`

Clear all stored backtest results.

### `GET /backtest/scenarios`

List available backtest scenarios.

---

## Intelligence Endpoints

### `GET /intelligence/signals`

Latest intelligence signals from all enabled providers.

Returns `{"status": "awaiting_first_cycle"}` if the agent hasn't run its first cycle, or `{"status": "not_enabled"}` if no intelligence providers are enabled.

### `GET /intelligence/providers`

List all intelligence providers and their enabled status.

```json
{
  "providers": [
    {"name": "onchain", "enabled": true},
    {"name": "orderbook", "enabled": true},
    {"name": "correlation", "enabled": true},
    {"name": "whale_tracker", "enabled": true},
    {"name": "news_sentiment", "enabled": true}
  ]
}
```

---

## Arbitrage Endpoints

### `GET /arbitrage/opportunities`

Current arbitrage opportunities detected by the scanner.

### `GET /arbitrage/fees`

Fee summary from the fee calculator.

---

## Risk Simulation Endpoints

### `GET /risk/simulation`

Get Monte Carlo simulation results. Returns `{"status": "not_run"}` if no simulation has been run.

### `POST /risk/monte-carlo`

Run a Monte Carlo simulation in a background thread. Only one simulation can run at a time.

```json
// Request body (all optional)
{
  "n_simulations": 10000,
  "n_days": 365,
  "initial_equity": 1000
}
```

| Field | Type | Limits | Default |
|-------|------|--------|---------|
| `n_simulations` | int | 1-100,000 | from config |
| `n_days` | int | 1-1,000 | from config |
| `initial_equity` | float | > 0 | INITIAL_CAPITAL from .env |

### `GET /risk/stress-tests`

List available stress test scenarios.

---

## Telegram Endpoints

### `POST /telegram/webhook`

Receives incoming Telegram updates (no API key auth, validated by `X-Telegram-Bot-Api-Secret-Token` header).

### `GET /telegram/status`

Telegram bot status.

```json
{
  "enabled": true,
  "webhook_url": "https://example.ngrok-free.dev/telegram/webhook",
  "trade_confirmation": true,
  "confirmation_timeout": 60
}
```

---

## WebSocket

### `ws://localhost:8000/ws`

Real-time event stream. The server pushes events as they occur.

**Authentication:** After connecting, send an auth message as the first message:
```json
{"type": "auth", "api_key": "<your-key>"}
```
The server waits up to 5 seconds for the auth message. If no auth is received or the key is invalid, the connection is closed with code 4001. When `API_AUTH_KEY` is not set, auth is skipped.

**Message format:**
```json
{
  "type": "snapshot",
  "data": { ... },
  "ts": "2025-01-15T10:30:00.123456"
}
```

**Event types:**
| Type | Trigger |
|------|---------|
| `snapshot` | Agent completes a cycle (every ~300s) |
| `equity` | New equity data point |
| `trade` | Trade opened or closed |
| `event` | Autonomous event logged |

**Client keepalive:** Send `"ping"` text message to receive `{"type":"pong"}`.

---

## Error Responses

### 401 Unauthorized
```json
{ "detail": "Invalid or missing API key. Set X-API-Key header." }
```

### 403 Forbidden (Telegram webhook)
```json
{ "detail": "Invalid secret" }
```

### 4001 WebSocket Close
WebSocket closed with code 4001 when API key is invalid.

---

## Interactive Docs

FastAPI auto-generates interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
