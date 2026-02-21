# Crypto Trading Agent v6.0 — Full Enhancement Plan

## Context

The crypto trading agent v5.0 is a mature Python system with 6 strategies, HMM regime detection, ML ensemble, genetic strategy evolution, self-healing, and meta-learning. It runs locally, trades single pairs on a single exchange, has no dashboard, no Docker deployment, no on-chain intelligence, no multi-exchange arbitrage, and no risk simulation. This plan adds all 6 enhancements in a phased approach, transforming it into a production-grade multi-exchange trading platform with real-time monitoring.

**User constraints**: Local machine deployment, React dashboard (beginner), free APIs only, Binance + Coinbase + Kraken, beginner at statistics.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                       │
├──────────────┬──────────────┬───────────────────────────────┤
│  Trading     │  API Server  │  React Dashboard              │
│  Agent       │  (FastAPI)   │  (Next.js)                    │
│  (agent.py)  │  (api/)      │  (dashboard/)                 │
├──────────────┴──────────────┴───────────────────────────────┤
│                   Shared Volumes                             │
│  /data  (state, logs, models)                               │
│  /config (.env, exchange keys)                              │
└─────────────────────────────────────────────────────────────┘
```

Three containers: (1) Trading agent runs cycles, (2) FastAPI serves data to dashboard, (3) Next.js dashboard for monitoring. All share a `/data` volume for state files.

---

## Phase 1: Data Store + API Layer (Foundation)

**Why first**: Every other enhancement needs a way to expose data. The API layer is the backbone that the dashboard, backtesting results, intelligence signals, and arbitrage status all feed into.

### New Files
- `api/` — FastAPI application directory
  - `api/server.py` — FastAPI app, CORS, lifespan (starts agent in background thread)
  - `api/routes/status.py` — GET `/status`, `/health`, `/config`
  - `api/routes/trading.py` — GET `/trades`, `/positions`, `/equity`
  - `api/routes/autonomous.py` — GET `/autonomous/status`, `/autonomous/events`
  - `api/routes/backtest.py` — POST `/backtest/run`, GET `/backtest/results`
  - `api/routes/intelligence.py` — GET `/intelligence/signals` (Phase 4)
  - `api/routes/arbitrage.py` — GET `/arbitrage/opportunities` (Phase 5)
  - `api/routes/risk.py` — GET `/risk/simulation`, POST `/risk/monte-carlo` (Phase 6)
  - `api/data_store.py` — Thread-safe in-memory store + periodic JSON flush

### Modify
- `agent.py` — At end of `run_cycle()`, push snapshot to `data_store`
- `config.py` — Add `API_PORT`, `API_HOST`, `ENABLE_API` settings
- `requirements.txt` — Add `fastapi`, `uvicorn[standard]`, `websockets`

### Key Design: DataStore
```python
class DataStore:
    """Thread-safe bridge between agent and API."""
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = {}       # Latest cycle snapshot
        self._equity_history = [] # Time series for chart
        self._trade_log = []      # Completed trades
        self._events = []         # Autonomous events

    def update_snapshot(self, agent): ...  # Called each cycle
    def get_snapshot(self) -> dict: ...    # Called by API routes
```

The agent writes, the API reads. No coupling between them.

---

## Phase 2: Docker Deployment

**Why second**: Once the API exists, we containerize everything for reliable 24/7 local operation.

### New Files
- `Dockerfile` — Python 3.11-slim, install deps, copy code, run agent+API
- `docker-compose.yml` — 3 services: agent, api, dashboard
- `docker/entrypoint.sh` — Startup script with health check
- `.dockerignore` — Exclude state files, logs, .env, __pycache__

### Docker Compose Services
```yaml
services:
  agent:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data          # State persistence
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      retries: 3

  dashboard:
    build: ./dashboard
    ports:
      - "3000:3000"
    depends_on:
      - agent
```

### Modify
- `config.py` — Add `DATA_DIR` env var (default `./data`), route all state files there
- `state_manager.py` — Use `Config.DATA_DIR` for file paths
- `logger.py` — Write logs to `Config.DATA_DIR/agent.log`

### Key Design: Single Container (Recommended)
For simplicity on local machine, run agent + FastAPI in ONE container using uvicorn with the agent running in a background thread. The dashboard is a separate container (or just `npm run dev` locally during development). This avoids inter-container networking complexity.

---

## Phase 3: React Dashboard

**Why third**: With API + Docker running, the dashboard can display live data.

### New Directory: `dashboard/`
```
dashboard/
├── package.json
├── next.config.js
├── tsconfig.json
├── Dockerfile
├── app/
│   ├── layout.tsx          — Root layout (dark theme, sidebar nav)
│   ├── page.tsx            — Home: overview cards (capital, PnL, win rate, status)
│   ├── trades/page.tsx     — Trade history table + filters
│   ├── equity/page.tsx     — Equity curve chart (recharts)
│   ├── autonomous/page.tsx — Strategy evolution, meta-learning, health
│   ├── backtest/page.tsx   — Run backtests, view results (Phase 3b)
│   ├── intelligence/page.tsx — Intelligence signals (Phase 4)
│   ├── arbitrage/page.tsx  — Arbitrage monitor (Phase 5)
│   └── risk/page.tsx       — Monte Carlo results (Phase 6)
├── components/
│   ├── MetricCard.tsx      — Reusable stat card
│   ├── EquityChart.tsx     — Line chart (recharts)
│   ├── TradeTable.tsx      — Sortable trade history
│   ├── StatusBadge.tsx     — Decision state indicator
│   ├── RegimeIndicator.tsx — Market regime visual
│   └── Sidebar.tsx         — Navigation
├── lib/
│   ├── api.ts              — Fetch wrapper for FastAPI endpoints
│   └── types.ts            — TypeScript interfaces matching API responses
└── public/
```

### Tech Choices (Beginner-Friendly)
- **Next.js 14 App Router** — File-based routing, no config needed
- **Tailwind CSS** — Utility classes, no CSS files to manage
- **Recharts** — Simple React charting library
- **SWR** — Auto-refreshing data fetching (polls API every 5s)
- No state management library — SWR handles server state, `useState` for local

### Dashboard Pages
1. **Home** — 4 metric cards (Capital, Total PnL, Win Rate, Decision State) + mini equity chart + recent trades
2. **Equity** — Full equity curve with drawdown overlay, zoom/pan
3. **Trades** — Table with entry/exit prices, PnL, strategy, regime, exit reason
4. **Autonomous** — Strategy evolution progress, meta-learner weights, health metrics, event log
5. **Backtest** — Form to configure and run backtests, results display (Phase 3b)
6. **Intelligence** — On-chain signals, order book depth, correlations (Phase 4)
7. **Arbitrage** — Cross-exchange price table, opportunities (Phase 5)
8. **Risk** — Monte Carlo visualization, VaR/CVaR (Phase 6)

### Data Flow
```
Agent → DataStore → FastAPI → SWR (poll 5s) → React Components
```

No WebSocket initially — polling is simpler and sufficient for 5-minute trading cycles. Can upgrade to WebSocket later if needed.

---

## Phase 4: Extended Backtesting

**Why fourth**: Dashboard can now display backtest results. Extend the backtester before adding new intelligence signals.

### Modify
- `backtester.py` — Major extension:
  - Add `run_multi_pair(pairs: list[str])` method
  - Add `run_scenario(scenario: str)` for predefined market conditions
  - Add `run_multi_timeframe(timeframes: list[str])` for TF comparison
  - Store results in DataStore for dashboard display

### New Files
- `scenarios.py` — Predefined market scenarios:
  - `bull_run` — Synthetic trending-up data
  - `bear_market` — Sustained downtrend
  - `sideways_chop` — Range-bound with false breakouts
  - `flash_crash` — Sudden 30%+ drop and recovery
  - `black_swan` — COVID-style crash (March 2020 pattern)
  - `accumulation` — Low volatility consolidation
- `backtest_runner.py` — Orchestrates multi-pair, multi-scenario runs

### API Routes (already scaffolded in Phase 1)
- `POST /backtest/run` — Trigger backtest with params (pair, scenario, timeframe)
- `GET /backtest/results` — Retrieve all saved backtest results
- `GET /backtest/compare` — Side-by-side comparison of runs

### Multi-Pair Approach
The existing `DataFetcher.fetch_ohlcv(symbol=...)` already accepts any pair. The backtester needs to:
1. Iterate over pairs list
2. Run independent backtests per pair
3. Aggregate results with cross-pair correlation analysis

---

## Phase 5: Additional Intelligence (Free APIs)

**Why fifth**: New signals feed into existing strategies and ML model, improving trading decisions before going multi-exchange.

### New Files
- `intelligence/` — New module directory
  - `intelligence/__init__.py`
  - `intelligence/onchain.py` — On-chain analytics
  - `intelligence/orderbook.py` — Order book depth analysis
  - `intelligence/correlation.py` — Stock index correlation
  - `intelligence/whale_tracker.py` — Whale wallet monitoring
  - `intelligence/news_sentiment.py` — NLP news sentiment
  - `intelligence/aggregator.py` — Combines all signals into unified score

### Free Data Sources
| Signal | Source | Rate Limit | Update Freq |
|--------|--------|-----------|-------------|
| On-chain (BTC hash rate, active addresses) | Blockchain.com API | Free, no key | 10min |
| Fear & Greed Index | Already implemented (`sentiment.py`) | Free | 24h |
| Order book depth | CCXT `fetch_order_book()` — already exists in `data_fetcher.py` | Per exchange | Real-time |
| S&P 500 / NASDAQ correlation | Yahoo Finance (`yfinance` lib) | Free | 1min during market hours |
| Whale alerts | Whale Alert free tier (10 req/min) | Free API key | Real-time |
| News sentiment | CryptoPanic free API | Free API key | 5min |
| Reddit sentiment | Reddit JSON API (`.json` suffix) | No key needed | 15min |

### Integration with Existing Code
```python
# In agent.py run_cycle(), after step 6 (regime detection):
intelligence = self.intelligence_aggregator.get_signals()
# Adjust confidence: final_confidence *= intelligence.adjustment_factor
```

The aggregator produces a single `adjustment_factor` (0.5 to 1.5) and a `bias` (bullish/bearish/neutral) that modifies the existing signal confidence. This is minimally invasive — doesn't change the strategy architecture.

### Modify
- `agent.py` — Import and call `IntelligenceAggregator` in `run_cycle()`
- `decision_engine.py` — Accept intelligence signals in `orchestrate()`
- `config.py` — Add toggles: `ENABLE_ONCHAIN`, `ENABLE_WHALE_TRACKING`, `ENABLE_NEWS_NLP`, etc.
- `requirements.txt` — Add `yfinance`, `textblob` or use existing `sklearn` for NLP

### NLP Approach (Free, Lightweight)
Use `sklearn`'s `TfidfVectorizer` + `LogisticRegression` (already a dependency) trained on a small hand-labeled dataset of crypto headlines. No need for heavy NLP libraries. Start simple:
1. Fetch headlines from CryptoPanic
2. Extract keywords (pump, dump, hack, regulation, partnership, etc.)
3. Score as positive/negative/neutral
4. Aggregate into sentiment score

---

## Phase 6: Multi-Exchange Arbitrage

**Why sixth**: Requires stable multi-exchange connections (Phase 2 Docker) and intelligence layer (Phase 5 order book analysis).

### New Files
- `arbitrage/` — New module directory
  - `arbitrage/__init__.py`
  - `arbitrage/price_monitor.py` — Concurrent price fetching from 3 exchanges
  - `arbitrage/opportunity_detector.py` — Find profitable spread differences
  - `arbitrage/fee_calculator.py` — Per-exchange fee schedules, withdrawal costs
  - `arbitrage/execution_engine.py` — Simultaneous buy/sell execution
  - `arbitrage/latency_tracker.py` — API response time monitoring

### Design
```python
class ArbitrageMonitor:
    """Monitors price spreads across Binance, Coinbase, Kraken."""

    def __init__(self, exchanges: dict[str, ccxt.Exchange]):
        self.exchanges = exchanges  # {"binance": ..., "coinbase": ..., "kraken": ...}

    async def scan_opportunities(self, pairs: list[str]) -> list[ArbitrageOpportunity]:
        """Fetch prices concurrently, identify profitable spreads."""
        # Use asyncio.gather for concurrent API calls
        # Filter by: spread > fees + slippage + withdrawal cost
        pass
```

### Key Considerations
- **Latency**: Use async fetching (`aiohttp` already in requirements). Target <500ms per scan.
- **Fee-Adjusted**: Only flag opportunities where `spread > buy_fee + sell_fee + withdrawal_fee + slippage_buffer`
- **Capital Pre-positioning**: Keep funds on all 3 exchanges to avoid withdrawal delays
- **Paper Mode First**: Simulate arbitrage execution before going live
- **Transfer costs**: Include blockchain transfer fees and times in profit calculation

### Modify
- `config.py` — Add `ARBITRAGE_ENABLED`, per-exchange API key configs:
  ```
  BINANCE_API_KEY, BINANCE_API_SECRET
  COINBASE_API_KEY, COINBASE_API_SECRET
  KRAKEN_API_KEY, KRAKEN_API_SECRET
  ```
- `data_fetcher.py` — Add `MultiExchangeFetcher` that manages 3 exchange connections
- `agent.py` — Add arbitrage scan step in `run_cycle()` (optional, toggled by config)

### Integration
Arbitrage runs as a parallel concern — it doesn't replace the main trading strategy. The agent's cycle becomes:
1. Main strategy analysis (existing)
2. Arbitrage scan (new, optional)
3. Execute whichever has higher expected value

---

## Phase 7: Risk Simulation (Monte Carlo)

**Why last**: Needs historical trade data (Phase 4 backtesting), intelligence signals (Phase 5), and arbitrage data (Phase 6) to simulate realistic scenarios.

### New Files
- `risk_simulation/` — New module directory
  - `risk_simulation/__init__.py`
  - `risk_simulation/monte_carlo.py` — Core MC simulation engine
  - `risk_simulation/scenarios.py` — Historical black swan event replayer
  - `risk_simulation/var_calculator.py` — VaR and CVaR computation
  - `risk_simulation/visualizer.py` — Generate data for dashboard charts

### Monte Carlo Design (Beginner-Friendly)
```python
class MonteCarloSimulator:
    """Simulate thousands of possible future equity paths."""

    def run(self, trade_history: list, n_simulations: int = 10000,
            n_days: int = 252) -> MonteCarloResult:
        """
        1. Calculate historical return distribution from trade_history
        2. Sample random returns from this distribution
        3. Project equity paths forward
        4. Compute VaR, CVaR, worst-case, best-case
        """
        pass

@dataclass
class MonteCarloResult:
    paths: np.ndarray          # (n_simulations, n_days) equity paths
    var_95: float              # 95% Value at Risk
    var_99: float              # 99% Value at Risk
    cvar_95: float             # Conditional VaR (expected loss beyond VaR)
    max_drawdown_dist: list    # Distribution of max drawdowns
    probability_of_ruin: float # P(equity < 0)
    median_final_equity: float
    percentile_5: float        # Worst 5% outcome
    percentile_95: float       # Best 5% outcome
```

### Black Swan Scenarios
Replay real historical crashes:
- **March 2020 COVID crash** — BTC dropped 50% in 2 days
- **May 2021 China ban** — BTC dropped 55% over 2 weeks
- **FTX collapse Nov 2022** — BTC dropped 25% in a week
- **Terra/Luna May 2022** — Market-wide contagion

Each scenario applies the historical drawdown pattern to the current portfolio and shows projected losses.

### Dashboard Visualization
- **Fan chart**: 10,000 equity paths with 5th/25th/50th/75th/95th percentile bands
- **Histogram**: Distribution of final equity values
- **Drawdown heatmap**: Probability of hitting various drawdown levels
- **Key metrics cards**: VaR, CVaR, P(ruin), median outcome — all with plain-English explanations

### Modify
- `config.py` — Add `MC_SIMULATIONS`, `MC_HORIZON_DAYS`
- `api/routes/risk.py` — Already scaffolded in Phase 1

---

## Implementation Order & Dependencies

```
Phase 1: Data Store + API ──────────┐
Phase 2: Docker Deployment ─────────┤
Phase 3: React Dashboard ──────────←┘ (needs API)
Phase 4: Extended Backtesting ──────── (needs dashboard for results display)
Phase 5: Intelligence Signals ──────── (needs config toggles, feeds into strategies)
Phase 6: Multi-Exchange Arbitrage ──── (needs multi-exchange config, async fetching)
Phase 7: Monte Carlo Risk Sim ─────── (needs trade history, dashboard for viz)
```

Phases 4-7 are largely independent and could be parallelized, but the order above minimizes integration friction.

---

## New Dependencies

```
# Phase 1
fastapi>=0.109.0
uvicorn[standard]>=0.27.0

# Phase 5
yfinance>=0.2.36            # Yahoo Finance for stock indices
textblob>=0.18.0            # Simple NLP sentiment (optional, can use sklearn)

# Phase 7
scipy>=1.11.0               # Already transitive dep of sklearn, used for distributions
```

Note: `scipy` is already installed as a transitive dependency of `scikit-learn`. `yfinance` and `textblob` are the only truly new dependencies.

Dashboard dependencies (separate `package.json`):
```
next, react, react-dom, tailwindcss, recharts, swr, typescript
```

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `config.py` | API settings, multi-exchange keys, intelligence toggles, data dir, MC params |
| `agent.py` | DataStore integration, intelligence aggregator call, arbitrage scan step |
| `decision_engine.py` | Accept intelligence adjustment, arbitrage routing |
| `data_fetcher.py` | MultiExchangeFetcher class for arbitrage |
| `backtester.py` | Multi-pair, multi-scenario, multi-timeframe support |
| `requirements.txt` | FastAPI, uvicorn, yfinance, textblob |
| `state_manager.py` | Use Config.DATA_DIR for all paths |
| `logger.py` | Use Config.DATA_DIR for log path |

---

## Verification Plan

### Phase 1 Verification
```bash
# Start API server
python -m api.server
# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/status
curl http://localhost:8000/trades
```

### Phase 2 Verification
```bash
docker compose build
docker compose up -d
docker compose logs -f agent
curl http://localhost:8000/health
```

### Phase 3 Verification
```bash
cd dashboard && npm install && npm run dev
# Open http://localhost:3000
# Verify: metric cards populate, equity chart renders, trades table shows data
```

### Phase 4 Verification
```bash
# Via API
curl -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"pair": "ETH/USDT", "scenario": "bull_run"}'
# Check dashboard backtest page
```

### Phase 5 Verification
```bash
# Enable in .env
ENABLE_ONCHAIN=true
ENABLE_NEWS_NLP=true
# Run agent, check intelligence signals in logs and dashboard
python agent.py 5
```

### Phase 6 Verification
```bash
# Configure exchanges in .env
ARBITRAGE_ENABLED=true
BINANCE_API_KEY=...
# Run agent, check arbitrage opportunities in logs
python agent.py 5
```

### Phase 7 Verification
```bash
curl -X POST http://localhost:8000/risk/monte-carlo \
  -H "Content-Type: application/json" \
  -d '{"simulations": 1000, "horizon_days": 30}'
# Check risk page in dashboard for fan chart and VaR metrics
```
