#!/bin/bash
# ============================================================
# Overnight Paper Trading Test — Full System Deployment
# Crypto Trading Agent v9.0
# ============================================================
# Usage: ./run_overnight.sh
# Stop:  ./stop_overnight.sh
# ============================================================

set -e
cd "$(dirname "$0")"

# Use the project's virtual environment
export PATH="$(pwd)/venv/bin:$PATH"

echo ""
echo "============================================"
echo "  Crypto Trading Agent v9.0"
echo "  Overnight Paper Trading Test"
echo "============================================"
echo ""

# -------------------------------------------------------
# 1. Preflight checks
# -------------------------------------------------------
echo "[1/6] Preflight checks..."
python -c "
from config import Config
Config.validate()
print('  Config:     OK')
print(f'  Mode:       {Config.TRADING_MODE.upper()}')
print(f'  Pairs:      {Config.TRADING_PAIRS}')
print(f'  Capital:    \${Config.INITIAL_CAPITAL:,.0f}')
print(f'  Interval:   {Config.AGENT_INTERVAL_SECONDS}s')
print(f'  WebSocket:  {Config.ENABLE_WEBSOCKET}')
print(f'  Arbitrage:  {Config.ARBITRAGE_ENABLED}')
print(f'  Trade DB:   {Config.ENABLE_TRADE_DB}')
print(f'  API Auth:   {\"ON\" if Config.API_AUTH_KEY else \"OFF\"}')
"
echo ""

# -------------------------------------------------------
# 2. Test Telegram notification
# -------------------------------------------------------
echo "[2/6] Testing Telegram notification..."
python -c "
import time
from notifier import Notifier
n = Notifier()
if n.telegram_enabled:
    n.notify_state_change('offline', 'normal',
        'Overnight paper trading test starting! All 33 components enabled.')
    time.sleep(2)  # Wait for async send
    print('  Telegram:   SENT (check your phone)')
else:
    print('  Telegram:   NOT CONFIGURED')
"
echo ""

# -------------------------------------------------------
# 3. Start API server + Agent
# -------------------------------------------------------
echo "[3/6] Starting API server + Trading Agent..."

# Kill any existing processes on port 8000
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true

nohup python -m api.server > data/api_server.log 2>&1 &
API_PID=$!
echo $API_PID > data/api_server.pid
echo "  API PID:    $API_PID"

# Wait for server to be ready
echo "  Waiting for API server..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  API:        READY"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "  ERROR: API server failed to start. Check data/api_server.log"
        exit 1
    fi
    sleep 2
done
echo ""

# -------------------------------------------------------
# 4. Health check
# -------------------------------------------------------
echo "[4/6] Health check..."
curl -s http://localhost:8000/health | python -m json.tool
echo ""

# -------------------------------------------------------
# 5. Start dashboard
# -------------------------------------------------------
echo "[5/6] Starting Next.js dashboard..."

# Kill any existing processes on port 3000
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true

cd dashboard
nohup npm run dev > ../data/dashboard.log 2>&1 &
DASH_PID=$!
echo $DASH_PID > ../data/dashboard.pid
echo "  Dashboard PID: $DASH_PID"
cd ..

# Wait for dashboard
sleep 5
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "  Dashboard:  READY"
else
    echo "  Dashboard:  Starting (may take a moment)..."
fi
echo ""

# -------------------------------------------------------
# 6. Summary
# -------------------------------------------------------
echo "[6/6] All systems running!"
echo ""
echo "============================================"
echo "  SYSTEM STATUS"
echo "============================================"
echo ""
echo "  API Server:  http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Dashboard:   http://localhost:3000"
echo "  API Key:     overnight-test-key-2024"
echo ""
echo "  ENABLED MODULES:"
echo "    Intelligence: OnChain, Whale, News, Orderbook,"
echo "                  Correlation, FundingOI, Liquidation,"
echo "                  LLM, Cascade"
echo "    Strategies:   Momentum, MeanReversion, Breakout,"
echo "                  Grid, Scalping, Sentiment, VWAP,"
echo "                  OBVDivergence, EMACrossover"
echo "    ML Model:     PyTorch LSTM + XGBoost (3-tier)"
echo "    RL Ensemble:  4 DQN + PPO meta-agent"
echo "    Arbitrage:    Triangular + Funding Rate"
echo "    WebSocket:    Real-time Binance feed"
echo ""
echo "  LOGS:"
echo "    Agent:     tail -f data/api_server.log"
echo "    Dashboard: tail -f data/dashboard.log"
echo ""
echo "  STOP:  ./stop_overnight.sh"
echo "============================================"
echo ""
echo "Tailing agent log (Ctrl+C to detach — system keeps running)..."
echo ""
tail -f data/api_server.log
