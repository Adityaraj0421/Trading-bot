#!/bin/bash
# ============================================================
# Stop Overnight Paper Trading Test
# Gracefully shuts down all components
# ============================================================

cd "$(dirname "$0")"

echo ""
echo "Stopping overnight test..."
echo ""

# Stop API server + Agent
if [ -f data/api_server.pid ]; then
    PID=$(cat data/api_server.pid)
    if kill -0 $PID 2>/dev/null; then
        # Send SIGTERM for graceful shutdown (triggers GracefulShutdown handler)
        kill $PID
        echo "  Agent (PID $PID):     Stopping..."
        # Wait up to 10 seconds for graceful shutdown
        for i in $(seq 1 10); do
            if ! kill -0 $PID 2>/dev/null; then
                echo "  Agent:                Stopped gracefully"
                break
            fi
            if [ $i -eq 10 ]; then
                kill -9 $PID 2>/dev/null
                echo "  Agent:                Force killed"
            fi
            sleep 1
        done
    else
        echo "  Agent (PID $PID):     Already stopped"
    fi
    rm -f data/api_server.pid
else
    echo "  Agent:                No PID file found"
fi

# Stop Dashboard
if [ -f data/dashboard.pid ]; then
    PID=$(cat data/dashboard.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID 2>/dev/null
        sleep 2
        kill -0 $PID 2>/dev/null && kill -9 $PID 2>/dev/null
        echo "  Dashboard (PID $PID): Stopped"
    else
        echo "  Dashboard (PID $PID): Already stopped"
    fi
    rm -f data/dashboard.pid
else
    echo "  Dashboard:            No PID file found"
fi

# Also kill any orphaned processes on the ports
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true

echo ""
echo "============================================"
echo "  All systems stopped."
echo ""
echo "  Review your overnight results:"
echo "    Agent log:   data/api_server.log"
echo "    Dashboard:   data/dashboard.log"
echo "    Trade DB:    data/trades.db"
echo "    State:       data/agent_state.json"
echo "    Trade log:   trade_log.json"
echo "============================================"
echo ""
