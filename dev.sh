#!/bin/bash
# ============================================================
# dev.sh — One-command development startup
# Starts the API server + Dashboard with clean Ctrl+C handling
# ============================================================
# Usage: ./dev.sh  or  make dev
# ============================================================

set -e
cd "$(dirname "$0")"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

API_PID=""
API_LOG="data/dev_api.log"

# -------------------------------------------------------
# Cleanup handler — runs on Ctrl+C or script exit
# -------------------------------------------------------
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"

    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null
        # Wait up to 5s for graceful shutdown
        for i in $(seq 1 5); do
            kill -0 "$API_PID" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still running
        kill -0 "$API_PID" 2>/dev/null && kill -9 "$API_PID" 2>/dev/null
        echo -e "  API server:  ${GREEN}stopped${NC}"
    fi

    # Clean up orphans on ports
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null || true

    echo -e "  Dashboard:   ${GREEN}stopped${NC}"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# -------------------------------------------------------
# 1. Check prerequisites
# -------------------------------------------------------
echo ""
echo -e "${BOLD}Crypto Trading Agent — Dev Mode${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python venv
if [ -d "venv" ]; then
    export PATH="$(pwd)/venv/bin:$PATH"
    echo -e "  Python venv: ${GREEN}found${NC}"
else
    echo -e "  Python venv: ${YELLOW}not found (using system Python)${NC}"
fi

# Check node_modules
if [ ! -d "dashboard/node_modules" ]; then
    echo -e "  node_modules: ${YELLOW}missing — installing...${NC}"
    (cd dashboard && npm install)
fi
echo -e "  node_modules: ${GREEN}ready${NC}"

# Check if ports are free
if lsof -ti:8000 > /dev/null 2>&1; then
    echo -e "  Port 8000:   ${RED}in use${NC} — killing existing process"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

if lsof -ti:3000 > /dev/null 2>&1; then
    echo -e "  Port 3000:   ${RED}in use${NC} — killing existing process"
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Ensure data directory exists
mkdir -p data

echo ""

# -------------------------------------------------------
# 2. Start API server (background)
# -------------------------------------------------------
echo -e "${BLUE}[API]${NC} Starting Python API server..."
python -m api.server > "$API_LOG" 2>&1 &
API_PID=$!
echo -e "${BLUE}[API]${NC} PID: $API_PID (logging to $API_LOG)"

# Wait for API to be healthy
echo -e "${BLUE}[API]${NC} Waiting for server to be ready..."
for i in $(seq 1 20); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${BLUE}[API]${NC} ${GREEN}Ready${NC} at http://localhost:8000"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo -e "${RED}[API] Server crashed! Check $API_LOG${NC}"
        tail -20 "$API_LOG"
        exit 1
    fi
    if [ "$i" -eq 20 ]; then
        echo -e "${RED}[API] Timeout waiting for server. Check $API_LOG${NC}"
        tail -20 "$API_LOG"
        exit 1
    fi
    sleep 2
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${GREEN}API:${NC}       http://localhost:8000"
echo -e "  ${GREEN}Dashboard:${NC} http://localhost:3000"
echo -e "  ${GREEN}API Docs:${NC}  http://localhost:8000/docs"
echo -e "  API log:   tail -f $API_LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Press ${BOLD}Ctrl+C${NC} to stop everything"
echo ""

# -------------------------------------------------------
# 3. Start Dashboard (foreground)
# -------------------------------------------------------
echo -e "${GREEN}[Dashboard]${NC} Starting Next.js dev server..."
cd dashboard
npm run dev
