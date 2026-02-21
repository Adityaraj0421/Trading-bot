# ============================================================
# Makefile — Crypto Trading Agent
# ============================================================
# Usage: make help
# ============================================================

.PHONY: dev api dashboard stop status install logs start test test-coverage build lint format help

# Detect venv
VENV_BIN   := ./venv/bin
PYTHON     := $(shell [ -d $(VENV_BIN) ] && echo $(VENV_BIN)/python || echo python)
PIP        := $(shell [ -d $(VENV_BIN) ] && echo $(VENV_BIN)/pip || echo pip)

# -------------------------------------------------------
# Development
# -------------------------------------------------------

dev: ## Start API + Dashboard for development (Ctrl+C stops both)
	@bash dev.sh

api: ## Start only the API server (foreground)
	$(PYTHON) -m api.server

dashboard: ## Start only the Next.js dashboard
	cd dashboard && npm run dev

# -------------------------------------------------------
# Background mode (headless)
# -------------------------------------------------------

start: ## Start both in background (for overnight/headless runs)
	@bash run_overnight.sh

stop: ## Stop all running API and dashboard processes
	@bash stop_overnight.sh

logs: ## Tail background logs (API + Dashboard)
	@tail -f data/api_server.log data/dashboard.log 2>/dev/null || \
		tail -f data/dev_api.log 2>/dev/null || \
		echo "No log files found. Start the system first with: make dev"

# -------------------------------------------------------
# Status & Health
# -------------------------------------------------------

status: ## Check if API and Dashboard are running
	@echo ""
	@echo "Service Status"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if lsof -ti:8000 > /dev/null 2>&1; then \
		echo "  API (port 8000):       ✓ running (PID $$(lsof -ti:8000 | head -1))"; \
	else \
		echo "  API (port 8000):       ✗ not running"; \
	fi
	@if lsof -ti:3000 > /dev/null 2>&1; then \
		echo "  Dashboard (port 3000): ✓ running (PID $$(lsof -ti:3000 | head -1))"; \
	else \
		echo "  Dashboard (port 3000): ✗ not running"; \
	fi
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""

# -------------------------------------------------------
# Setup & Build
# -------------------------------------------------------

install: ## Install all dependencies (Python + Node)
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installing Node dependencies..."
	cd dashboard && npm install
	@echo ""
	@echo "All dependencies installed."

build: ## Build the dashboard for production
	cd dashboard && npm run build

# -------------------------------------------------------
# Testing
# -------------------------------------------------------

test: ## Run Python tests
	$(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "HTML coverage report: htmlcov/index.html"

test-config: ## Run config and model tests only
	$(PYTHON) -m pytest tests/test_config.py tests/test_models.py -v

lint: ## Run ruff linter
	$(PYTHON) -m ruff check .

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

# -------------------------------------------------------
# Help
# -------------------------------------------------------

help: ## Show all available commands
	@echo ""
	@echo "Crypto Trading Agent — Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

.DEFAULT_GOAL := help
