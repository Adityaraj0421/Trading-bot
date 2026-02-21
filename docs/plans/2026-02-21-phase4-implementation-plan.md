# Phase 4: Deep Polish — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the codebase production-grade with comprehensive type hints, docstrings, clean dependencies, standard tooling, and consistent patterns.

**Architecture:** Bottom-up by layer — infrastructure first, then data/strategy modules, then agent/API. Each task is a self-contained commit.

**Tech Stack:** Python 3.14, ruff (linting/formatting), pytest, FastAPI

---

## Task 1: Clean Dependencies & Add Tooling

**Files:**
- Modify: `requirements.txt`
- Create: `pyproject.toml`
- Modify: `Makefile`

**Step 1: Remove unused packages from requirements.txt**

Remove these 5 packages (verified as never imported anywhere in the codebase):
- `ta>=0.11.0` — custom `Indicators` class replaces it
- `schedule>=1.2.0` — agent uses `threading.Event.wait()` instead
- `httpx>=0.27.0` — `requests` is used for all HTTP
- `yfinance>=0.2.36` — never imported
- `python-telegram-bot>=21.0` — `telegram_bot.py` uses raw `requests.post()`, not this library

Keep `pytest-cov` (used by `make test-coverage`).

**Step 2: Create pyproject.toml**

```toml
[project]
name = "crypto-trading-agent"
version = "9.0.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = [
    "E501",    # line too long — handled by formatter
    "B905",    # zip-without-strict
    "SIM108",  # ternary instead of if-else
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["B011", "SIM300"]
```

**Step 3: Add lint/format targets to Makefile**

Add after the existing `test-config` target:

```makefile
lint: ## Run ruff linter
	$(PYTHON) -m ruff check .

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .
```

Also add `ruff>=0.4.0` to requirements.txt under a `# Development` section.

**Step 4: Install ruff and verify**

Run: `./venv/bin/pip install ruff`
Run: `make lint` — expect warnings (will fix in later tasks)
Run: `make test` — expect 890 passed

**Step 5: Commit**

```
git add requirements.txt pyproject.toml Makefile
git commit -m "Clean deps, add pyproject.toml + ruff linting (Phase 4.1)"
```

---

## Task 2: Create exceptions.py + Logging Standardization

**Files:**
- Create: `exceptions.py`
- Modify: all production .py files (logging only)

**Step 1: Create exceptions.py**

```python
"""
Custom exception hierarchy for the trading agent.
All domain-specific exceptions inherit from TradingError.
"""

class TradingError(Exception):
    """Base exception for all trading agent errors."""

class DataFetchError(TradingError):
    """Failed to fetch market data from exchange or API."""

class ExecutionError(TradingError):
    """Failed to execute, modify, or cancel an order."""

class ValidationError(TradingError):
    """Configuration or input validation failure."""

class ModelError(TradingError):
    """ML model training or prediction failure."""
```

**Step 2: Standardize logging across all modules**

For every production .py file, ensure:
1. Module-level logger: `_log = logging.getLogger(__name__)`
2. Replace stray `print()` calls with `_log.info()` (except in `_print_banner()` and CLI output)
3. Replace `logging.getLogger("something")` with `logging.getLogger(__name__)`

Files to check (in order):
- `config.py` — uses `print()` for banner, `logging` for warnings (banner OK, keep)
- `agent.py` — uses both `print()` and `logging.getLogger(__name__)`
- `data_fetcher.py` — check for `print()` usage
- `executor.py` — check for `print()` usage
- `model.py` — check for `print()` usage
- `backtester.py` — check for `print()` usage
- All remaining modules

**Step 3: Run tests**

Run: `make test` — expect 890 passed

**Step 4: Commit**

```
git add exceptions.py *.py
git commit -m "Add exception hierarchy + standardize logging (Phase 4.2)"
```

---

## Task 3: Type Hints — Infrastructure Layer

**Files:**
- Modify: `config.py`, `graceful_shutdown.py`, `trade_db.py`, `logger.py`, `demo_data.py`, `exceptions.py`

Add return type annotations to EVERY public function. Add parameter types where missing.

**Key patterns:**
- `-> None` for void methods
- `-> dict` / `-> dict[str, Any]` for snapshot-like returns
- `-> list[dict]` for query results
- `-> Optional[X]` for nullable returns
- `-> pd.DataFrame` for data generation

**Files and function signatures to annotate:**

`config.py`: `validate() -> None`, `is_paper_mode() -> bool`, `any_notifications_enabled() -> bool`, `any_intelligence_enabled() -> bool`

`graceful_shutdown.py`: All `GracefulShutdown` and `RateLimiter` methods

`trade_db.py`: All `TradeDB` methods (already mostly typed, verify completeness)

`logger.py`: All `StructuredLogger` methods

`demo_data.py`: `generate_ohlcv() -> pd.DataFrame`

**Run tests after, commit:**
```
git commit -m "Type hints: infrastructure layer (Phase 4.3)"
```

---

## Task 4: Type Hints — Data & Strategy Layer

**Files:**
- Modify: `data_fetcher.py`, `indicators.py`, `sentiment.py`, `regime_detector.py`, `strategies.py`, `model.py`, `portfolio.py`, `market_impact.py`

**Key signatures to annotate:**

`data_fetcher.py`: `DataFetcher.__init__`, `fetch_ohlcv() -> pd.DataFrame`, `_init_exchange() -> Any`

`indicators.py`: `Indicators.add_all(df) -> pd.DataFrame`, cache methods

`sentiment.py`: All `SentimentAnalyzer` methods (mostly done, verify)

`regime_detector.py`: `RegimeDetector.detect() -> MarketRegime`, HMM methods

`strategies.py`: Every strategy's `generate_signal() -> tuple[str, float]`

`model.py`: `TradingModel.train() -> dict`, `predict() -> tuple[str, float]`

`portfolio.py`: All `PortfolioManager` methods

`market_impact.py`: All `MarketImpactModel` methods

**Run tests after, commit:**
```
git commit -m "Type hints: data & strategy layer (Phase 4.4)"
```

---

## Task 5: Type Hints — Agent & Decision Layer

**Files:**
- Modify: `agent.py`, `decision_engine.py`, `risk_manager.py`, `executor.py`
- Modify: `rl_ensemble.py`, `strategy_selector.py`, `walk_forward.py`
- Modify: `auto_optimizer.py`, `backtester.py`
- Modify: `meta_learner.py`, `drift_detector.py`, `self_healer.py`, `strategy_evolver.py`
- Modify: `multi_timeframe.py`, `backtest_runner.py`

**Key signatures:**

`risk_manager.py`: `Position` dataclass fields, `RiskManager.calculate_position_size() -> float`, `should_trade() -> bool`, `check_exit() -> Optional[str]`

`decision_engine.py`: `DecisionEngine.decide() -> dict`, `emergency_halt() -> None`

`executor.py`: `PaperExecutor.execute() -> dict`, `LiveExecutor.execute() -> dict`

`agent.py`: `TradingAgent.run_cycle() -> None`, `run() -> None`, `set_data_store() -> None`

`rl_ensemble.py`: All `RLEnsemble`, `QLearnerAgent`, `ReplayBuffer` methods

`strategy_selector.py`: All `StrategyMetaSelector` methods

Plus remaining modules: `walk_forward.py`, `auto_optimizer.py`, `backtester.py`, `meta_learner.py`, `drift_detector.py`, `self_healer.py`, `strategy_evolver.py`, `multi_timeframe.py`, `backtest_runner.py`

**Run tests after, commit:**
```
git commit -m "Type hints: agent & decision layer (Phase 4.5)"
```

---

## Task 6: Type Hints — API & Notification Layer

**Files:**
- Modify: `notifier.py`, `websocket_streamer.py`, `telegram_bot.py`
- Modify: `api/data_store.py`, `api/server.py`
- Modify: `api/routes/status.py`, `api/routes/trading.py`, `api/routes/autonomous.py`
- Modify: `api/routes/backtest.py`, `api/routes/intelligence.py`
- Modify: `api/routes/arbitrage.py`, `api/routes/risk.py`, `api/routes/telegram.py`
- Modify: `api/routes/websocket.py`, `api/websocket_manager.py`

**Run tests after, commit:**
```
git commit -m "Type hints: API & notification layer (Phase 4.6)"
```

---

## Task 7: Docstrings — All Modules (Batch 1: Core)

**Files:**
- Modify: `config.py`, `agent.py`, `risk_manager.py`, `decision_engine.py`, `model.py`, `executor.py`

For each file:
1. Class-level docstring explaining purpose, lifecycle, key attributes
2. Method docstrings for ALL public methods: one-line summary, params, returns
3. Module-level docstring if missing

**Format:**
```python
def method(self, param: str) -> dict:
    """One-line summary of what this does.

    Args:
        param: Description of the parameter.

    Returns:
        Dictionary containing keys 'x' and 'y'.
    """
```

**Run tests after, commit:**
```
git commit -m "Docstrings: core modules (Phase 4.7)"
```

---

## Task 8: Docstrings — All Modules (Batch 2: Strategy & Data)

**Files:**
- Modify: `strategies.py`, `data_fetcher.py`, `indicators.py`, `sentiment.py`
- Modify: `regime_detector.py`, `portfolio.py`, `market_impact.py`

Same docstring treatment as Task 7.

**Run tests after, commit:**
```
git commit -m "Docstrings: strategy & data modules (Phase 4.8)"
```

---

## Task 9: Docstrings — All Modules (Batch 3: ML & Remaining)

**Files:**
- Modify: `rl_ensemble.py`, `strategy_selector.py`, `walk_forward.py`
- Modify: `auto_optimizer.py`, `backtester.py`, `meta_learner.py`
- Modify: `drift_detector.py`, `self_healer.py`, `strategy_evolver.py`
- Modify: `multi_timeframe.py`, `backtest_runner.py`
- Modify: `notifier.py`, `websocket_streamer.py`, `telegram_bot.py`
- Modify: `graceful_shutdown.py`, `trade_db.py`, `logger.py`, `demo_data.py`

**Run tests after, commit:**
```
git commit -m "Docstrings: ML, notification, and remaining modules (Phase 4.9)"
```

---

## Task 10: Docstrings — API Layer

**Files:**
- Modify: `api/data_store.py`, `api/server.py`, `api/websocket_manager.py`
- Modify: all `api/routes/*.py` files

**Run tests after, commit:**
```
git commit -m "Docstrings: API layer (Phase 4.10)"
```

---

## Task 11: Dead Code Removal + Cleanup

**Files:**
- Modify: various .py files

**Actions:**
1. Remove unused imports (run `ruff check --select F401`)
2. Remove `indicators.invalidate_cache()` if confirmed never called
3. Remove any other confirmed dead functions
4. Fix any remaining `ruff` warnings that are safe to fix

**Run: `make lint` — expect zero errors**
**Run: `make test` — expect 890 passed**

**Commit:**
```
git commit -m "Remove dead code and fix lint warnings (Phase 4.11)"
```

---

## Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Changes:**
1. Fix `agent.py` line count (says "57K lines" — actually 1314)
2. Update test count (now 890 tests across 45+ files)
3. Add `make lint` and `make format` to commands table
4. Add Phase 1-4 hardening summary section
5. Add troubleshooting section for common issues
6. Add exceptions.py to architecture diagram

**Run: `make test` — final verification, 890 passed**

**Commit:**
```
git commit -m "Update CLAUDE.md with accurate stats and Phase 1-4 summary (Phase 4.12)"
```

---

## Verification Checklist

After all tasks:
- [ ] `make test` → 890 passed, 0 failed
- [ ] `make lint` → 0 errors
- [ ] Every public function has a return type annotation
- [ ] Every public class has a docstring
- [ ] No stray `print()` in production modules (except banner/CLI)
- [ ] `requirements.txt` has no unused packages
- [ ] `CLAUDE.md` is accurate
