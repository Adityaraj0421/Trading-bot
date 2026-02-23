# Phase 4: Full Code Quality Sweep

## Context

The trading agent is at v9.1 with 10 strategies, 935 tests passing, and 0 ruff lint errors. The
tooling foundation (pyproject.toml, Makefile lint/format targets, exceptions.py) is already in place.
However, type annotations cover ~30% of public methods and Google-style docstrings cover ~20%. Four
unused packages bloat `requirements.txt`. This plan brings all 37+ production modules to 100%
type-hint and docstring coverage, removes dead dependencies, and refreshes CLAUDE.md.

**No behavioral changes** — this is purely annotation and documentation work.

## Type Hint Pattern

```python
# Use throughout
def method(self, param: str, price: float = 0.0) -> dict[str, Any]: ...
def fetch(self, symbol: str) -> pd.DataFrame: ...
def get_snapshot(self) -> dict[str, Any] | None: ...
```

## Docstring Format (Google style — already in use)

```python
def method(self, param: str) -> dict[str, Any]:
    """One-line summary.

    Args:
        param: Description of the parameter.

    Returns:
        Dictionary with keys 'x' and 'y'.

    Raises:
        DataFetchError: If exchange is unreachable.
    """
```

## Implementation Plan

### Step 0 — Write design doc
Create `docs/plans/2026-02-24-phase4-code-quality-design.md` with this plan content and commit it.

### Batch 0 — Dependency cleanup (do first, fastest win)

**File:** `requirements.txt`

Remove these 4 unused packages:
- `ta>=0.11.0` — replaced by custom `Indicators` class
- `schedule>=1.2.0` — replaced by `threading.Event.wait()`
- `httpx>=0.27.0` — `requests` used for all HTTP
- `python-telegram-bot>=21.0` — `telegram_bot.py` uses raw `requests.post()`

Keep `yfinance` — used in `intelligence/correlation.py`.

Verify: `make test` (935 tests pass) + `make lint` (0 errors).

### Batch 1 — Infrastructure layer

**Files:** `config.py`, `graceful_shutdown.py`, `trade_db.py`, `logger.py`, `demo_data.py`

For each:
1. Add return type annotations to all public methods
2. Add `Args:` / `Returns:` / `Raises:` docstrings to all public methods
3. Ensure module-level docstring exists

Key patterns:
- `config.py`: `load_config() -> Config`, properties typed as `-> str | None`, `-> int`, etc.
- `trade_db.py`: `insert_trade(...) -> int`, `get_trade_history(...) -> list[dict[str, Any]]`
- `graceful_shutdown.py`: `GracefulShutdown.register(...)` and `RateLimiter.is_allowed()` return types

Verify: `make test` + `make lint`

### Batch 2 — Data & indicators layer

**Files:** `data_fetcher.py`, `indicators.py`, `sentiment.py`, `regime_detector.py`, `portfolio.py`, `market_impact.py`

Key patterns:
- `data_fetcher.py`: `fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame`
- `indicators.py`: `add_all(df: pd.DataFrame) -> pd.DataFrame`, `FEATURE_COLUMNS: list[str]` typed constant
- `market_impact.py`: `simulate_execution(order: dict[str, Any]) -> dict[str, Any]`

Verify: `make test` + `make lint`

### Batch 3 — Strategies

**Files:** `strategies.py`, `pair_scorer.py`

Key patterns:
- Each strategy class `generate_signal(df: pd.DataFrame) -> tuple[str, float]` — action + confidence
- `pair_scorer.py`: `score_pairs(symbols: list[str], data: dict[str, pd.DataFrame]) -> list[PairScore]`
- All 10 strategy classes get class-level docstrings describing their signal logic

Verify: `make test` + `make lint`

### Batch 4 — ML models

**Files:** `model.py`, `rl_ensemble.py`, `strategy_selector.py`, `meta_learner.py`, `drift_detector.py`

Key patterns:
- `model.py`: `train(X: np.ndarray, y: np.ndarray) -> None`, `predict(X: np.ndarray) -> np.ndarray`
- `rl_ensemble.py`: `vote(signals: list[dict[str, Any]]) -> dict[str, Any]`
- `drift_detector.py`: `detect(current: np.ndarray, reference: np.ndarray) -> bool`

Verify: `make test` + `make lint`

### Batch 5 — Agent & decision core (most critical)

**Files:** `agent.py`, `decision_engine.py`, `risk_manager.py`, `executor.py`

Key patterns:
- `agent.py`: `run() -> None`, `_execute_cycle() -> None`, `_reconcile_trade_db() -> None`
- `decision_engine.py`: `make_decision(df: pd.DataFrame, symbol: str) -> DecisionResult`
- `risk_manager.py`: `calculate_position_size(...) -> float`, `calculate_stop_take(...) -> tuple[float, float]`
- `executor.py`: `execute_trade(signal: dict[str, Any]) -> dict[str, Any] | None`

Note: `agent.py` is 1,300+ lines — add annotations method-by-method, don't change logic.

Verify: `make test` + `make lint`

### Batch 6 — Backtesting & optimization

**Files:** `backtester.py`, `auto_optimizer.py`, `walk_forward.py`, `strategy_evolver.py`, `self_healer.py`, `multi_timeframe.py`, `backtest_runner.py`

Key patterns:
- `backtester.py`: `run(data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult`
- `strategy_evolver.py`: `evolve(genomes: list[Genome]) -> list[Genome]`
- `self_healer.py`: `check_and_heal(state: dict[str, Any]) -> bool`

Verify: `make test` + `make lint`

### Batch 7 — API & notification layer

**Files:** `notifier.py`, `websocket_streamer.py`, `telegram_bot.py`, `async_fetcher.py`,
`api/data_store.py`, `api/server.py`, `api/websocket_manager.py`,
`api/routes/status.py`, `api/routes/trading.py`, `api/routes/autonomous.py`,
`api/routes/backtest.py`, `api/routes/intelligence.py`, `api/routes/arbitrage.py`,
`api/routes/risk.py`, `api/routes/telegram.py`, `api/routes/websocket.py`

Key patterns:
- FastAPI route handlers already typed via Pydantic — focus on helper functions and class methods
- `api/data_store.py`: `DataStore.get_snapshot() -> dict[str, Any]`, `append_trade(trade: dict[str, Any]) -> None`
- `notifier.py`: `send_alert(message: str, level: str = "info") -> None`

Verify: `make test` + `make lint`

### Batch 8 — CLAUDE.md refresh

Update `CLAUDE.md`:
- Fix agent.py line count (~1,314 lines, not "~1.4K lines")
- Update test count: 935 tests, 44 test files
- Add Phase 1–4 summary to the overview section
- Update "Latest commit" references
- Remove any stale gotchas that were fixed

Commit all changes with message: `chore: Phase 4 code quality sweep (type hints, docstrings, dep cleanup)`

## Critical Files

| File | Purpose | Batch |
|------|---------|-------|
| `requirements.txt` | Remove 4 unused deps | 0 |
| `config.py` | Foundation — typed first | 1 |
| `risk_manager.py` | Safety-critical | 5 |
| `agent.py` | Main loop, 1,300+ lines | 5 |
| `strategies.py` | 10 strategy classes | 3 |
| `api/data_store.py` | Thread-safe bridge | 7 |
| `CLAUDE.md` | Project docs refresh | 8 |

## Reuse Existing Patterns

- `exceptions.py` — already has `TradingError`, `DataFetchError`, `ExecutionError`, `ValidationError`, `ModelError` — use these in `Raises:` sections
- `indicators.py:add_all()` already annotated `-> pd.DataFrame` — follow this pattern
- `position.py` dataclass (in `risk_manager.py`) — follow its `@dataclass` field typing pattern

## Verification

After each batch:
```bash
make test      # must show 935 passing, 0 failures
make lint      # must show 0 errors
```

Final verification:
```bash
make test-coverage   # check coverage stays stable
./venv/bin/python -m ruff check --select ANN src/   # spot remaining annotation gaps
```

No tests should fail at any point — type annotations are additive and don't change runtime behavior.
