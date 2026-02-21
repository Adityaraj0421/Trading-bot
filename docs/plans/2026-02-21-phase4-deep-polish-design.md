# Phase 4: Deep Polish — Design

**Date:** 2026-02-21
**Status:** Approved
**Prerequisite:** Phases 1-3 complete (commits `3c16d8b`, `150b802`, `04fdf94`, `970e2c2`)

## Goal

Make the codebase production-grade: comprehensive type hints, docstrings on every public API, clean dependencies, standard tooling, consistent error handling, and an accurate CLAUDE.md.

## Approach

Bottom-up by layer: config/infra → data → strategies → agent → API.
Each layer is complete before moving up so type hints propagate naturally.

---

## Work Package 1: Dependencies & Tooling

### Clean requirements.txt
- Remove unused packages: `ta`, `schedule`, `httpx`
- Verify `python-telegram-bot` and `yfinance` are actually imported; remove if not
- Ensure all actually-used packages are listed

### Add pyproject.toml
- `[project]` metadata (name, version, python-requires)
- `[tool.pytest.ini_options]` — testpaths, markers
- `[tool.ruff]` — line length, select rules, per-file ignores for tests

### Update Makefile
- Add `lint` target (ruff check)
- Add `format` target (ruff format)
- Keep existing `test`, `install`, `dev` targets

---

## Work Package 2: Type Hints (all modules)

Add return type annotations to every public function (~40+ across 15+ modules).
Add `Optional[]`, `list[]`, `dict[]` parameter hints where missing.

Priority order:
1. `config.py` — foundation for everything
2. `risk_manager.py` — safety-critical position management
3. `decision_engine.py` — autonomous decision logic
4. `model.py` — ML tier selection and prediction
5. `agent.py` — main entry point
6. `strategies.py` — signal generation
7. `executor.py` — order execution
8. `data_fetcher.py` — exchange connectivity
9. Remaining modules: `sentiment.py`, `indicators.py`, `portfolio.py`, `regime_detector.py`, `backtester.py`, `auto_optimizer.py`, `walk_forward.py`, `rl_ensemble.py`, `strategy_selector.py`, `market_impact.py`, `notifier.py`, `websocket_streamer.py`, `graceful_shutdown.py`, `trade_db.py`, `logger.py`, `telegram_bot.py`, `demo_data.py`

---

## Work Package 3: Docstrings (all public APIs)

- Class-level docstrings for every public class: purpose, lifecycle, key attributes
- Method docstrings for all public methods: params, returns, raises
- Module-level docstrings where missing
- Same priority order as type hints

---

## Work Package 4: Consistency & Cleanup

### Custom exception hierarchy
```
TradingError (base)
├── DataFetchError      — exchange/API connectivity
├── ExecutionError      — order placement/cancellation
├── ValidationError     — config or input validation
└── ModelError          — ML training/prediction failures
```
Define in a new `exceptions.py`, use in modules that currently raise bare Exception or return error dicts.

### Logging standardization
- Every module: `_log = logging.getLogger(__name__)`
- Remove stray `print()` calls in production code paths (keep in `_print_banner()` and CLI output)
- Consistent pattern: `_log.info()` for operations, `_log.warning()` for degraded, `_log.error()` for failures

### Dead code removal
- Remove verified unused functions/imports
- Remove `indicators.invalidate_cache()` if confirmed unused

### CLAUDE.md update
- Fix stale line counts
- Add test suite commands (`make test`, individual test files)
- Add Phase 1-4 summary
- Add troubleshooting for common startup issues

---

## Verification

1. `make test` → 890 passed, 0 failed
2. `make lint` → zero errors on all changed files
3. All public functions have return type annotations
4. All public classes have docstrings
5. No stray `print()` in production modules
6. `git diff --stat` confirms scope (no accidental changes)

## Files Changed (estimated)

- ~20 production .py files (type hints + docstrings + logging)
- `requirements.txt` (remove unused deps)
- `pyproject.toml` (new)
- `Makefile` (add targets)
- `exceptions.py` (new)
- `CLAUDE.md` (update)
