# Phase 3: Testing Gaps — Design

## Problem
12+ production modules have zero test coverage. Most critical gaps: shutdown safety, data persistence, real-time streaming, notifications, and RL systems.

## Scope: 10 new test files + shared infrastructure

| File | Module Under Test | Test Count (est.) |
|------|-------------------|-------------------|
| `conftest.py` | Shared fixtures | — |
| `test_graceful_shutdown.py` | `graceful_shutdown.py` (GracefulShutdown + RateLimiter) | ~15 |
| `test_trade_db.py` | `trade_db.py` (SQLite CRUD, equity, events, analytics) | ~20 |
| `test_websocket_streamer.py` | `websocket_streamer.py` (message parsing, status) | ~15 |
| `test_notifier.py` | `notifier.py` (multi-channel dispatch, rate limiting) | ~12 |
| `test_market_impact.py` | `market_impact.py` (execution sim, stress, partial fills) | ~15 |
| `test_telegram_bot.py` | `telegram_bot.py` (commands, confirmations, auth) | ~12 |
| `test_async_fetcher.py` | `async_fetcher.py` (demo fallback, session cleanup) | ~8 |
| `test_logger.py` | `logger.py` (structured events, queries) | ~8 |
| `test_strategy_selector.py` | `strategy_selector.py` (DQN fallback, serialization) | ~12 |
| `test_rl_ensemble.py` | `rl_ensemble.py` (ensemble voting, tabular fallback) | ~12 |
| `test_walk_forward.py` | `walk_forward.py` (window gen, Monte Carlo) | ~10 |

**Estimated total: ~140 new tests**

## Approach
- Each test file is self-contained with minimal mocking
- External APIs (Telegram, Discord, exchanges) mocked at the `requests.post` / `websockets.connect` level
- SQLite tests use in-memory or tmp_path databases
- RL tests work with both PyTorch and tabular fallback
- All tests runnable in <30s total
