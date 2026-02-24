"""
Backtest Runner v3.0
=====================
Orchestrates multi-pair, multi-scenario, multi-timeframe backtests.
Results are stored in DataStore for dashboard display.

v3.0: Added walk-forward validation and Monte Carlo robustness testing.
v2.1: passes symbol through to Backtester, flattens strategy attribution
      into results for dashboard display.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from backtester import Backtester
from config import Config
from data_fetcher import DataFetcher
from scenarios import generate_scenario, list_scenarios
from walk_forward import WalkForwardValidator

_log = logging.getLogger(__name__)


class BacktestRunner:
    """Orchestrates complex backtest runs across pairs, scenarios, and timeframes.

    Provides convenience methods for:

    - Synthetic scenario backtests (``run_scenario``, ``run_all_scenarios``).
    - Multi-pair live-data backtests (``run_multi_pair``).
    - Multi-timeframe backtests (``run_multi_timeframe``).
    - Walk-forward validation with Monte Carlo robustness testing
      (``run_walk_forward``, ``run_walk_forward_multi_pair``).

    All results are accumulated in ``self.results`` for later retrieval via
    ``get_all_results()``.
    """

    def __init__(self) -> None:
        """Initialise the runner with a fresh result list and data fetcher."""
        self.results: list[dict[str, Any]] = []
        self.fetcher = DataFetcher()

    @staticmethod
    def _extract_strategy_stats(bt: Backtester) -> list[dict[str, Any]]:
        """Pull per-strategy attribution from a finished backtester."""
        stats = []
        for strat_name, trades in bt.strategy_trades.items():
            n = len(trades)
            if n == 0:
                continue
            wins = sum(1 for t in trades if t.pnl_net > 0)
            total_pnl = sum(t.pnl_net for t in trades)
            stats.append(
                {
                    "strategy": strat_name,
                    "trades": n,
                    "win_rate": round(wins / n * 100, 1),
                    "pnl": round(total_pnl, 2),
                }
            )
        stats.sort(key=lambda s: s["pnl"], reverse=True)
        return stats

    def run_scenario(self, scenario: str, periods: int = 500, base_price: float = 100000.0) -> dict[str, Any]:
        """Run a backtest on a synthetic market scenario.

        Args:
            scenario: Scenario name as recognised by ``generate_scenario()``
                (e.g. "trending_up", "ranging", "volatile_crash").
            periods: Number of OHLCV bars to generate.
            base_price: Starting price for the synthetic data.

        Returns:
            Result dict with keys ``type`` ("scenario"), ``scenario``,
            ``pair`` ("SYNTHETIC"), ``timeframe``, ``periods``, ``run_at``
            (ISO-8601 string), ``metrics`` (from ``Backtester.run()``),
            ``strategy_stats`` (per-strategy attribution list), and
            ``equity_curve`` (last 200 equity values).
        """
        df = generate_scenario(scenario, periods=periods, base_price=base_price)
        bt = Backtester(
            fee_pct=Config.FEE_PCT,
            slippage_pct=Config.SLIPPAGE_PCT,
            symbol="SYNTHETIC",
        )
        metrics = bt.run(df, verbose=False)
        result = {
            "type": "scenario",
            "scenario": scenario,
            "pair": "SYNTHETIC",
            "timeframe": Config.TIMEFRAME,
            "periods": periods,
            "run_at": datetime.now().isoformat(),
            "metrics": metrics,
            "strategy_stats": self._extract_strategy_stats(bt),
            "equity_curve": bt.equity_curve[-200:] if bt.equity_curve else [],
        }
        self.results.append(result)
        return result

    def run_multi_pair(self, pairs: list[str] | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Run backtests across multiple trading pairs.

        Fetches live OHLCV data for each pair and runs a full backtest.
        Pairs with insufficient data or fetch errors produce error entries
        in the returned list instead of raising.

        Args:
            pairs: List of trading pair symbols to test. Defaults to
                ``Config.TRADING_PAIRS`` when ``None``.
            limit: Maximum number of OHLCV bars to fetch per pair.

        Returns:
            List of result dicts (one per pair) with keys ``type``
            ("multi_pair"), ``pair``, and either a full result payload
            (``timeframe``, ``periods``, ``run_at``, ``metrics``,
            ``strategy_stats``, ``equity_curve``) or an ``error`` key.
        """
        pairs = pairs or Config.TRADING_PAIRS
        results = []
        for pair in pairs:
            try:
                df = self.fetcher.fetch_ohlcv(symbol=pair, limit=limit)
                if df is None or df.empty or len(df) < 100:
                    results.append(
                        {
                            "type": "multi_pair",
                            "pair": pair,
                            "error": "insufficient_data",
                        }
                    )
                    continue

                bt = Backtester(
                    fee_pct=Config.FEE_PCT,
                    slippage_pct=Config.SLIPPAGE_PCT,
                    symbol=pair,
                )
                metrics = bt.run(df, verbose=False)
                result = {
                    "type": "multi_pair",
                    "pair": pair,
                    "timeframe": Config.TIMEFRAME,
                    "periods": len(df),
                    "run_at": datetime.now().isoformat(),
                    "metrics": metrics,
                    "strategy_stats": self._extract_strategy_stats(bt),
                    "equity_curve": bt.equity_curve[-200:] if bt.equity_curve else [],
                }
                results.append(result)
                self.results.append(result)
            except Exception as e:
                results.append({"type": "multi_pair", "pair": pair, "error": str(e)})

        return results

    def run_all_scenarios(self) -> list[dict[str, Any]]:
        """Run backtests on every predefined market scenario.

        Iterates over all scenario names returned by ``list_scenarios()``
        and calls ``run_scenario()`` for each one.

        Returns:
            List of result dicts, one per scenario.
        """
        results = []
        for scenario in list_scenarios():
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def run_multi_timeframe(self, timeframes: list[str] | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Run backtests across multiple timeframes for the default pair.

        Args:
            timeframes: List of timeframe strings to test (e.g.
                ``["15m", "1h", "4h"]``). Defaults to those three when
                ``None``.
            limit: Maximum number of OHLCV bars to fetch per timeframe.

        Returns:
            List of result dicts (one per timeframe) with keys ``type``
            ("multi_timeframe"), ``pair``, ``timeframe``, ``periods``,
            ``run_at``, ``metrics``, ``strategy_stats``, and
            ``equity_curve``, or an ``error`` key on failure.
        """
        timeframes = timeframes or ["15m", "1h", "4h"]
        results = []
        for tf in timeframes:
            try:
                df = self.fetcher.fetch_ohlcv(timeframe=tf, limit=limit)
                if df is None or df.empty or len(df) < 100:
                    continue
                bt = Backtester(symbol=Config.TRADING_PAIR)
                metrics = bt.run(df, verbose=False)
                result = {
                    "type": "multi_timeframe",
                    "pair": Config.TRADING_PAIR,
                    "timeframe": tf,
                    "periods": len(df),
                    "run_at": datetime.now().isoformat(),
                    "metrics": metrics,
                    "strategy_stats": self._extract_strategy_stats(bt),
                    "equity_curve": bt.equity_curve[-200:] if bt.equity_curve else [],
                }
                results.append(result)
                self.results.append(result)
            except Exception as e:
                results.append({"type": "multi_timeframe", "timeframe": tf, "error": str(e)})

        return results

    def run_walk_forward(
        self,
        pair: str | None = None,
        limit: int = 1000,
        train_bars: int = 400,
        test_bars: int = 100,
        step_bars: int = 50,
        mc_simulations: int = 500,
    ) -> dict[str, Any]:
        """Run walk-forward validation with Monte Carlo robustness testing.

        This is the gold-standard validation method.  Trains on rolling
        windows and reports only out-of-sample performance, combined with
        a Monte Carlo permutation test to distinguish real edge from luck.

        Args:
            pair: Trading pair symbol. Defaults to ``Config.TRADING_PAIR``.
            limit: Maximum OHLCV bars to fetch.
            train_bars: Bars per training window (passed to
                ``WalkForwardValidator``).
            test_bars: Bars per test window.
            step_bars: Bars to advance the window each fold.
            mc_simulations: Number of Monte Carlo trade-order shuffles.

        Returns:
            Result dict with keys ``type`` ("walk_forward"), ``pair``,
            ``timeframe``, ``periods``, ``run_at``, ``wfo``
            (``WFOResult.to_dict()``), and ``is_robust``.  On error,
            returns a dict with ``type``, ``pair``, and ``error``.
        """
        pair = pair or Config.TRADING_PAIR
        try:
            df = self.fetcher.fetch_ohlcv(symbol=pair, limit=limit)
            if df is None or df.empty or len(df) < train_bars + test_bars + 50:
                return {
                    "type": "walk_forward",
                    "pair": pair,
                    "error": "insufficient_data",
                }

            wfv = WalkForwardValidator(
                train_bars=train_bars,
                test_bars=test_bars,
                step_bars=step_bars,
                mc_simulations=mc_simulations,
                symbol=pair,
                verbose=True,
            )
            wfo_result = wfv.validate(df)

            result = {
                "type": "walk_forward",
                "pair": pair,
                "timeframe": Config.TIMEFRAME,
                "periods": len(df),
                "run_at": datetime.now().isoformat(),
                "wfo": wfo_result.to_dict(),
                "is_robust": wfo_result.is_robust,
            }
            self.results.append(result)
            return result

        except Exception as e:
            return {"type": "walk_forward", "pair": pair, "error": str(e)}

    def run_walk_forward_multi_pair(self, pairs: list[str] | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        """Run walk-forward validation across multiple trading pairs.

        Args:
            pairs: Trading pair symbols to validate. Defaults to
                ``Config.TRADING_PAIRS`` when ``None``.
            limit: Maximum OHLCV bars to fetch per pair.

        Returns:
            List of result dicts (one per pair) as returned by
            ``run_walk_forward()``.
        """
        pairs = pairs or Config.TRADING_PAIRS
        results = []
        for pair in pairs:
            result = self.run_walk_forward(pair=pair, limit=limit)
            results.append(result)
        return results

    def get_all_results(self) -> list[dict[str, Any]]:
        """Return all backtest results accumulated during this session.

        Returns:
            List of all result dicts appended by previous ``run_*`` calls.
        """
        return self.results
