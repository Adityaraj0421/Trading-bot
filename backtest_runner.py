"""
Backtest Runner v3.0
=====================
Orchestrates multi-pair, multi-scenario, multi-timeframe backtests.
Results are stored in DataStore for dashboard display.

v3.0: Added walk-forward validation and Monte Carlo robustness testing.
v2.1: passes symbol through to Backtester, flattens strategy attribution
      into results for dashboard display.
"""

from datetime import datetime
from backtester import Backtester
from scenarios import generate_scenario, list_scenarios
from data_fetcher import DataFetcher
from walk_forward import WalkForwardValidator, PurgedKFoldCV
from config import Config


class BacktestRunner:
    """Orchestrates complex backtest runs."""

    def __init__(self):
        self.results: list[dict] = []
        self.fetcher = DataFetcher()

    @staticmethod
    def _extract_strategy_stats(bt: Backtester) -> list[dict]:
        """Pull per-strategy attribution from a finished backtester."""
        stats = []
        for strat_name, trades in bt.strategy_trades.items():
            n = len(trades)
            if n == 0:
                continue
            wins = sum(1 for t in trades if t.pnl_net > 0)
            total_pnl = sum(t.pnl_net for t in trades)
            stats.append({
                "strategy": strat_name,
                "trades": n,
                "win_rate": round(wins / n * 100, 1),
                "pnl": round(total_pnl, 2),
            })
        stats.sort(key=lambda s: s["pnl"], reverse=True)
        return stats

    def run_scenario(self, scenario: str, periods: int = 500,
                     base_price: float = 100000.0) -> dict:
        """Run backtest on a synthetic market scenario."""
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

    def run_multi_pair(self, pairs: list[str] = None,
                       limit: int = 500) -> list[dict]:
        """Run backtests across multiple trading pairs."""
        pairs = pairs or Config.TRADING_PAIRS
        results = []
        for pair in pairs:
            try:
                df = self.fetcher.fetch_ohlcv(symbol=pair, limit=limit)
                if df is None or df.empty or len(df) < 100:
                    results.append({
                        "type": "multi_pair",
                        "pair": pair,
                        "error": "insufficient_data",
                    })
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

    def run_all_scenarios(self) -> list[dict]:
        """Run backtests on all predefined market scenarios."""
        results = []
        for scenario in list_scenarios():
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def run_multi_timeframe(self, timeframes: list[str] = None,
                            limit: int = 500) -> list[dict]:
        """Run backtests across multiple timeframes."""
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

    def run_walk_forward(self, pair: str = None, limit: int = 1000,
                         train_bars: int = 400, test_bars: int = 100,
                         step_bars: int = 50,
                         mc_simulations: int = 500) -> dict:
        """
        Run walk-forward validation with Monte Carlo robustness testing.
        This is the gold standard for strategy validation — trains on rolling
        windows and only reports out-of-sample performance.
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

    def run_walk_forward_multi_pair(self, pairs: list[str] = None,
                                    limit: int = 1000) -> list[dict]:
        """Run walk-forward validation across multiple trading pairs."""
        pairs = pairs or Config.TRADING_PAIRS
        results = []
        for pair in pairs:
            result = self.run_walk_forward(pair=pair, limit=limit)
            results.append(result)
        return results

    def get_all_results(self) -> list[dict]:
        return self.results
