"""
Walk-Forward Validation Engine v1.0
=====================================
Rolling walk-forward optimization + Monte Carlo permutation testing.
Prevents backtest overfitting by training/testing on rolling windows
and computing confidence intervals via trade-order shuffling.

Key concepts:
  - Walk-Forward Optimization (WFO): train on N bars, test on M bars,
    roll forward by S bars, repeat. Only out-of-sample results count.
  - Monte Carlo Simulation: shuffle trade order 1000x to build
    confidence intervals. If strategy's Sharpe > 95th percentile of
    shuffled Sharpes, performance is likely real, not luck.
  - Combinatorial Purged Cross-Validation (CPCV): purges overlapping
    samples to prevent look-ahead bias in ML model evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from indicators import Indicators
from model import TradingModel
from regime_detector import RegimeDetector
from sentiment import SentimentAnalyzer
from strategies import StrategyEngine

_log = logging.getLogger(__name__)


@dataclass
class WFOWindow:
    """Represents one train/test window in the walk-forward sequence."""

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_bars: int = 0
    test_bars: int = 0
    # Results populated after backtest
    metrics: dict[str, Any] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute derived bar counts from the start/end indices."""
        self.train_bars = self.train_end - self.train_start
        self.test_bars = self.test_end - self.test_start


@dataclass
class WFOResult:
    """Aggregated walk-forward validation result."""

    n_folds: int = 0
    total_oos_bars: int = 0  # Total out-of-sample bars tested
    oos_total_return_pct: float = 0.0
    oos_sharpe: float = 0.0
    oos_max_drawdown_pct: float = 0.0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_total_trades: int = 0
    # Monte Carlo results
    mc_sharpe_mean: float = 0.0
    mc_sharpe_std: float = 0.0
    mc_sharpe_p95: float = 0.0
    mc_p_value: float = 1.0  # Probability that result is due to chance
    # Per-fold details
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    # Quality assessment
    is_robust: bool = False  # Passes all robustness checks
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the walk-forward result to a JSON-compatible dict.

        Returns:
            Dict with keys ``n_folds``, ``total_oos_bars``,
            ``oos_total_return_pct``, ``oos_sharpe``,
            ``oos_max_drawdown_pct``, ``oos_win_rate``,
            ``oos_profit_factor``, ``oos_total_trades``,
            ``mc_sharpe_mean``, ``mc_sharpe_std``, ``mc_sharpe_p95``,
            ``mc_p_value``, ``is_robust``, ``rejection_reasons``,
            and ``fold_results``.
        """
        return {
            "n_folds": self.n_folds,
            "total_oos_bars": self.total_oos_bars,
            "oos_total_return_pct": round(self.oos_total_return_pct, 2),
            "oos_sharpe": round(self.oos_sharpe, 3),
            "oos_max_drawdown_pct": round(self.oos_max_drawdown_pct, 2),
            "oos_win_rate": round(self.oos_win_rate, 1),
            "oos_profit_factor": round(self.oos_profit_factor, 2),
            "oos_total_trades": self.oos_total_trades,
            "mc_sharpe_mean": round(self.mc_sharpe_mean, 3),
            "mc_sharpe_std": round(self.mc_sharpe_std, 3),
            "mc_sharpe_p95": round(self.mc_sharpe_p95, 3),
            "mc_p_value": round(self.mc_p_value, 4),
            "is_robust": self.is_robust,
            "rejection_reasons": self.rejection_reasons,
            "fold_results": self.fold_results,
        }


class WalkForwardValidator:
    """
    Walk-Forward Optimization engine with Monte Carlo robustness testing.

    Usage:
        wfv = WalkForwardValidator(train_bars=400, test_bars=100, step_bars=50)
        result = wfv.validate(df_ohlcv)
        print(f"Robust: {result.is_robust}, OOS Sharpe: {result.oos_sharpe}")
    """

    def __init__(
        self,
        train_bars: int = 400,  # ~17 days at 1h
        test_bars: int = 100,  # ~4 days at 1h
        step_bars: int = 50,  # ~2 days at 1h
        min_trades_per_fold: int = 3,
        mc_simulations: int = 500,  # Monte Carlo shuffles
        mc_confidence: float = 0.95,  # Confidence level for MC test
        min_oos_sharpe: float = 0.3,  # Minimum OOS Sharpe to pass
        max_oos_drawdown: float = -20.0,  # Max drawdown % to pass
        fee_pct: float | None = None,
        slippage_pct: float | None = None,
        symbol: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialise the walk-forward validator with window and quality parameters.

        Args:
            train_bars: Number of bars in each training window (~17 days at 1 h).
            test_bars: Number of out-of-sample bars after each training window
                (~4 days at 1 h).
            step_bars: Number of bars to advance the window on each fold
                (~2 days at 1 h).
            min_trades_per_fold: Minimum trades required per fold for the
                total trade count check.
            mc_simulations: Number of Monte Carlo trade-order shuffles used
                to compute the null distribution of Sharpe ratios.
            mc_confidence: Confidence level for the Monte Carlo significance
                test (e.g. 0.95 means the real Sharpe must beat the 95th
                percentile of shuffled Sharpes).
            min_oos_sharpe: Minimum acceptable out-of-sample Sharpe ratio.
            max_oos_drawdown: Maximum acceptable out-of-sample drawdown (as a
                negative percentage, e.g. -20.0 means −20 %).
            fee_pct: Trading fee as a decimal fraction. Defaults to
                ``Config.FEE_PCT``.
            slippage_pct: Slippage per side as a decimal fraction. Defaults
                to ``Config.SLIPPAGE_PCT``.
            symbol: Trading pair symbol. Defaults to ``Config.TRADING_PAIR``.
            verbose: Print progress and the final report when ``True``.
        """
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.min_trades_per_fold = min_trades_per_fold
        self.mc_simulations = mc_simulations
        self.mc_confidence = mc_confidence
        self.min_oos_sharpe = min_oos_sharpe
        self.max_oos_drawdown = max_oos_drawdown
        self.fee_pct = fee_pct or Config.FEE_PCT
        self.slippage_pct = slippage_pct or Config.SLIPPAGE_PCT
        self.symbol = symbol or Config.TRADING_PAIR
        self.verbose = verbose

    def _generate_windows(self, n_bars: int) -> list[WFOWindow]:
        """Generate rolling train/test windows."""
        windows = []
        fold_id = 0
        start = 0

        while start + self.train_bars + self.test_bars <= n_bars:
            train_start = start
            train_end = start + self.train_bars
            test_start = train_end
            test_end = min(test_start + self.test_bars, n_bars)

            windows.append(
                WFOWindow(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_id += 1
            start += self.step_bars

        return windows

    def _backtest_fold(self, df_ind: pd.DataFrame, window: WFOWindow) -> dict[str, Any]:
        """
        Run a single fold: train on training window, then test on test window.
        Returns metrics dict + list of trade PnLs.
        """
        # Create fresh components for this fold (prevents state leakage)
        model = TradingModel()
        regime_detector = RegimeDetector()
        sentiment = SentimentAnalyzer()
        strategy_engine = StrategyEngine()

        # Train on training window
        train_data = df_ind.iloc[window.train_start : window.train_end]
        model.train(df=None, df_ind=train_data)

        # Walk through test window
        capital = Config.INITIAL_CAPITAL
        initial_capital = capital
        positions = []
        trades = []
        equity_curve = [capital]

        for i in range(window.test_start, window.test_end):
            row = df_ind.iloc[i]
            price = row["close"]

            # Mark-to-market
            unrealized = sum(
                (price - p["entry"]) * p["qty"] if p["side"] == "long" else (p["entry"] - price) * p["qty"]
                for p in positions
            )
            equity_curve.append(capital + unrealized + sum(p["entry"] * p["qty"] for p in positions))

            # Check existing positions for exits
            for pos in list(positions):
                bars_held = i - pos["entry_bar"]
                exit_reason = None

                if pos["side"] == "long":
                    if row["high"] > pos["highest"]:
                        pos["highest"] = row["high"]
                        pos["trail"] = max(pos["trail"], row["high"] * (1 - Config.TRAILING_STOP_PCT))
                    if row["low"] <= pos["sl"]:
                        exit_reason = "stop_loss"
                        exit_price = pos["sl"]
                    elif row["high"] >= pos["tp"]:
                        exit_reason = "take_profit"
                        exit_price = pos["tp"]
                    elif row["low"] <= pos["trail"] and pos["trail"] > pos["sl"]:
                        exit_reason = "trailing_stop"
                        exit_price = pos["trail"]
                    elif bars_held >= Config.MAX_HOLD_BARS:
                        exit_reason = "max_duration"
                        exit_price = price
                else:  # short
                    if row["low"] < pos["lowest"]:
                        pos["lowest"] = row["low"]
                        pos["trail"] = min(pos["trail"], row["low"] * (1 + Config.TRAILING_STOP_PCT))
                    if row["high"] >= pos["sl"]:
                        exit_reason = "stop_loss"
                        exit_price = pos["sl"]
                    elif row["low"] <= pos["tp"]:
                        exit_reason = "take_profit"
                        exit_price = pos["tp"]
                    elif row["high"] >= pos["trail"] and pos["trail"] < pos["sl"]:
                        exit_reason = "trailing_stop"
                        exit_price = pos["trail"]
                    elif bars_held >= Config.MAX_HOLD_BARS:
                        exit_reason = "max_duration"
                        exit_price = price

                if exit_reason:
                    # Apply slippage + fees
                    if pos["side"] == "long":
                        actual_exit = exit_price * (1 - self.slippage_pct)
                        pnl = (actual_exit - pos["actual_entry"]) * pos["qty"]
                    else:
                        actual_exit = exit_price * (1 + self.slippage_pct)
                        pnl = (pos["actual_entry"] - actual_exit) * pos["qty"]
                    fee = actual_exit * pos["qty"] * self.fee_pct
                    pnl_net = pnl - fee

                    trades.append(
                        {
                            "pnl_net": pnl_net,
                            "side": pos["side"],
                            "hold_bars": bars_held,
                            "exit_reason": exit_reason,
                        }
                    )
                    capital += actual_exit * pos["qty"] - fee
                    positions.remove(pos)

            # Skip if at max positions
            if len(positions) >= Config.MAX_OPEN_POSITIONS:
                continue

            # Get signals from strategy pipeline
            lookback = max(0, i - 200)
            analysis_window = df_ind.iloc[lookback : i + 1]
            if len(analysis_window) < 50:
                continue

            regime_state = regime_detector.detect(analysis_window, df_ind=analysis_window)
            sentiment_state = sentiment.analyze(analysis_window, df_ind=analysis_window)
            strat_signal = strategy_engine.run(analysis_window, regime_state.regime, sentiment_state)
            ml_signal, ml_conf = model.predict(df_ind=analysis_window)

            # Combine signals (same logic as backtester)
            final_signal, final_conf = self._combine(strat_signal, ml_signal, ml_conf)

            # Execute
            if final_signal in ("BUY", "SELL") and final_conf >= Config.MIN_CONFIDENCE:
                target_side = "long" if final_signal == "BUY" else "short"
                has_conflict = any(True for p in positions)
                if not has_conflict:
                    risk_amount = capital * Config.MAX_POSITION_PCT
                    if target_side == "long":
                        actual_entry = price * (1 + self.slippage_pct)
                    else:
                        actual_entry = price * (1 - self.slippage_pct)
                    qty = risk_amount / actual_entry
                    entry_fee = actual_entry * qty * self.fee_pct

                    if actual_entry * qty + entry_fee <= capital:
                        capital -= actual_entry * qty + entry_fee

                        sl_pct = strat_signal.suggested_sl_pct
                        tp_pct = strat_signal.suggested_tp_pct
                        if target_side == "long":
                            sl = actual_entry * (1 - sl_pct)
                            tp = actual_entry * (1 + tp_pct)
                            trail = actual_entry * (1 - Config.TRAILING_STOP_PCT)
                        else:
                            sl = actual_entry * (1 + sl_pct)
                            tp = actual_entry * (1 - tp_pct)
                            trail = actual_entry * (1 + Config.TRAILING_STOP_PCT)

                        positions.append(
                            {
                                "side": target_side,
                                "entry": price,
                                "actual_entry": actual_entry,
                                "qty": qty,
                                "entry_bar": i,
                                "sl": sl,
                                "tp": tp,
                                "trail": trail,
                                "highest": price,
                                "lowest": price,
                            }
                        )

        # Close remaining positions at last price
        if positions:
            last_price = df_ind["close"].iloc[window.test_end - 1]
            for pos in list(positions):
                if pos["side"] == "long":
                    actual_exit = last_price * (1 - self.slippage_pct)
                    pnl = (actual_exit - pos["actual_entry"]) * pos["qty"]
                else:
                    actual_exit = last_price * (1 + self.slippage_pct)
                    pnl = (pos["actual_entry"] - actual_exit) * pos["qty"]
                fee = actual_exit * pos["qty"] * self.fee_pct
                trades.append(
                    {
                        "pnl_net": pnl - fee,
                        "side": pos["side"],
                        "hold_bars": window.test_end - 1 - pos["entry_bar"],
                        "exit_reason": "fold_end",
                    }
                )
                capital += actual_exit * pos["qty"] - fee

        equity_curve.append(capital)

        # Compute fold metrics
        metrics = self._compute_fold_metrics(equity_curve, trades, initial_capital)
        return {"metrics": metrics, "trades": trades}

    def _combine(self, strat_sig: Any, ml_signal: str, ml_conf: float) -> tuple[str, float]:
        """Combine strategy and ML signals (mirrors backtester logic)."""
        s = strat_sig.signal
        sc = strat_sig.confidence
        if s == ml_signal:
            return s, min(sc * 0.6 + ml_conf * 0.4 + 0.1, 0.95)
        elif s == "HOLD" or ml_signal == "HOLD":
            active = s if s != "HOLD" else ml_signal
            active_c = sc if s != "HOLD" else ml_conf
            return active, active_c * 0.6
        else:
            return s, sc * 0.4

    def _compute_fold_metrics(
        self, equity_curve: list[float], trades: list[dict[str, Any]], initial_capital: float
    ) -> dict[str, Any]:
        """Compute performance metrics for a single fold."""
        equity = np.array(equity_curve)
        if len(equity) < 2:
            return {
                "total_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0,
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
            }

        total_return = (equity[-1] / equity[0]) - 1
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        # Sharpe (annualized for hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(np.min(drawdown)) * 100

        n_trades = len(trades)
        if n_trades == 0:
            return {
                "total_return_pct": round(total_return * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(max_dd, 2),
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
            }

        wins = [t for t in trades if t["pnl_net"] > 0]
        losses = [t for t in trades if t["pnl_net"] <= 0]
        win_rate = len(wins) / n_trades * 100
        gross_profit = sum(t["pnl_net"] for t in wins)
        gross_loss = abs(sum(t["pnl_net"] for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": n_trades,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
        }

    def monte_carlo_test(self, trades: list[dict[str, Any]], n_simulations: int | None = None) -> dict[str, Any]:
        """Run a Monte Carlo permutation significance test on trade PnLs.

        Shuffles the trade order ``n_simulations`` times and computes the
        Sharpe ratio of each shuffled sequence to build the null distribution.
        If the real Sharpe exceeds the 95th percentile, the strategy's edge is
        statistically significant.

        Args:
            trades: List of trade dicts, each containing a ``pnl_net`` key.
            n_simulations: Number of shuffle iterations.  Defaults to
                ``self.mc_simulations``.

        Returns:
            Dict with keys:
                ``real_sharpe`` (float): Sharpe of the actual trade sequence.
                ``mean_sharpe`` (float): Mean Sharpe across shuffled sequences.
                ``std_sharpe`` (float): Std dev of shuffled Sharpes.
                ``p95_sharpe`` (float): 95th-percentile Sharpe of shuffled sequences.
                ``p_value`` (float): Fraction of shuffled runs that beat the
                real Sharpe (lower is better; < 0.05 is significant).
        """
        n_sims = n_simulations or self.mc_simulations
        if len(trades) < 5:
            return {
                "mean_sharpe": 0,
                "std_sharpe": 0,
                "p95_sharpe": 0,
                "p_value": 1.0,
            }

        pnls = [t["pnl_net"] for t in trades]
        real_sharpe = self._pnl_sharpe(pnls)

        # Shuffle trades and recompute Sharpe each time
        shuffled_sharpes = []
        for _ in range(n_sims):
            shuffled = pnls.copy()
            np.random.shuffle(shuffled)
            shuffled_sharpes.append(self._pnl_sharpe(shuffled))

        sharpes = np.array(shuffled_sharpes)
        p95 = float(np.percentile(sharpes, 95))
        # p-value: fraction of shuffled runs that beat the real Sharpe
        p_value = float(np.mean(sharpes >= real_sharpe))

        return {
            "real_sharpe": round(real_sharpe, 3),
            "mean_sharpe": round(float(np.mean(sharpes)), 3),
            "std_sharpe": round(float(np.std(sharpes)), 3),
            "p95_sharpe": round(p95, 3),
            "p_value": round(p_value, 4),
        }

    def _pnl_sharpe(self, pnls: list[float]) -> float:
        """Compute Sharpe ratio from a sequence of trade PnLs."""
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls, dtype=float)
        std = np.std(arr)
        if std == 0:
            return 0.0
        # Annualize assuming ~250 trading days, ~3 trades/day
        return float(np.mean(arr) / std * np.sqrt(750))

    def validate(self, df: pd.DataFrame) -> WFOResult:
        """
        Run full walk-forward validation with Monte Carlo robustness test.

        Args:
            df: OHLCV DataFrame with timestamp index

        Returns:
            WFOResult with aggregated out-of-sample metrics + MC test results
        """
        if self.verbose:
            print("=" * 60)
            print("  WALK-FORWARD VALIDATION ENGINE v1.0")
            print(f"  Train: {self.train_bars} bars | Test: {self.test_bars} bars | Step: {self.step_bars} bars")
            print(f"  Data: {len(df)} bars | Symbol: {self.symbol}")
            print("=" * 60)

        # Compute indicators on full dataset
        Indicators.invalidate_cache()
        df_ind = Indicators.add_all(df)

        if len(df_ind) < self.train_bars + self.test_bars + 50:
            return WFOResult(rejection_reasons=["insufficient_data"])

        # Generate rolling windows
        windows = self._generate_windows(len(df_ind))
        if not windows:
            return WFOResult(rejection_reasons=["no_valid_windows"])

        if self.verbose:
            print(f"  Generated {len(windows)} walk-forward folds\n")

        # Run each fold
        all_oos_trades = []
        all_oos_equity = [Config.INITIAL_CAPITAL]
        fold_results = []
        running_capital = Config.INITIAL_CAPITAL

        for window in windows:
            if self.verbose:
                print(
                    f"  Fold {window.fold_id + 1}/{len(windows)}: "
                    f"Train [{window.train_start}:{window.train_end}] "
                    f"Test [{window.test_start}:{window.test_end}]",
                    end="",
                )

            fold_output = self._backtest_fold(df_ind, window)
            fold_metrics = fold_output["metrics"]
            fold_trades = fold_output["trades"]

            fold_results.append(
                {
                    "fold_id": window.fold_id,
                    "train_bars": window.train_bars,
                    "test_bars": window.test_bars,
                    **fold_metrics,
                }
            )

            all_oos_trades.extend(fold_trades)

            # Track running equity from fold returns
            fold_return = fold_metrics.get("total_return_pct", 0) / 100
            running_capital *= 1 + fold_return
            all_oos_equity.append(running_capital)

            if self.verbose:
                n_trades = fold_metrics.get("total_trades", 0)
                ret = fold_metrics.get("total_return_pct", 0)
                sr = fold_metrics.get("sharpe_ratio", 0)
                print(f" -> {n_trades} trades, {ret:+.2f}%, Sharpe={sr:.2f}")

        # Aggregate OOS metrics
        result = WFOResult(
            n_folds=len(windows),
            total_oos_bars=sum(w.test_bars for w in windows),
            fold_results=fold_results,
        )

        # Compute aggregate OOS metrics from all trades
        if all_oos_trades:
            agg = self._compute_aggregate_metrics(all_oos_equity, all_oos_trades)
            result.oos_total_return_pct = agg["total_return_pct"]
            result.oos_sharpe = agg["sharpe_ratio"]
            result.oos_max_drawdown_pct = agg["max_drawdown_pct"]
            result.oos_win_rate = agg["win_rate"]
            result.oos_profit_factor = agg["profit_factor"]
            result.oos_total_trades = agg["total_trades"]

        # Monte Carlo test
        if self.verbose:
            print(f"\n  Running Monte Carlo test ({self.mc_simulations} simulations)...")

        mc_results = self.monte_carlo_test(all_oos_trades)
        result.mc_sharpe_mean = mc_results["mean_sharpe"]
        result.mc_sharpe_std = mc_results["std_sharpe"]
        result.mc_sharpe_p95 = mc_results["p95_sharpe"]
        result.mc_p_value = mc_results["p_value"]

        # Robustness assessment
        result.is_robust, result.rejection_reasons = self._assess_robustness(result)

        if self.verbose:
            self._print_report(result, mc_results)

        return result

    def _compute_aggregate_metrics(self, equity_curve: list[float], trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate metrics from all OOS folds."""
        equity = np.array(equity_curve)
        total_return = (equity[-1] / equity[0]) - 1

        # Fold-level returns for Sharpe
        fold_returns = np.diff(equity) / equity[:-1]
        fold_returns = fold_returns[np.isfinite(fold_returns)]

        if len(fold_returns) > 1 and np.std(fold_returns) > 0:
            # Annualize based on ~52 folds/year (weekly steps)
            sharpe = np.mean(fold_returns) / np.std(fold_returns) * np.sqrt(52)
        else:
            sharpe = 0.0

        # Max drawdown from equity curve
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(np.min(drawdown)) * 100

        n_trades = len(trades)
        wins = [t for t in trades if t["pnl_net"] > 0]
        losses = [t for t in trades if t["pnl_net"] <= 0]
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
        gross_profit = sum(t["pnl_net"] for t in wins)
        gross_loss = abs(sum(t["pnl_net"] for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            "total_return_pct": round(total_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": n_trades,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
        }

    def _assess_robustness(self, result: WFOResult) -> tuple[bool, list[str]]:
        """
        Assess whether the strategy passes robustness checks.
        Returns (is_robust, list_of_rejection_reasons).
        """
        reasons = []

        # Check 1: Minimum OOS Sharpe
        if result.oos_sharpe < self.min_oos_sharpe:
            reasons.append(f"OOS Sharpe {result.oos_sharpe:.3f} < min {self.min_oos_sharpe}")

        # Check 2: Maximum drawdown
        if result.oos_max_drawdown_pct < self.max_oos_drawdown:
            reasons.append(f"OOS drawdown {result.oos_max_drawdown_pct:.1f}% < limit {self.max_oos_drawdown}%")

        # Check 3: Minimum trades
        if result.oos_total_trades < self.min_trades_per_fold * result.n_folds:
            reasons.append(
                f"Too few OOS trades: {result.oos_total_trades} < {self.min_trades_per_fold * result.n_folds}"
            )

        # Check 4: Monte Carlo significance
        if result.mc_p_value > (1 - self.mc_confidence):
            reasons.append(
                f"MC p-value {result.mc_p_value:.3f} > "
                f"threshold {1 - self.mc_confidence:.3f} "
                f"(performance may be due to luck)"
            )

        # Check 5: Consistency across folds
        fold_returns = [f.get("total_return_pct", 0) for f in result.fold_results]
        if fold_returns:
            n_positive = sum(1 for r in fold_returns if r > 0)
            positive_pct = n_positive / len(fold_returns) * 100
            if positive_pct < 40:
                reasons.append(f"Only {positive_pct:.0f}% of folds profitable (need >40%)")

        return len(reasons) == 0, reasons

    def _print_report(self, result: WFOResult, mc_results: dict[str, Any]) -> None:
        """Print walk-forward validation report."""
        print("\n" + "=" * 60)
        print("  WALK-FORWARD VALIDATION RESULTS")
        print("=" * 60)
        print(f"  Folds:            {result.n_folds}")
        print(f"  OOS Bars:         {result.total_oos_bars}")
        print(f"  OOS Trades:       {result.oos_total_trades}")
        print()
        print(f"  OOS Return:       {result.oos_total_return_pct:+.2f}%")
        print(f"  OOS Sharpe:       {result.oos_sharpe:.3f}")
        print(f"  OOS Max DD:       {result.oos_max_drawdown_pct:.2f}%")
        print(f"  OOS Win Rate:     {result.oos_win_rate:.1f}%")
        print(f"  OOS Profit Factor:{result.oos_profit_factor:.2f}")
        print()
        print(f"  Monte Carlo Test ({self.mc_simulations} sims):")
        print(f"    Real Sharpe:    {mc_results.get('real_sharpe', 0):.3f}")
        print(f"    Shuffled Mean:  {result.mc_sharpe_mean:.3f} +/- {result.mc_sharpe_std:.3f}")
        print(f"    95th Pctile:    {result.mc_sharpe_p95:.3f}")
        print(f"    p-value:        {result.mc_p_value:.4f}")
        print()

        if result.is_robust:
            print("  VERDICT: ROBUST - Strategy passes all checks")
        else:
            print("  VERDICT: REJECTED - Strategy fails robustness checks:")
            for reason in result.rejection_reasons:
                print(f"    - {reason}")

        # Per-fold summary
        print("\n  Per-Fold Performance:")
        print(f"  {'Fold':>4s} {'Trades':>6s} {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'WinR':>5s}")
        print(f"  {'-' * 40}")
        for f in result.fold_results:
            print(
                f"  {f['fold_id'] + 1:>4d} "
                f"{f.get('total_trades', 0):>6d} "
                f"{f.get('total_return_pct', 0):>+7.2f}% "
                f"{f.get('sharpe_ratio', 0):>7.3f} "
                f"{f.get('max_drawdown_pct', 0):>6.2f}% "
                f"{f.get('win_rate', 0):>4.0f}%"
            )

        print("=" * 60)


class PurgedKFoldCV:
    """Combinatorial Purged Cross-Validation for ML model evaluation.

    Purges training samples that are within a lookback window of the test
    boundary to prevent look-ahead bias when evaluating the ML model on
    time-series data.
    """

    def __init__(self, n_splits: int = 5, purge_window: int = 30) -> None:
        """Initialise the purged k-fold splitter.

        Args:
            n_splits: Number of folds.
            purge_window: Number of bars on each side of the test boundary to
                exclude from training to prevent look-ahead contamination.
        """
        self.n_splits = n_splits
        self.purge_window = purge_window

    def split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test index pairs.

        Each fold's training set excludes a ``purge_window``-bar buffer zone
        around the test fold boundaries to prevent label leakage.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of ``(train_indices, test_indices)`` tuples as numpy arrays.
            Folds where either split has zero samples are omitted.
        """
        fold_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)

            # Purge zone: exclude training samples near test boundary
            purge_start = max(0, test_start - self.purge_window)
            purge_end = min(n_samples, test_end + self.purge_window)

            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate(
                [
                    np.arange(0, purge_start),
                    np.arange(purge_end, n_samples),
                ]
            )

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def evaluate_model(self, model: TradingModel, df_ind: pd.DataFrame) -> dict[str, Any]:
        """Evaluate the ML model using purged k-fold cross-validation.

        Trains a fresh ``TradingModel`` on each fold's purged training set and
        scores it on the held-out test set.  Accuracy is measured as the
        fraction of test bars where the predicted signal matches the label
        derived from the forward return.

        Args:
            model: A ``TradingModel`` instance (used only to determine the
                class; a fresh copy is trained per fold).
            df_ind: Indicator-augmented OHLCV DataFrame with a
                ``future_return`` column.

        Returns:
            Dict with keys:
                ``cv_accuracy`` (float): Mean accuracy across all folds.
                ``cv_std`` (float): Standard deviation of fold accuracies.
                ``n_folds`` (int): Number of folds that produced predictions.
                ``fold_accuracies`` (list[float]): Per-fold accuracy scores.
        """
        from indicators import FEATURE_COLUMNS
        from model import Signal

        X = df_ind[FEATURE_COLUMNS].values
        future_return = df_ind["future_return"].values

        # Create labels
        y = np.full(len(future_return), Signal.HOLD)
        y[future_return > 0.01] = Signal.BUY
        y[future_return < -0.01] = Signal.SELL

        splits = self.split(len(X))
        fold_accuracies = []

        for train_idx, test_idx in splits:
            # Create a fresh model for each fold
            fold_model = TradingModel()

            # Build a mini dataframe for training
            train_df = df_ind.iloc[train_idx].copy()
            fold_model.train(df=None, df_ind=train_df)

            if not fold_model.is_trained:
                continue

            # Predict on test set
            correct = 0
            total = 0
            for idx in test_idx:
                if idx < 50:
                    continue
                window = df_ind.iloc[max(0, idx - 50) : idx + 1]
                pred_signal, pred_conf = fold_model.predict(df_ind=window)

                actual = y[idx]
                if pred_signal == actual:
                    correct += 1
                total += 1

            if total > 0:
                fold_accuracies.append(correct / total)

        if not fold_accuracies:
            return {"cv_accuracy": 0, "cv_std": 0, "n_folds": 0}

        return {
            "cv_accuracy": round(float(np.mean(fold_accuracies)), 4),
            "cv_std": round(float(np.std(fold_accuracies)), 4),
            "n_folds": len(fold_accuracies),
            "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        }
