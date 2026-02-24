"""
Backtesting Engine v1.0
========================
Replays historical OHLCV data through the full strategy pipeline.
Computes: equity curve, Sharpe ratio, max drawdown, Sortino, Calmar,
win rate, profit factor, average trade duration, per-strategy attribution.

Includes transaction cost modeling (maker/taker fees + slippage).

Usage:
    from backtester import Backtester
    bt = Backtester(initial_capital=10000, fee_pct=0.001, slippage_pct=0.0005)
    results = bt.run(df_ohlcv)
    bt.print_report()
    bt.plot_equity_curve("equity.html")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
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
class BacktestTrade:
    """Record of a completed backtest trade with costs and attribution.

    Attributes:
        symbol: Trading pair symbol (e.g. "BTC/USDT").
        side: Trade direction — "long" or "short".
        entry_price: Raw fill price at entry (before slippage).
        exit_price: Raw fill price at exit (before slippage).
        entry_price_actual: Entry price after slippage applied.
        exit_price_actual: Exit price after slippage applied.
        quantity: Asset quantity traded.
        pnl_gross: Gross profit/loss before fees.
        pnl_net: Net profit/loss after fees and slippage costs.
        fees_paid: Total maker/taker fees paid (entry + exit).
        slippage_cost: Total slippage cost (entry + exit).
        entry_time: Timestamp of trade entry.
        exit_time: Timestamp of trade exit.
        exit_reason: Why the trade was closed (e.g. "stop_loss",
            "take_profit", "trailing_stop", "max_duration", "backtest_end").
        strategy_name: Name of the strategy that generated the signal.
        regime: Market regime at time of entry (e.g. "trending_up").
        hold_bars: Number of bars the position was held.
    """

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_price_actual: float  # After slippage
    exit_price_actual: float  # After slippage
    quantity: float
    pnl_gross: float  # Before fees
    pnl_net: float  # After fees + slippage
    fees_paid: float
    slippage_cost: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    strategy_name: str
    regime: str
    hold_bars: int


@dataclass
class BacktestPosition:
    """Tracks an open position during backtesting.

    Attributes:
        symbol: Trading pair symbol.
        side: Position direction — "long" or "short".
        entry_price: Mid-market price at entry bar.
        entry_price_actual: Actual fill price after slippage.
        quantity: Asset quantity held.
        entry_bar: Bar index at entry (relative to test window).
        entry_time: Timestamp of entry.
        stop_loss: Static stop-loss price level.
        take_profit: Static take-profit price level.
        trailing_stop: Current trailing stop level (adjusted upward for longs).
        highest_price: Highest observed close since entry (used for long trailing stop).
        lowest_price: Lowest observed close since entry (used for short trailing stop).
        strategy_name: Strategy that generated the entry signal.
        regime: Market regime at time of entry.
    """

    symbol: str
    side: str
    entry_price: float
    entry_price_actual: float
    quantity: float
    entry_bar: int
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: float  # Trailing stop level
    highest_price: float  # For trailing stop (longs)
    lowest_price: float  # For trailing stop (shorts)
    strategy_name: str
    regime: str


class Backtester:
    """
    Full-featured backtesting engine with transaction costs, trailing stops,
    and per-strategy performance attribution.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        fee_pct: float = 0.001,  # 0.1% maker/taker fee (Binance default)
        slippage_pct: float = 0.0005,  # 0.05% slippage estimate
        trailing_stop_pct: float = 0.015,  # 1.5% trailing stop activation
        max_hold_bars: int = 100,  # Force close after 100 bars
        min_confidence: float | None = None,
        symbol: str | None = None,  # v7.0: explicit symbol for multi-pair
    ) -> None:
        """Initialise the backtesting engine with cost and risk parameters.

        Args:
            initial_capital: Starting capital in USDT. Defaults to
                ``Config.INITIAL_CAPITAL`` when ``None``.
            fee_pct: Maker/taker fee as a decimal fraction (e.g. 0.001 = 0.1%).
            slippage_pct: Estimated slippage per side as a decimal fraction.
            trailing_stop_pct: Trailing stop activation distance as a decimal
                fraction of the current price.
            max_hold_bars: Maximum number of bars a position may be held before
                forced closure.
            min_confidence: Minimum signal confidence to open a position.
                Defaults to ``Config.MIN_CONFIDENCE`` when ``None``.
            symbol: Trading pair symbol to use for position tracking.
                Defaults to ``Config.TRADING_PAIR`` when ``None``.
        """
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_bars = max_hold_bars
        self.min_confidence = min_confidence or Config.MIN_CONFIDENCE
        self.symbol = symbol or Config.TRADING_PAIR

        # State
        self.capital = self.initial_capital
        self.positions: list[BacktestPosition] = []
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[float] = []
        self.equity_timestamps: list[Any] = []
        self.signals_log: list[dict[str, Any]] = []

        # Per-strategy tracking
        self.strategy_trades: dict[str, list[BacktestTrade]] = defaultdict(list)

        # Components
        self.model = TradingModel()
        self.regime_detector = RegimeDetector()
        self.sentiment = SentimentAnalyzer()
        self.strategy_engine = StrategyEngine()

    def _apply_slippage(self, price: float, side: str, is_entry: bool) -> float:
        """Simulate slippage: worse fill for entries, better for exits."""
        if is_entry:
            if side == "long":
                return price * (1 + self.slippage_pct)  # Pay more to enter long
            else:
                return price * (1 - self.slippage_pct)  # Get less to enter short
        else:
            if side == "long":
                return price * (1 - self.slippage_pct)  # Get less when exiting long
            else:
                return price * (1 + self.slippage_pct)  # Pay more when exiting short

    def _calculate_fees(self, price: float, quantity: float) -> float:
        """Calculate trading fees for a given notional value."""
        return price * quantity * self.fee_pct

    def _open_position(
        self,
        bar_idx: int,
        timestamp: Any,
        price: float,
        signal: str,
        confidence: float,
        strat_sig: Any,
        regime_name: str,
    ) -> None:
        """Open a new position with slippage, fees, and risk limits applied."""
        side = "long" if signal == "BUY" else "short"
        risk_amount = self.capital * Config.MAX_POSITION_PCT
        actual_entry = self._apply_slippage(price, side, is_entry=True)
        quantity = risk_amount / actual_entry
        entry_fee = self._calculate_fees(actual_entry, quantity)

        if actual_entry * quantity + entry_fee > self.capital:
            return  # Can't afford

        self.capital -= actual_entry * quantity + entry_fee

        sl_pct = strat_sig.suggested_sl_pct
        tp_pct = strat_sig.suggested_tp_pct
        if side == "long":
            sl = actual_entry * (1 - sl_pct)
            tp = actual_entry * (1 + tp_pct)
            trailing = actual_entry * (1 - self.trailing_stop_pct)
        else:
            sl = actual_entry * (1 + sl_pct)
            tp = actual_entry * (1 - tp_pct)
            trailing = actual_entry * (1 + self.trailing_stop_pct)

        pos = BacktestPosition(
            symbol=self.symbol,
            side=side,
            entry_price=price,
            entry_price_actual=actual_entry,
            quantity=quantity,
            entry_bar=bar_idx,
            entry_time=timestamp,
            stop_loss=sl,
            take_profit=tp,
            trailing_stop=trailing,
            highest_price=price,
            lowest_price=price,
            strategy_name=strat_sig.strategy_name,
            regime=regime_name,
        )
        self.positions.append(pos)

    def _close_position(self, pos: BacktestPosition, bar_idx: int, timestamp: Any, price: float, reason: str) -> None:
        """Close a position, record trade with PnL, fees, and slippage."""
        actual_exit = self._apply_slippage(price, pos.side, is_entry=False)
        exit_fee = self._calculate_fees(actual_exit, pos.quantity)

        if pos.side == "long":
            pnl_gross = (price - pos.entry_price) * pos.quantity
            pnl_net = (actual_exit - pos.entry_price_actual) * pos.quantity - exit_fee
        else:
            pnl_gross = (pos.entry_price - price) * pos.quantity
            pnl_net = (pos.entry_price_actual - actual_exit) * pos.quantity - exit_fee

        entry_fee = self._calculate_fees(pos.entry_price_actual, pos.quantity)
        slippage = (
            abs(pos.entry_price_actual - pos.entry_price) * pos.quantity + abs(actual_exit - price) * pos.quantity
        )

        trade = BacktestTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_price_actual=pos.entry_price_actual,
            exit_price_actual=actual_exit,
            quantity=pos.quantity,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            fees_paid=entry_fee + exit_fee,
            slippage_cost=slippage,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            exit_reason=reason,
            strategy_name=pos.strategy_name,
            regime=pos.regime,
            hold_bars=bar_idx - pos.entry_bar,
        )
        self.trades.append(trade)
        self.strategy_trades[pos.strategy_name].append(trade)

        self.capital += actual_exit * pos.quantity - exit_fee
        self.positions.remove(pos)

    def _check_positions(self, bar_idx: int, timestamp: Any, row: Any) -> None:
        """Update trailing stops and check exit conditions for all positions."""
        price = row["close"]
        high = row["high"]
        low = row["low"]

        for pos in list(self.positions):
            # Update trailing stop
            if pos.side == "long":
                if high > pos.highest_price:
                    pos.highest_price = high
                    pos.trailing_stop = max(pos.trailing_stop, high * (1 - self.trailing_stop_pct))
                # Check exits: SL, TP, trailing, max duration
                if low <= pos.stop_loss:
                    self._close_position(pos, bar_idx, timestamp, pos.stop_loss, "stop_loss")
                elif high >= pos.take_profit:
                    self._close_position(pos, bar_idx, timestamp, pos.take_profit, "take_profit")
                elif low <= pos.trailing_stop and pos.trailing_stop > pos.stop_loss:
                    self._close_position(pos, bar_idx, timestamp, pos.trailing_stop, "trailing_stop")
                elif bar_idx - pos.entry_bar >= self.max_hold_bars:
                    self._close_position(pos, bar_idx, timestamp, price, "max_duration")
            else:  # short
                if low < pos.lowest_price:
                    pos.lowest_price = low
                    pos.trailing_stop = min(pos.trailing_stop, low * (1 + self.trailing_stop_pct))
                if high >= pos.stop_loss:
                    self._close_position(pos, bar_idx, timestamp, pos.stop_loss, "stop_loss")
                elif low <= pos.take_profit:
                    self._close_position(pos, bar_idx, timestamp, pos.take_profit, "take_profit")
                elif high >= pos.trailing_stop and pos.trailing_stop < pos.stop_loss:
                    self._close_position(pos, bar_idx, timestamp, pos.trailing_stop, "trailing_stop")
                elif bar_idx - pos.entry_bar >= self.max_hold_bars:
                    self._close_position(pos, bar_idx, timestamp, price, "max_duration")

    def run(
        self, df: pd.DataFrame, train_split: float = 0.3, retrain_every: int = 50, verbose: bool = True
    ) -> dict[str, Any]:
        """
        Run backtest on historical OHLCV data.

        Args:
            df: OHLCV DataFrame (timestamp index)
            train_split: fraction of data for initial training
            retrain_every: retrain ML model every N bars
            verbose: print progress

        Returns:
            dict of performance metrics
        """
        if verbose:
            print("=" * 60)
            print("  BACKTEST ENGINE v2.0")
            print(f"  Symbol: {self.symbol} | Data: {len(df)} bars | Capital: ${self.initial_capital:,.2f}")
            print(f"  Fees: {self.fee_pct:.2%} | Slippage: {self.slippage_pct:.2%}")
            print(f"  Trailing stop: {self.trailing_stop_pct:.2%} | Max hold: {self.max_hold_bars} bars")
            print("=" * 60)

        # Compute indicators on full dataset
        Indicators.invalidate_cache()
        df_ind = Indicators.add_all(df)

        if len(df_ind) < 100:
            return {"error": "insufficient_data"}

        train_end = int(len(df_ind) * train_split)
        if train_end < 80:
            train_end = 80

        # Initial training
        train_data = df_ind.iloc[:train_end]
        self.model.train(df=None, df_ind=train_data)

        # Walk forward through test data
        test_start = train_end
        total_bars = len(df_ind) - test_start
        last_retrain = test_start

        if verbose:
            print(f"  Training on bars 0-{train_end} | Testing on {test_start}-{len(df_ind)}")
            print(f"  Walk-forward retrain every {retrain_every} bars\n")

        for i in range(test_start, len(df_ind)):
            bar_idx = i - test_start
            row = df_ind.iloc[i]
            timestamp = df_ind.index[i]
            price = row["close"]

            # Equity snapshot (mark-to-market)
            unrealized = sum(
                (price - p.entry_price_actual) * p.quantity
                if p.side == "long"
                else (p.entry_price_actual - price) * p.quantity
                for p in self.positions
            )
            self.equity_curve.append(
                self.capital + unrealized + sum(p.entry_price_actual * p.quantity for p in self.positions)
            )
            self.equity_timestamps.append(timestamp)

            # Check existing positions
            self._check_positions(bar_idx, timestamp, row)

            # Walk-forward retraining
            if i - last_retrain >= retrain_every and i > train_end + 20:
                Indicators.invalidate_cache()
                retrain_data = Indicators.add_all(df.iloc[:i])
                self.model.train(df=None, df_ind=retrain_data)
                last_retrain = i

            # Skip if at max positions
            if len(self.positions) >= Config.MAX_OPEN_POSITIONS:
                continue

            # Analysis (use a window of data up to current bar)
            window = df_ind.iloc[max(0, i - 200) : i + 1]
            if len(window) < 50:
                continue

            regime_state = self.regime_detector.detect(df.iloc[max(0, i - 200) : i + 1], df_ind=window)
            sentiment_state = self.sentiment.analyze(df.iloc[max(0, i - 200) : i + 1], df_ind=window)
            strat_signal = self.strategy_engine.run(window, regime_state.regime, sentiment_state)
            ml_signal, ml_conf = self.model.predict(df_ind=window)

            # Combine
            final_signal, final_conf = self._combine(strat_signal, ml_signal, ml_conf)

            self.signals_log.append(
                {
                    "bar": bar_idx,
                    "timestamp": str(timestamp),
                    "price": price,
                    "signal": final_signal,
                    "confidence": final_conf,
                    "regime": regime_state.regime.value,
                    "strategy": strat_signal.strategy_name,
                }
            )

            # Execute
            if final_signal in ("BUY", "SELL") and final_conf >= self.min_confidence:
                # Check no conflicting/duplicate positions
                has_conflict = any(p.symbol == self.symbol for p in self.positions)
                if not has_conflict:
                    self._open_position(
                        bar_idx,
                        timestamp,
                        price,
                        final_signal,
                        final_conf,
                        strat_signal,
                        regime_state.regime.value,
                    )

            # Progress
            if verbose and bar_idx % 100 == 0 and bar_idx > 0:
                pct = bar_idx / total_bars * 100
                print(
                    f"  Bar {bar_idx}/{total_bars} ({pct:.0f}%) | "
                    f"Trades: {len(self.trades)} | "
                    f"Equity: ${self.equity_curve[-1]:,.2f}"
                )

        # Close remaining positions at last price
        last_price = df_ind["close"].iloc[-1]
        last_ts = df_ind.index[-1]
        for pos in list(self.positions):
            self._close_position(pos, total_bars, last_ts, last_price, "backtest_end")

        # Final equity
        self.equity_curve.append(self.capital)
        self.equity_timestamps.append(df_ind.index[-1])

        results = self._compute_metrics()
        if verbose:
            self.print_report(results)
        return results

    def _combine(self, strat_sig: Any, ml_signal: str, ml_conf: float) -> tuple[str, float]:
        """Combine strategy and ML signals into a final (signal, confidence) pair."""
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

    def _compute_metrics(self) -> dict[str, Any]:
        """Compute all performance metrics from the equity curve and trades."""
        equity = np.array(self.equity_curve)
        if len(equity) < 2:
            return {"error": "no_data"}

        total_return = (equity[-1] / equity[0]) - 1
        n_trades = len(self.trades)

        if n_trades == 0:
            return {
                "total_return_pct": total_return * 100,
                "total_trades": 0,
                "final_equity": equity[-1],
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "avg_hold_bars": 0,
                "total_fees": 0,
                "total_slippage": 0,
                "exit_reasons": {},
            }

        # Returns series
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        # Sharpe (annualized, assuming 1h bars = 8760 per year)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)
        else:
            sharpe = 0

        # Sortino (downside deviation only)
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 1:
            downside_std = np.std(neg_returns)
            sortino = np.mean(returns) / downside_std * np.sqrt(8760) if downside_std > 0 else 0
        else:
            sortino = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(np.min(drawdown))

        # Calmar ratio
        calmar = (total_return * (8760 / len(equity))) / abs(max_dd) if max_dd != 0 else 0

        # Trade stats
        winning = [t for t in self.trades if t.pnl_net > 0]
        losing = [t for t in self.trades if t.pnl_net <= 0]
        win_rate = len(winning) / n_trades if n_trades > 0 else 0

        gross_profit = sum(t.pnl_net for t in winning)
        gross_loss = abs(sum(t.pnl_net for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl_net for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_net for t in losing]) if losing else 0
        avg_hold = np.mean([t.hold_bars for t in self.trades])

        total_fees = sum(t.fees_paid for t in self.trades)
        total_slippage = sum(t.slippage_cost for t in self.trades)

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in self.trades:
            exit_reasons[t.exit_reason] += 1

        return {
            "total_return_pct": round(total_return * 100, 2),
            "final_equity": round(equity[-1], 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "total_trades": n_trades,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_hold_bars": round(avg_hold, 1),
            "total_fees": round(total_fees, 2),
            "total_slippage": round(total_slippage, 2),
            "exit_reasons": dict(exit_reasons),
        }

    def print_report(self, results: dict[str, Any] | None = None) -> None:
        """Print formatted backtest results with per-strategy attribution.

        Outputs a human-readable table of key performance metrics (return,
        drawdown, Sharpe, win rate, fees) followed by a per-strategy
        breakdown sorted by total PnL.

        Args:
            results: Pre-computed metrics dict as returned by ``run()``.
                When ``None``, ``_compute_metrics()`` is called automatically.
        """
        if results is None:
            results = self._compute_metrics()

        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total Return:     {results['total_return_pct']:+.2f}%")
        print(f"  Final Equity:     ${results['final_equity']:,.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:    {results['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio:     {results['calmar_ratio']:.3f}")
        print()
        print(f"  Total Trades:     {results['total_trades']}")
        print(f"  Win Rate:         {results['win_rate']}%")
        print(f"  Profit Factor:    {results['profit_factor']:.2f}")
        print(f"  Avg Win:          ${results['avg_win']:,.2f}")
        print(f"  Avg Loss:         ${results['avg_loss']:,.2f}")
        print(f"  Avg Hold:         {results['avg_hold_bars']:.1f} bars")
        print()
        print(f"  Total Fees:       ${results['total_fees']:,.2f}")
        print(f"  Total Slippage:   ${results['total_slippage']:,.2f}")
        print(f"  Cost Drag:        ${results['total_fees'] + results['total_slippage']:,.2f}")

        if results.get("exit_reasons"):
            print("\n  Exit Reasons:")
            for reason, count in sorted(results["exit_reasons"].items(), key=lambda x: -x[1]):
                print(f"    {reason:20s} {count}")

        # Per-strategy attribution
        if self.strategy_trades:
            print("\n  Strategy Performance:")
            print(f"  {'Strategy':<30s} {'Trades':>6s} {'Win%':>6s} {'PnL':>10s} {'Avg':>8s}")
            print(f"  {'-' * 62}")
            for strat_name, trades in sorted(
                self.strategy_trades.items(), key=lambda x: sum(t.pnl_net for t in x[1]), reverse=True
            ):
                n = len(trades)
                wins = sum(1 for t in trades if t.pnl_net > 0)
                total_pnl = sum(t.pnl_net for t in trades)
                avg_pnl = total_pnl / n if n > 0 else 0
                wr = wins / n * 100 if n > 0 else 0
                print(f"  {strat_name:<30s} {n:>6d} {wr:>5.1f}% ${total_pnl:>9,.2f} ${avg_pnl:>7,.2f}")

        print("=" * 60)

    def plot_equity_curve(self, filepath: str = "equity_curve.html") -> None:
        """Generate an interactive HTML equity curve chart using Chart.js.

        Creates a self-contained HTML file with an equity curve line chart
        and a drawdown chart beneath it, along with a summary metrics grid.
        Uses Chart.js (CDN) so no additional Python dependencies are required.

        Args:
            filepath: Output path for the HTML file. Defaults to
                ``"equity_curve.html"`` in the current working directory.
        """
        if not self.equity_curve:
            print("No equity data to plot")
            return

        equity = self.equity_curve
        timestamps = [str(t) for t in self.equity_timestamps]
        peak = np.maximum.accumulate(equity)
        drawdown = [(e - p) / p * 100 for e, p in zip(equity, peak)]

        # Trade markers
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        for t in self.trades:
            entry_ts = str(t.entry_time)
            exit_ts = str(t.exit_time)
            if t.side == "long":
                buy_x.append(entry_ts)
                sell_x.append(exit_ts)
            else:
                sell_x.append(entry_ts)
                buy_x.append(exit_ts)
            # Find closest equity point
            buy_y.append(t.entry_price_actual * t.quantity)
            sell_y.append(t.exit_price_actual * t.quantity)

        html = f"""<!DOCTYPE html>
<html><head><title>Backtest Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
.metric {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }}
.metric .label {{ font-size: 12px; color: #8b949e; }}
.metric .value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
canvas {{ background: #161b22; border-radius: 8px; padding: 10px; margin: 15px 0; }}
</style></head><body>
<div class="container">
<h1>Backtest Results</h1>
<div class="metrics">
<div class="metric"><div class="label">Total Return</div>
<div class="value {"positive" if equity[-1] > equity[0] else "negative"}">{((equity[-1] / equity[0]) - 1) * 100:+.2f}%</div></div>
<div class="metric"><div class="label">Sharpe Ratio</div>
<div class="value">{self._compute_metrics()["sharpe_ratio"]:.3f}</div></div>
<div class="metric"><div class="label">Max Drawdown</div>
<div class="value negative">{min(drawdown):.2f}%</div></div>
<div class="metric"><div class="label">Win Rate</div>
<div class="value">{self._compute_metrics()["win_rate"]}%</div></div>
</div>
<canvas id="equityChart" height="300"></canvas>
<canvas id="drawdownChart" height="150"></canvas>
</div>
<script>
const labels = {timestamps[:500] if len(timestamps) > 500 else timestamps};
const equity = {[round(e, 2) for e in equity[:500]] if len(equity) > 500 else [round(e, 2) for e in equity]};
const dd = {[round(d, 2) for d in drawdown[:500]] if len(drawdown) > 500 else [round(d, 2) for d in drawdown]};

new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{ labels, datasets: [{{
    label: 'Equity ($)', data: equity,
    borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)',
    fill: true, pointRadius: 0, borderWidth: 2,
  }}]}},
  options: {{ responsive: true, scales: {{
    x: {{ display: true, ticks: {{ maxTicksLimit: 10, color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
  }} }}
}});

new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{ labels, datasets: [{{
    label: 'Drawdown (%)', data: dd,
    borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.15)',
    fill: true, pointRadius: 0, borderWidth: 1.5,
  }}]}},
  options: {{ responsive: true, scales: {{
    x: {{ display: false }},
    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
  }} }}
}});
</script></body></html>"""

        with open(filepath, "w") as f:
            f.write(html)
        print(f"\n  Equity curve saved to {filepath}")
