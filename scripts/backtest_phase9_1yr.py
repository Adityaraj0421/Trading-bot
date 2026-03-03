"""Phase 9 backtest — 1 year BTC/USDT, ETH/USDT, SOL/USDT.

Uses the Phase 9 Context + Trigger pipeline (decision.evaluate() path):
  - ContextEngine: SwingAnalyzer on rolling 4h window; funding/whale/OI stubbed
    to neutral (no historical feed available from Binance CCXT).
  - MomentumTrigger: 1h RSI zero-cross + MACD zero-cross + volume confirmation.
  - evaluate(): context gate → directional consensus (≥2 signals) → score ≥ 0.50

Trade simulation (spot only — one position per pair at a time):
  - Enter at next bar's open after a trade Decision.
  - Exit: trailing stop | take-profit | max hold bars.
  - Fees: 0.10% per side | Slippage: 0.05% per side (round-trip 0.30%).

Consensus note:
  MomentumTrigger fires at most 1 signal per 1h bar. To satisfy evaluate()'s
  2-signal consensus requirement, this script carries each bar's signals into
  the next bar (simulating the 75-min TTL at 1h resolution). Two consecutive
  same-direction signals = consensus reached.

Run:
  python scripts/backtest_phase9_1yr.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from context_engine import ContextEngine
from data_fetcher import DataFetcher
from data_snapshot import DataSnapshot
from decision import TriggerSignal, evaluate

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
YEARS = 1
INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.001         # 0.10% per side
SLIPPAGE_PCT = 0.0005   # 0.05% per side
POSITION_PCT = 0.10     # 10% of capital per trade
TAKE_PROFIT_PCT = 0.06  # 6% take-profit
MAX_HOLD_BARS = 72      # 3 days at 1h
CONTEXT_REFRESH_BARS = 4  # Rebuild context every 4 bars (simulates ~4h cadence)
WARMUP_BARS = 210         # 200 (EMA-200) + 10 buffer

# Per-pair trailing stop — SOL needs wider stop (2-3× BTC hourly ATR)
TRAILING_STOP_PCT: dict[str, float] = {
    "BTC/USDT": 0.025,
    "ETH/USDT": 0.025,
    "SOL/USDT": 0.040,
}

BATCH_SIZE = 1000
RATE_LIMIT_S = 0.5


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    direction: str       # "long" | "short"
    entry_price: float
    trailing_stop: float
    take_profit: float
    entry_bar: int
    size: float          # USD notional


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv(fetcher: DataFetcher, symbol: str) -> pd.DataFrame:
    """Fetch 1 year of hourly OHLCV data using paginated CCXT calls."""
    since_dt = datetime.now(UTC) - timedelta(days=365 * YEARS)
    since_ms = int(since_dt.timestamp() * 1000)
    print(f"\n  Fetching {symbol} from {since_dt.strftime('%Y-%m-%d')} …")

    all_bars: list[list] = []
    current_since = since_ms

    while True:
        try:
            raw = fetcher.exchange.fetch_ohlcv(
                symbol, TIMEFRAME, since=current_since, limit=BATCH_SIZE
            )
        except Exception as exc:
            print(f"  [!] Fetch error: {exc} — stopping pagination")
            break

        if not raw:
            break

        all_bars.extend(raw)
        last_ts = raw[-1][0]
        fetched_dt = datetime.fromtimestamp(last_ts / 1000, tz=UTC)
        print(
            f"  … {len(all_bars):,} bars  (up to {fetched_dt.strftime('%Y-%m-%d')})",
            end="\r",
        )

        if len(raw) < BATCH_SIZE:
            break

        current_since = last_ts + 1
        time.sleep(RATE_LIMIT_S)

    print(f"  Fetched {len(all_bars):,} bars total for {symbol}        ")

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df


def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h bars for SwingAnalyzer."""
    return (
        df_1h.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


# ── Phase 9 backtest loop ─────────────────────────────────────────────────────

def run_backtest(symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> dict:
    """Run Phase 9 pipeline bar-by-bar and simulate spot P&L.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
        df_1h: 1h OHLCV DataFrame indexed by UTC timestamp.
        df_4h: 4h OHLCV DataFrame indexed by UTC timestamp.

    Returns:
        Dict with trade summary statistics and individual trade records.
    """
    # Lazy import avoids heavy module import at script top
    from triggers.momentum import MomentumTrigger  # noqa: PLC0415

    trail_pct = TRAILING_STOP_PCT.get(symbol, 0.025)
    capital = INITIAL_CAPITAL

    ctx_engine = ContextEngine()
    momentum = MomentumTrigger(symbol=symbol)

    position: Position | None = None
    trades: list[dict] = []
    context = None
    prev_signals: list[TriggerSignal] = []  # Carry to simulate 75-min TTL

    total_bars = len(df_1h) - WARMUP_BARS
    start_time = time.monotonic()

    for i in range(WARMUP_BARS, len(df_1h)):
        # Progress indicator every 500 bars
        if (i - WARMUP_BARS) % 500 == 0:
            pct = (i - WARMUP_BARS) / total_bars * 100
            elapsed = time.monotonic() - start_time
            print(f"  [{symbol}] {pct:5.1f}%  bar {i - WARMUP_BARS:,}/{total_bars:,}"
                  f"  capital=${capital:,.0f}  trades={len(trades)}", end="\r")

        row = df_1h.iloc[i]
        high_p = float(row["high"])
        low_p = float(row["low"])
        close_p = float(row["close"])

        # ── 1. Check open position for exit ───────────────────────────────────
        if position is not None:
            bars_held = i - position.entry_bar
            exit_price: float | None = None
            exit_reason: str | None = None

            if position.direction == "long":
                # Ratchet trailing stop upward
                new_stop = high_p * (1 - trail_pct)
                if new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop

                if low_p <= position.trailing_stop:
                    exit_price = position.trailing_stop
                    exit_reason = "trailing_stop"
                elif high_p >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
                elif bars_held >= MAX_HOLD_BARS:
                    exit_price = close_p
                    exit_reason = "max_hold"

            else:  # short
                # Ratchet trailing stop downward
                new_stop = low_p * (1 + trail_pct)
                if new_stop < position.trailing_stop:
                    position.trailing_stop = new_stop

                if high_p >= position.trailing_stop:
                    exit_price = position.trailing_stop
                    exit_reason = "trailing_stop"
                elif low_p <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
                elif bars_held >= MAX_HOLD_BARS:
                    exit_price = close_p
                    exit_reason = "max_hold"

            if exit_price is not None:
                gross = (
                    (exit_price - position.entry_price) / position.entry_price
                    if position.direction == "long"
                    else (position.entry_price - exit_price) / position.entry_price
                )
                net_pnl = gross - 2 * FEE_PCT - 2 * SLIPPAGE_PCT
                pnl_usd = position.size * net_pnl
                capital += pnl_usd
                trades.append({
                    "direction": position.direction,
                    "entry_bar": position.entry_bar,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": net_pnl * 100,
                    "pnl_usd": pnl_usd,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                })
                position = None

        # ── 2. Refresh context (every CONTEXT_REFRESH_BARS bars) ──────────────
        if context is None or i % CONTEXT_REFRESH_BARS == 0:
            bar_ts = df_1h.index[i]
            df_4h_slice = df_4h[df_4h.index <= bar_ts].copy()
            df_1h_slice = df_1h.iloc[:i + 1].copy()

            snap = DataSnapshot(
                df_1h=df_1h_slice,
                df_4h=df_4h_slice if len(df_4h_slice) >= 50 else None,
                df_15m=None,
                symbol=symbol,
            )

            # price_change_pct over last 4 bars (≈ 4h)
            if i >= 4:
                prev_close = float(df_1h["close"].iloc[i - 4])
                price_change_pct: float | None = (
                    (close_p - prev_close) / prev_close * 100 if prev_close != 0 else None
                )
            else:
                price_change_pct = None

            context = ctx_engine.build(
                snapshot=snap,
                funding_rate=None,      # Not available from CCXT historically
                net_whale_flow=None,    # Not available from CCXT historically
                oi_change_pct=None,     # Not available from CCXT historically
                price_change_pct=price_change_pct,
            )

        # ── 3. Generate 1h momentum triggers ──────────────────────────────────
        # Rolling 100-bar window is enough for all indicators (max need: 26 for MACD)
        window_start = max(0, i - 100)
        df_trigger = df_1h.iloc[window_start:i + 1].copy()
        new_signals = momentum.evaluate(df_trigger)

        # Combine prev bar's signals (TTL 75min > 1 bar = 60min) + this bar's
        triggers = prev_signals + new_signals
        prev_signals = new_signals  # Carry to next bar

        # ── 4. Evaluate and potentially open position ──────────────────────────
        if context is not None and position is None and i + 1 < len(df_1h):
            decision = evaluate(context, triggers)
            if decision.action == "trade":
                next_open = float(df_1h.iloc[i + 1]["open"])
                if decision.direction == "long":
                    actual_entry = next_open * (1 + SLIPPAGE_PCT)
                    stop = actual_entry * (1 - trail_pct)
                    tp = actual_entry * (1 + TAKE_PROFIT_PCT)
                else:
                    actual_entry = next_open * (1 - SLIPPAGE_PCT)
                    stop = actual_entry * (1 + trail_pct)
                    tp = actual_entry * (1 - TAKE_PROFIT_PCT)

                position = Position(
                    direction=decision.direction,
                    entry_price=actual_entry,
                    trailing_stop=stop,
                    take_profit=tp,
                    entry_bar=i + 1,
                    size=capital * POSITION_PCT,
                )

    # ── Close any remaining open position at last bar's close ─────────────────
    if position is not None:
        last_close = float(df_1h.iloc[-1]["close"])
        gross = (
            (last_close - position.entry_price) / position.entry_price
            if position.direction == "long"
            else (position.entry_price - last_close) / position.entry_price
        )
        net_pnl = gross - 2 * FEE_PCT - 2 * SLIPPAGE_PCT
        pnl_usd = position.size * net_pnl
        capital += pnl_usd
        trades.append({
            "direction": position.direction,
            "entry_bar": position.entry_bar,
            "entry_price": position.entry_price,
            "exit_price": last_close,
            "pnl_pct": net_pnl * 100,
            "pnl_usd": pnl_usd,
            "bars_held": len(df_1h) - 1 - position.entry_bar,
            "exit_reason": "end_of_data",
        })

    elapsed = time.monotonic() - start_time
    print(f"  [{symbol}] 100.0%  done in {elapsed:.1f}s{' ' * 30}")

    # ── Summary statistics ─────────────────────────────────────────────────────
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    total_fees = n_trades * 2 * FEE_PCT * (INITIAL_CAPITAL * POSITION_PCT)
    exit_counts = Counter(t["exit_reason"] for t in trades)
    long_trades = [t for t in trades if t["direction"] == "long"]
    short_trades = [t for t in trades if t["direction"] == "short"]

    # Max drawdown (simple equity curve)
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    for t in trades:
        equity += t["pnl_usd"]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "symbol": symbol,
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_pnl_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "final_capital": capital,
        "max_drawdown_pct": max_dd,
        "total_fees_usd": total_fees,
        "exit_reasons": dict(exit_counts),
        "trades": trades,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_result(r: dict) -> None:
    """Print a formatted result card for one pair."""
    SEP = "─" * 62
    total_pnl = r["total_pnl_pct"]
    print(f"\n{'═' * 62}")
    print(f"  RESULTS: {r['symbol']}  (Phase 9 · {YEARS}-Year Backtest, {TIMEFRAME} bars)")
    print(f"{'═' * 62}")
    print(f"  Capital:         ${INITIAL_CAPITAL:>10,.2f}  →  ${r['final_capital']:>12,.2f}")
    print(f"  Total Return:    {total_pnl:>+9.2f}%")
    print(f"  Max Drawdown:    {r['max_drawdown_pct']:>9.2f}%")
    print(SEP)
    print(f"  Total Trades:    {r['n_trades']:>9d}  "
          f"(long={r['n_long']}, short={r['n_short']})")
    print(f"  Win Rate:        {r['win_rate']:>9.1f}%")
    print(f"  Avg Win:         {r['avg_win_pct']:>+9.2f}%")
    print(f"  Avg Loss:        {r['avg_loss_pct']:>+9.2f}%")
    print(SEP)
    print(f"  Est. Fees:       ${r['total_fees_usd']:>9.2f}")
    print("\n  Exit Reasons:")
    for reason, count in sorted(r["exit_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason:<24} {count:>4}")


def print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table for all pairs."""
    print(f"\n\n{'═' * 62}")
    print(f"  PHASE 9 SUMMARY — {YEARS}-Year Backtest")
    print(f"{'═' * 62}")
    print(f"  {'Pair':<12} {'Trades':>6}  {'Win%':>6}  {'PnL%':>7}  {'MaxDD%':>7}")
    print(f"  {'─'*12} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
    for r in results:
        print(
            f"  {r['symbol']:<12} {r['n_trades']:>6}  "
            f"{r['win_rate']:>5.1f}%  "
            f"{r['total_pnl_pct']:>+7.2f}%  "
            f"{r['max_drawdown_pct']:>6.2f}%"
        )
    avg_pnl = sum(r["total_pnl_pct"] for r in results) / len(results) if results else 0
    print(f"  {'─'*12} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
    print(f"  {'Average':<12} {'':>6}  {'':>6}  {avg_pnl:>+7.2f}%")
    print(f"{'═' * 62}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Fetch data, run Phase 9 backtest for all pairs, print results."""
    print("=" * 62)
    print("  Phase 9 Backtest — 1yr (BTC / ETH / SOL)")
    print("  Pipeline: ContextEngine + MomentumTrigger + evaluate()")
    print(f"  Capital:  ${INITIAL_CAPITAL:,.0f}   Fee: {FEE_PCT*100:.2f}%/side   "
          f"Slip: {SLIPPAGE_PCT*100:.3f}%/side")
    print(f"  Position: {POSITION_PCT*100:.0f}% capital   TP: {TAKE_PROFIT_PCT*100:.0f}%   "
          f"MaxHold: {MAX_HOLD_BARS}bars")
    print("=" * 62)

    fetcher = DataFetcher()
    all_results: list[dict] = []

    for pair in PAIRS:
        t0 = time.monotonic()
        df_1h = fetch_ohlcv(fetcher, pair)
        if df_1h.empty:
            print(f"  [!] No data for {pair} — skipping")
            continue

        df_4h = resample_4h(df_1h)
        print(f"  Resampled to {len(df_4h):,} 4h bars")
        print(f"  Running Phase 9 bar-by-bar ({len(df_1h):,} 1h bars, warmup={WARMUP_BARS}) …")

        result = run_backtest(pair, df_1h, df_4h)
        all_results.append(result)
        print_result(result)
        print(f"\n  ⏱  {pair} completed in {time.monotonic() - t0:.1f}s")

    if all_results:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
