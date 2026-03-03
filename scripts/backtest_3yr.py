"""3-year backtest for BTC/USDT, ETH/USDT, SOL/USDT.

Fetches 3 years of hourly OHLCV data via CCXT pagination,
then runs the full Backtester (walk-forward ML, all 11 strategies).

Retrain schedule: every 2160 bars (~3 months at 1h) — quarterly walk-forward,
which gives ~12 retrains per pair.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester import Backtester
from data_fetcher import DataFetcher

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "1h"
YEARS = 3
INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.001        # 0.10% maker/taker (Binance default)
SLIPPAGE_PCT = 0.0005  # 0.05% per side
BATCH_SIZE = 1000      # CCXT bars per request
RATE_LIMIT_S = 0.5     # Seconds between paginated requests
RETRAIN_EVERY = 2160   # Quarterly (2160 h = 90 days) — ~12 retrains/pair
MIN_CONFIDENCE = 0.68       # Cycle 3: raised from 0.65 to 0.68 (matches backtest_5yr.py).
                            # Effective floor when ML=HOLD: 0.68/0.75=0.907 strategy conf.
                            # 0.72 was too tight (7 BTC trades/3yr); 0.68 balances quality vs count.
# Trailing stop is now per-pair from Config.get_trailing_stop_pct()
# (BTC/ETH=2.5%, SOL=4.0%) — no global constant needed


def fetch_ohlcv(fetcher: DataFetcher, symbol: str) -> pd.DataFrame:
    """Fetch 3 years of hourly OHLCV data using paginated CCXT calls."""
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
        print(f"  … {len(all_bars):,} bars  (up to {fetched_dt.strftime('%Y-%m-%d')})", end="\r")

        if len(raw) < BATCH_SIZE:
            break  # Last (partial) page

        # Advance cursor by 1 ms past the last bar to avoid duplicates
        current_since = last_ts + 1
        time.sleep(RATE_LIMIT_S)

    print(f"  Fetched {len(all_bars):,} bars total for {symbol}        ")

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


def run_backtest(symbol: str, df: pd.DataFrame) -> tuple[dict, Backtester]:
    """Run the full backtester on a pre-fetched DataFrame."""
    bt = Backtester(
        initial_capital=INITIAL_CAPITAL,
        fee_pct=FEE_PCT,
        slippage_pct=SLIPPAGE_PCT,
        symbol=symbol,
        # trailing_stop_pct omitted → resolved per-pair from Config
        min_confidence=MIN_CONFIDENCE,
    )
    results = bt.run(df, train_split=0.2, retrain_every=RETRAIN_EVERY, verbose=True)
    return results, bt


def print_summary(symbol: str, results: dict, bt: Backtester) -> None:
    """Print a formatted results card for one pair."""
    SEP = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {symbol}  ({YEARS}-Year Backtest, {TIMEFRAME} bars)")
    print(f"{'═' * 60}")

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    r = results
    print(f"  Capital:         ${INITIAL_CAPITAL:>10,.2f}  →  ${r.get('final_equity', 0):>12,.2f}")
    print(f"  Total Return:    {r.get('total_return_pct', 0):>+9.2f}%")
    print(f"  Max Drawdown:    {r.get('max_drawdown_pct', 0):>9.2f}%")
    print(f"  Sharpe Ratio:    {r.get('sharpe_ratio', 0):>9.3f}")
    print(f"  Sortino Ratio:   {r.get('sortino_ratio', 0):>9.3f}")
    print(f"  Calmar Ratio:    {r.get('calmar_ratio', 0):>9.3f}")
    print(SEP)
    print(f"  Total Trades:    {r.get('total_trades', 0):>9d}")
    print(f"  Win Rate:        {r.get('win_rate', 0):>9.1f}%")
    print(f"  Profit Factor:   {r.get('profit_factor', 0):>9.3f}")
    print(f"  Avg Win:         ${r.get('avg_win', 0):>9.2f}")
    print(f"  Avg Loss:        ${r.get('avg_loss', 0):>9.2f}")
    print(f"  Avg Hold (bars): {r.get('avg_hold_bars', 0):>9.1f}")
    print(SEP)
    print(f"  Total Fees:      ${r.get('total_fees', 0):>9.2f}")
    print(f"  Total Slippage:  ${r.get('total_slippage', 0):>9.2f}")

    # Exit reason breakdown
    exits = r.get("exit_reasons", {})
    if exits:
        print("\n  Exit Reasons:")
        for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
            print(f"    {reason:<22} {count:>5}")

    # Per-strategy breakdown
    strategy_stats = []
    for sname, trades in bt.strategy_trades.items():
        if not trades:
            continue
        wins = [t for t in trades if t.pnl_net > 0]
        total_pnl = sum(t.pnl_net for t in trades)
        strategy_stats.append({
            "name": sname,
            "trades": len(trades),
            "win_rate": 100 * len(wins) / len(trades) if trades else 0,
            "pnl": total_pnl,
        })

    if strategy_stats:
        strategy_stats.sort(key=lambda x: -x["pnl"])
        print("\n  Strategy Attribution (sorted by PnL):")
        print(f"    {'Strategy':<22} {'Trades':>6}  {'Win%':>6}  {'Net PnL':>10}")
        print(f"    {'─'*22} {'─'*6}  {'─'*6}  {'─'*10}")
        for s in strategy_stats:
            print(f"    {s['name']:<22} {s['trades']:>6}  {s['win_rate']:>5.1f}%  ${s['pnl']:>9.2f}")


def main() -> None:
    start_wall = time.time()
    fetcher = DataFetcher()

    print("\n" + "═" * 60)
    print(f"  3-YEAR BACKTEST  |  {', '.join(PAIRS)}")
    print(f"  Period: {(datetime.now(UTC)-timedelta(days=365*YEARS)).strftime('%Y-%m-%d')}  →  {datetime.now(UTC).strftime('%Y-%m-%d')}")
    print(f"  Timeframe: {TIMEFRAME}  |  Capital: ${INITIAL_CAPITAL:,.0f}  |  Fees: {FEE_PCT:.2%}")
    print(f"  Walk-forward retrain: every {RETRAIN_EVERY} bars (~{RETRAIN_EVERY//720} months)")
    print("═" * 60)

    all_results: dict[str, tuple[dict, Backtester]] = {}

    for pair in PAIRS:
        df = fetch_ohlcv(fetcher, pair)
        if df.empty:
            print(f"  [!] No data for {pair}, skipping.")
            continue

        print(f"\n  Running backtest for {pair} ({len(df):,} bars) …")
        results, bt = run_backtest(pair, df)
        all_results[pair] = (results, bt)
        print_summary(pair, results, bt)

    # ── Comparison table ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n\n{'═' * 60}")
        print("  COMPARISON SUMMARY")
        print("═" * 60)
        hdr = f"  {'Pair':<12} {'Return%':>8}  {'Drawdown%':>10}  {'Sharpe':>7}  {'Trades':>7}  {'Win%':>6}  {'PF':>5}"
        print(hdr)
        print(f"  {'─'*12} {'─'*8}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}")
        for pair, (r, _) in all_results.items():
            if "error" in r:
                continue
            print(
                f"  {pair:<12} {r.get('total_return_pct',0):>+7.2f}%  "
                f"{r.get('max_drawdown_pct',0):>9.2f}%  "
                f"{r.get('sharpe_ratio',0):>7.3f}  "
                f"{r.get('total_trades',0):>7}  "
                f"{r.get('win_rate',0):>5.1f}%  "
                f"{r.get('profit_factor',0):>5.3f}"
            )

    elapsed = time.time() - start_wall
    print(f"\n  Total runtime: {elapsed/60:.1f} min")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
