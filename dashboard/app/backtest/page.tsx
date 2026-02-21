// ============================================================
// backtest/page.tsx - Backtesting dashboard (v2.1)
// ============================================================

"use client";

import { useState } from "react";
import useSWR from "swr";
import ErrorBanner from "@/components/ErrorBanner";
import { CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

// --- Tiny sparkline component (no extra deps) ---
function Sparkline({ data, color }: { data: number[]; color: string }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 200;
  const h = 40;
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-10">
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={points} />
    </svg>
  );
}

// --- Metric badge ---
function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-gray-900 rounded px-3 py-2">
      <div className="text-[11px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`text-sm font-mono font-semibold ${color || "text-gray-200"}`}>{value}</div>
    </div>
  );
}

// --- Single result card ---
function ResultCard({ r }: { r: any }) {
  const m = r.metrics || {};
  const hasError = m.error || r.error;
  const returnPct = m.total_return_pct ?? 0;
  const isPositive = returnPct >= 0;

  const title =
    r.type === "scenario"
      ? `${(r.scenario || "").replace(/_/g, " ").toUpperCase()}`
      : r.type === "multi_pair"
      ? `${r.pair}`
      : r.type === "multi_timeframe"
      ? `${r.pair} (${r.timeframe})`
      : r.scenario || r.pair || "Backtest";

  if (hasError) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 opacity-60">
        <div className="flex justify-between">
          <h3 className="font-semibold">{title}</h3>
          <span className="text-red-400 text-sm">Error: {m.error || r.error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="font-semibold text-base">{title}</h3>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">{r.periods} bars</span>
          <span
            className={`text-lg font-mono font-bold ${isPositive ? "text-green-400" : "text-red-400"}`}
          >
            {returnPct >= 0 ? "+" : ""}
            {returnPct.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
        <Metric label="Final Equity" value={`$${(m.final_equity ?? 0).toLocaleString()}`} />
        <Metric label="Sharpe" value={(m.sharpe_ratio ?? 0).toFixed(3)} color={m.sharpe_ratio > 1 ? "text-green-400" : m.sharpe_ratio > 0 ? "text-yellow-400" : "text-red-400"} />
        <Metric label="Sortino" value={(m.sortino_ratio ?? 0).toFixed(3)} />
        <Metric label="Max DD" value={`${(m.max_drawdown_pct ?? 0).toFixed(2)}%`} color="text-red-400" />
        <Metric label="Win Rate" value={`${m.win_rate ?? 0}%`} color={(m.win_rate ?? 0) >= 50 ? "text-green-400" : "text-red-400"} />
        <Metric label="Profit Factor" value={(m.profit_factor ?? 0).toFixed(2)} color={(m.profit_factor ?? 0) >= 1.5 ? "text-green-400" : "text-yellow-400"} />
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
        <Metric label="Trades" value={String(m.total_trades ?? 0)} />
        <Metric label="Avg Win" value={`$${(m.avg_win ?? 0).toFixed(2)}`} color="text-green-400" />
        <Metric label="Avg Loss" value={`$${(m.avg_loss ?? 0).toFixed(2)}`} color="text-red-400" />
        <Metric label="Avg Hold" value={`${(m.avg_hold_bars ?? 0).toFixed(1)} bars`} />
        <Metric label="Fees" value={`$${(m.total_fees ?? 0).toFixed(2)}`} />
        <Metric label="Slippage" value={`$${(m.total_slippage ?? 0).toFixed(2)}`} />
      </div>

      {r.equity_curve && r.equity_curve.length > 2 && (
        <div className="pt-1">
          <div className="text-[10px] text-gray-500 mb-1">EQUITY CURVE</div>
          <Sparkline data={r.equity_curve} color={isPositive ? "#4ade80" : "#f87171"} />
        </div>
      )}

      {m.exit_reasons && Object.keys(m.exit_reasons).length > 0 && (
        <div className="flex gap-3 flex-wrap text-xs">
          {Object.entries(m.exit_reasons)
            .sort(([, a]: any, [, b]: any) => b - a)
            .map(([reason, count]: any) => (
              <span key={reason} className="bg-gray-900 rounded px-2 py-1">
                {reason.replace(/_/g, " ")}: <span className="text-gray-300 font-mono">{count}</span>
              </span>
            ))}
        </div>
      )}

      {r.strategy_stats && r.strategy_stats.length > 0 && (
        <div>
          <div className="text-[10px] text-gray-500 mb-1 mt-1">STRATEGY PERFORMANCE</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-700">
                  <th className="text-left py-1 pr-4">Strategy</th>
                  <th className="text-right py-1 px-2">Trades</th>
                  <th className="text-right py-1 px-2">Win%</th>
                  <th className="text-right py-1 pl-2">PnL</th>
                </tr>
              </thead>
              <tbody>
                {r.strategy_stats.map((s: any) => (
                  <tr key={s.strategy} className="border-b border-gray-700/50">
                    <td className="py-1 pr-4 font-medium">{s.strategy}</td>
                    <td className="text-right py-1 px-2 font-mono">{s.trades}</td>
                    <td className={`text-right py-1 px-2 font-mono ${s.win_rate >= 50 ? "text-green-400" : "text-red-400"}`}>
                      {s.win_rate}%
                    </td>
                    <td className={`text-right py-1 pl-2 font-mono ${s.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                      ${s.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="text-[10px] text-gray-600 text-right">
        {r.run_at ? new Date(r.run_at).toLocaleString() : ""}
      </div>
    </div>
  );
}

// --- Main page ---
export default function BacktestPage() {
  const { data: scenarios, error: scenariosErr } = useSWR("/backtest/scenarios", () =>
    api.getBacktestScenarios()
  );
  const { data: results, error: resultsErr, isLoading, mutate } = useSWR(
    "/backtest/results",
    () => api.getBacktestResults(),
    { refreshInterval: 30000 }
  );

  const [running, setRunning] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [selectedMode, setSelectedMode] = useState("scenario");

  const error = scenariosErr || resultsErr;

  const runBacktest = async () => {
    setRunning(true);
    try {
      if (selectedMode === "all_pairs") {
        await api.runBacktest({ mode: "all_pairs" });
      } else if (selectedMode === "all_timeframes") {
        await api.runBacktest({ mode: "all_timeframes" });
      } else {
        await api.runBacktest({
          scenario: selectedScenario || undefined,
          periods: 500,
        });
      }
      setTimeout(() => mutate(), 3000);
      setTimeout(() => mutate(), 8000);
      setTimeout(() => mutate(), 15000);
    } finally {
      setTimeout(() => setRunning(false), 2000);
    }
  };

  const clearResults = async () => {
    await api.clearBacktestResults();
    mutate();
  };

  const allResults = results?.results || [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Backtesting</h1>

      <ErrorBanner error={error} onRetry={() => mutate()} />

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Mode</label>
            <select
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
              value={selectedMode}
              onChange={(e) => setSelectedMode(e.target.value)}
            >
              <option value="scenario">Scenario</option>
              <option value="all_pairs">All Trading Pairs</option>
              <option value="all_timeframes">Multi-Timeframe</option>
            </select>
          </div>

          {selectedMode === "scenario" && (
            <div>
              <label className="block text-sm text-gray-400 mb-1">Scenario</label>
              <select
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                value={selectedScenario}
                onChange={(e) => setSelectedScenario(e.target.value)}
              >
                <option value="">All scenarios</option>
                {(scenarios?.scenarios || []).map((s: string) => (
                  <option key={s} value={s}>
                    {s.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={runBacktest}
            disabled={running}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded text-sm font-medium transition-colors"
          >
            {running ? "Running..." : "Run Backtest"}
          </button>

          {allResults.length > 0 && (
            <button
              onClick={clearResults}
              className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded text-sm text-gray-400 transition-colors"
            >
              Clear Results
            </button>
          )}
        </div>
      </div>

      {/* Results */}
      {isLoading && !results ? (
        <CardListSkeleton count={3} />
      ) : allResults.length > 0 ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <span>{allResults.length} backtest result{allResults.length !== 1 ? "s" : ""}</span>
            <span>
              Best:{" "}
              <span className="text-green-400 font-mono">
                {Math.max(
                  ...allResults
                    .filter((r: any) => r.metrics)
                    .map((r: any) => r.metrics.total_return_pct ?? -Infinity)
                ).toFixed(2)}%
              </span>
              {" | "}Worst:{" "}
              <span className="text-red-400 font-mono">
                {Math.min(
                  ...allResults
                    .filter((r: any) => r.metrics)
                    .map((r: any) => r.metrics.total_return_pct ?? Infinity)
                ).toFixed(2)}%
              </span>
            </span>
          </div>

          {allResults.map((r: any, i: number) => (
            <ResultCard key={`${r.scenario || r.pair}-${i}`} r={r} />
          ))}
        </div>
      ) : (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-12 text-center text-gray-500">
          <p className="text-lg mb-2">No backtest results yet</p>
          <p className="text-sm">
            Choose a mode above and click Run Backtest to test
            your strategies against historical or synthetic data.
          </p>
        </div>
      )}
    </div>
  );
}
