// ============================================================
// page.tsx - Home / Overview page (the main dashboard)
// ============================================================
// This is the first page users see. It shows:
//   - Key metrics (capital, PnL, win rate, regime)
//   - Equity curve chart
//   - Recent trades table
//
// Data refreshes every 5 seconds using the SWR library.
// "use client" is required because we use React hooks (useSWR).

"use client";

import useSWR from "swr";
import MetricCard from "@/components/MetricCard";
import EquityChart from "@/components/EquityChart";
import StatusBadge from "@/components/StatusBadge";
import TradeTable from "@/components/TradeTable";
import ErrorBanner from "@/components/ErrorBanner";
import PageSkeleton from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function Home() {
  // SWR handles initial load + fallback polling (30s).
  // WebSocket pushes inject data into these same cache keys for real-time updates.
  const { data: status, error: statusErr, isLoading: statusLoading } = useSWR(
    "/status", () => api.getStatus(), { refreshInterval: 30000 }
  );
  const { data: equity, error: equityErr } = useSWR(
    "/equity", () => api.getEquity(100), { refreshInterval: 30000 }
  );
  const { data: trades, error: tradesErr } = useSWR(
    "/trades", () => api.getTrades(10), { refreshInterval: 30000 }
  );
  const { data: modules } = useSWR(
    "/system/modules", () => api.getSystemModules(), { refreshInterval: 60000 }
  );

  // "s" is a shortcut for the status data (or empty object if loading)
  const s = (status as Record<string, any>) || {};
  const isWaiting = !s.cycle;
  const hasError = statusErr || equityErr || tradesErr;

  return (
    <div className="space-y-6">
      {/* Page header with status badge */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {s.autonomous && <StatusBadge state={s.autonomous.state} />}
      </div>

      <ErrorBanner error={hasError || undefined} />

      {/* Loading skeleton */}
      {statusLoading && !s.cycle && !hasError ? (
        <PageSkeleton />
      ) : isWaiting ? (
        // Show a "waiting" message if the agent hasn't started yet
        <div className="text-center py-20 text-gray-500">
          <p className="text-lg">Waiting for agent to start...</p>
          <p className="text-sm mt-2">
            Start the agent with: python -m api.server
          </p>
        </div>
      ) : (
        <>
          {/* Metrics row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Capital"
              value={`$${(s.capital || 0).toLocaleString()}`}
              color="blue"
            />
            <MetricCard
              label="Total PnL"
              value={`$${(s.total_pnl || 0).toFixed(2)}`}
              color={(s.total_pnl || 0) >= 0 ? "green" : "red"}
            />
            <MetricCard
              label="Win Rate"
              value={`${((s.win_rate || 0) * 100).toFixed(1)}%`}
              subtext={`${s.total_trades || 0} trades`}
            />
            <MetricCard
              label="Regime"
              value={s.regime || "Unknown"}
              subtext={`Cycle #${s.cycle || 0}`}
              color="yellow"
            />
          </div>

          {/* v7.0 System Modules status */}
          {modules && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h2 className="text-sm font-semibold text-gray-300 mb-3">System Modules</h2>
              <div className="flex gap-3 flex-wrap">
                {/* WebSocket */}
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  modules.websocket?.enabled
                    ? "bg-green-900/50 text-green-400 border border-green-700"
                    : "bg-gray-700 text-gray-500 border border-gray-600"
                }`}>
                  WebSocket: {modules.websocket?.enabled ? "ON" : "OFF"}
                </span>
                {/* Notifications */}
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  modules.notifications?.enabled
                    ? "bg-green-900/50 text-green-400 border border-green-700"
                    : "bg-gray-700 text-gray-500 border border-gray-600"
                }`}>
                  Alerts: {modules.notifications?.enabled
                    ? [
                        modules.notifications.channels?.telegram && "TG",
                        modules.notifications.channels?.discord && "DC",
                        modules.notifications.channels?.email && "EM",
                      ].filter(Boolean).join("+") || "ON"
                    : "OFF"}
                </span>
                {/* Trade DB */}
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  modules.trade_db?.enabled
                    ? "bg-green-900/50 text-green-400 border border-green-700"
                    : "bg-gray-700 text-gray-500 border border-gray-600"
                }`}>
                  Trade DB: {modules.trade_db?.enabled ? "ON" : "OFF"}
                </span>
                {/* Intelligence */}
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  modules.intelligence?.enabled
                    ? "bg-green-900/50 text-green-400 border border-green-700"
                    : "bg-gray-700 text-gray-500 border border-gray-600"
                }`}>
                  Intel: {modules.intelligence?.enabled
                    ? `${modules.intelligence.providers_enabled}/5`
                    : "OFF"}
                </span>
                {/* Rate Limiter */}
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-900/50 text-blue-400 border border-blue-700">
                  Rate: {modules.rate_limiter?.max_orders_per_minute || 10} ord/min
                </span>
              </div>
            </div>
          )}

          {/* Multi-pair portfolio overview */}
          {s.multi_pair && s.portfolio && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-gray-300">Portfolio</h2>
                <span className="text-xs text-gray-500">
                  {(s.trading_pairs || []).join(", ")}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div className="text-gray-400">Exposure</div>
                  <div className="font-mono text-white">
                    ${(s.portfolio.total_exposure || 0).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-gray-400">Concentration</div>
                  <div className="font-mono text-white">
                    {(s.portfolio.concentration || 0).toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-400">Correlation Risk</div>
                  <div className={`font-mono ${
                    s.portfolio.corr_risk === "high" ? "text-red-400" :
                    s.portfolio.corr_risk === "medium" ? "text-yellow-400" : "text-green-400"
                  }`}>
                    {s.portfolio.corr_risk || "low"}
                  </div>
                </div>
                <div>
                  <div className="text-gray-400">Weights</div>
                  <div className="font-mono text-xs text-gray-300">
                    {s.portfolio_weights
                      ? Object.entries(s.portfolio_weights as Record<string, number>)
                          .map(([p, w]) => `${p.split("/")[0]}: ${(w * 100).toFixed(0)}%`)
                          .join(", ")
                      : "equal"
                    }
                  </div>
                </div>
              </div>
              {/* Per-pair exposure breakdown */}
              {s.portfolio.pair_exposure && Object.keys(s.portfolio.pair_exposure).length > 0 && (
                <div className="mt-3 flex gap-2 flex-wrap">
                  {Object.entries(s.portfolio.pair_exposure as Record<string, number>).map(([pair, exp]) => (
                    <span key={pair} className="px-2 py-1 bg-gray-700 rounded text-xs">
                      {pair}: ${exp.toLocaleString()}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Equity curve chart */}
          <div>
            <h2 className="text-lg font-semibold mb-3">Equity Curve</h2>
            <EquityChart data={equity?.equity || []} height={250} />
          </div>

          {/* Recent trades table */}
          <div>
            <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <TradeTable trades={(trades?.trades || []) as any[]} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
