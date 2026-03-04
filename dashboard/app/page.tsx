// ============================================================
// page.tsx - Home / Overview page (the main dashboard)
// ============================================================

"use client";

import useSWR from "swr";
import MetricCard from "@/components/MetricCard";
import EquityChart from "@/components/EquityChart";
import StatusBadge from "@/components/StatusBadge";
import TradeTable from "@/components/TradeTable";
import ErrorBanner from "@/components/ErrorBanner";
import PageSkeleton from "@/components/PageSkeleton";
import api from "@/lib/api";

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

export default function Home() {
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
  const { data: lastDecisionData } = useSWR(
    "/phase9/decisions?limit=1",
    () => api.getPhase9Decisions(1),
    { refreshInterval: 15000 }
  );

  const s = (status as Record<string, any>) || {};
  const isWaiting = !s.cycle;
  const hasError = statusErr || equityErr || tradesErr;

  // Last Phase 9 decision — from JSONL audit log (newest first)
  const lastDecision = lastDecisionData?.decisions?.[0] ?? null;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {s.autonomous && <StatusBadge state={s.autonomous.state} />}
      </div>

      <ErrorBanner error={hasError || undefined} />

      {statusLoading && !s.cycle && !hasError ? (
        <PageSkeleton />
      ) : isWaiting ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="w-10 h-10 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin mb-4" />
          <p className="text-slate-400 font-medium">Waiting for agent to start…</p>
          <p className="text-slate-600 text-sm mt-2">
            Run: <code className="text-cyan-500 font-mono">python -m api.server</code>
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
              value={s.total_trades ? `${((s.win_rate || 0) * 100).toFixed(1)}%` : "—"}
              subtext={s.total_trades ? `${s.total_trades} trades` : "no trades yet"}
            />
            <MetricCard
              label="Regime"
              value={s.regime || "Unknown"}
              subtext={`Cycle #${s.cycle || 0}`}
              color="yellow"
            />
          </div>

          {/* Phase 9 Last Decision panel */}
          <div className="bg-slate-900 rounded-xl border border-violet-900/50 p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-sm">🤖</span>
                <h2 className="text-sm font-semibold text-violet-300">Phase 9 — Last Decision</h2>
              </div>
              {lastDecision && (
                <span className="text-xs text-slate-600">
                  {timeAgo(lastDecision.ts)}
                </span>
              )}
            </div>
            {lastDecision ? (() => {
              const ctx = lastDecision.context ?? {};
              const dec = lastDecision.decision ?? {};
              const isTraded = dec.action === "trade";
              return (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm">
                  {/* Symbol */}
                  <span className="text-xs font-medium text-slate-300">
                    {lastDecision.symbol ?? "—"}
                  </span>
                  {/* Action badge */}
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${
                    isTraded
                      ? "text-emerald-400 bg-emerald-950/50 border-emerald-800"
                      : "text-slate-500 bg-slate-800 border-slate-700"
                  }`}>
                    {dec.action ?? "—"}
                  </span>
                  {/* Reason */}
                  <span className="text-slate-400 text-xs">{dec.reason ?? "—"}</span>
                  {/* Context summary */}
                  <span className={`text-xs ${
                    ctx.swing_bias === "bullish" ? "text-emerald-400" :
                    ctx.swing_bias === "bearish" ? "text-red-400" : "text-slate-500"
                  }`}>
                    {ctx.swing_bias ?? "—"}
                  </span>
                  <span className="text-slate-600 text-xs">
                    conf {(ctx.confidence ?? 0).toFixed(3)}
                  </span>
                  {dec.score != null && (
                    <span className="text-violet-400 text-xs ml-auto">
                      score {dec.score.toFixed(3)}
                    </span>
                  )}
                </div>
              );
            })() : (
              <p className="text-slate-600 text-sm">No decisions yet — agent is initializing…</p>
            )}
          </div>

          {/* System Modules */}
          {modules && (
            <div className="bg-slate-900 rounded-xl border border-white/5 p-4">
              <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                System Modules
              </h2>
              <div className="flex gap-2 flex-wrap">
                <ModuleChip
                  label={`WS Feed: ${modules.websocket?.enabled ? "ON" : "OFF"}`}
                  active={modules.websocket?.enabled}
                  color="cyan"
                />
                <ModuleChip
                  label={`Alerts: ${
                    modules.notifications?.enabled
                      ? [
                          modules.notifications.channels?.telegram && "TG",
                          modules.notifications.channels?.discord && "DC",
                          modules.notifications.channels?.email && "EM",
                        ]
                          .filter(Boolean)
                          .join("+") || "ON"
                      : "OFF"
                  }`}
                  active={modules.notifications?.enabled}
                  color="cyan"
                />
                <ModuleChip
                  label={`Trade DB: ${modules.trade_db?.enabled ? "ON" : "OFF"}`}
                  active={modules.trade_db?.enabled}
                  color="cyan"
                />
                <ModuleChip
                  label={`Intel ${modules.intelligence?.enabled ? `${modules.intelligence.providers_enabled}/10` : "OFF"}`}
                  active={modules.intelligence?.enabled}
                  color="cyan"
                />
                <ModuleChip
                  label="Phase 9: ON"
                  active={true}
                  color="violet"
                />
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 border border-white/5">
                  {modules.rate_limiter?.max_orders_per_minute || 10} ord/min
                </span>
              </div>
            </div>
          )}

          {/* Multi-pair portfolio */}
          {s.multi_pair && s.portfolio && (
            <div className="bg-slate-900 rounded-xl border border-white/5 p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Portfolio
                </h2>
                <span className="text-xs text-slate-600">
                  {(s.trading_pairs || []).join(", ")}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div className="text-slate-500 text-xs">Exposure</div>
                  <div className="font-mono text-slate-200">
                    ${(s.portfolio.total_exposure || 0).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs">Concentration</div>
                  <div className="font-mono text-slate-200">
                    {(s.portfolio.concentration || 0).toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs">Corr. Risk</div>
                  <div
                    className={`font-mono ${
                      s.portfolio.corr_risk === "high"
                        ? "text-rose-400"
                        : s.portfolio.corr_risk === "medium"
                        ? "text-amber-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {s.portfolio.corr_risk || "low"}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs">Weights</div>
                  <div className="font-mono text-xs text-slate-400">
                    {s.portfolio_weights
                      ? Object.entries(
                          s.portfolio_weights as Record<string, number>
                        )
                          .map(
                            ([p, w]) =>
                              `${p.split("/")[0]}: ${(w * 100).toFixed(0)}%`
                          )
                          .join(", ")
                      : "equal"}
                  </div>
                </div>
              </div>
              {s.portfolio.pair_exposure &&
                Object.keys(s.portfolio.pair_exposure).length > 0 && (
                  <div className="mt-3 flex gap-2 flex-wrap">
                    {Object.entries(
                      s.portfolio.pair_exposure as Record<string, number>
                    ).map(([pair, exp]) => (
                      <span
                        key={pair}
                        className="px-2 py-1 bg-slate-800 border border-white/5 rounded text-xs text-slate-400"
                      >
                        {pair}: ${exp.toLocaleString()}
                      </span>
                    ))}
                  </div>
                )}
            </div>
          )}

          {/* Equity curve */}
          <div>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Equity Curve
            </h2>
            <EquityChart data={equity?.equity || []} height={250} />
          </div>

          {/* Recent trades */}
          <div>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Recent Trades
            </h2>
            <div className="bg-slate-900 rounded-xl border border-white/5 p-4">
              <TradeTable trades={(trades?.trades || []) as any[]} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Small helper component ────────────────────────────────────────────────────
function ModuleChip({
  label,
  active,
  color = "cyan",
}: {
  label: string;
  active?: boolean;
  color?: "cyan" | "violet";
}) {
  if (!active) {
    return (
      <span className="px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-500 border border-white/5">
        {label}
      </span>
    );
  }
  const styles =
    color === "violet"
      ? "bg-violet-950/50 text-violet-400 border-violet-800"
      : "bg-cyan-950/50 text-cyan-400 border-cyan-800";
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${styles}`}>
      {label}
    </span>
  );
}
