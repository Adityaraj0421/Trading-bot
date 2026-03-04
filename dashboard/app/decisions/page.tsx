// ============================================================
// decisions/page.tsx — Phase 9 decision audit log
// ============================================================
// Shows the last N evaluate() calls from data/phase9_decisions.jsonl.
// Columns: time, symbol, bias, confidence, action, reason, score, triggers.
// Auto-refreshes every 15 seconds to catch each context window's decision.

"use client";

import { useState } from "react";
import useSWR from "swr";
import api from "@/lib/api";

// ─── helpers ────────────────────────────────────────────────────────────────

function biasColor(bias: string): string {
  if (bias === "bullish") return "text-emerald-400";
  if (bias === "bearish") return "text-red-400";
  return "text-slate-400";
}

function actionColor(action: string): string {
  return action === "trade" ? "text-emerald-400 font-semibold" : "text-slate-400";
}

function reasonLabel(reason: string): string {
  // Convert snake_case reasons into readable short labels
  const map: Record<string, string> = {
    ok: "✅ ok",
    context_not_tradeable: "ctx not tradeable",
    no_allowed_directions: "no directions",
    funding_extreme_blocks_direction: "funding extreme",
    no_valid_triggers: "no triggers",
    insufficient_directional_agreement: "no consensus",
    event_blocked_by_risk_mode: "risk mode block",
  };
  if (reason in map) return map[reason];
  // Handle dynamic reasons like "score_below_threshold:0.28"
  if (reason.startsWith("score_below_threshold:")) {
    const score = reason.split(":")[1];
    return `score low (${score})`;
  }
  return reason;
}

function fmtTime(ts: string): string {
  try {
    return new Date(ts).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
}

// ─── component ──────────────────────────────────────────────────────────────

export default function DecisionsPage() {
  const [limit, setLimit] = useState(50);

  const { data, error, isLoading, mutate } = useSWR(
    `/phase9/decisions?limit=${limit}`,
    () => api.getPhase9Decisions(limit),
    { refreshInterval: 15000 }
  );

  const decisions = data?.decisions ?? [];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Phase 9 Decisions</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Audit log from <code className="text-xs text-cyan-400">data/phase9_decisions.jsonl</code>
            {" — "}auto-refreshes every 15 s
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Limit selector */}
          <label className="text-xs text-slate-400">Show last</label>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none"
          >
            {[25, 50, 100, 200].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          <button
            onClick={() => mutate()}
            className="text-xs px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 transition-colors"
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Stats bar */}
      {decisions.length > 0 && (
        <div className="flex gap-6 text-sm">
          {(() => {
            const trades = decisions.filter((d: any) => d.decision?.action === "trade").length;
            const rejects = decisions.length - trades;
            const latestCtx = decisions[0]?.context;
            return (
              <>
                <div>
                  <span className="text-slate-400">Showing </span>
                  <span className="font-medium text-slate-200">{decisions.length}</span>
                </div>
                <div>
                  <span className="text-emerald-400 font-medium">{trades}</span>
                  <span className="text-slate-400"> trades</span>
                </div>
                <div>
                  <span className="text-slate-400 font-medium">{rejects}</span>
                  <span className="text-slate-400"> rejects</span>
                </div>
                {latestCtx && (
                  <div>
                    <span className="text-slate-400">Latest bias: </span>
                    <span className={`font-medium ${biasColor(latestCtx.swing_bias)}`}>
                      {latestCtx.swing_bias}
                    </span>
                    <span className="text-slate-500 ml-1">
                      (conf {(latestCtx.confidence ?? 0).toFixed(3)})
                    </span>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}

      {/* Log path notice when not configured */}
      {!isLoading && data?.log_path === null && (
        <div className="bg-amber-900/30 border border-amber-700 rounded-lg p-4 text-sm text-amber-300">
          <strong>PHASE9_DECISION_LOG_PATH</strong> is not set in <code>.env</code>.
          Add <code>PHASE9_DECISION_LOG_PATH=data/phase9_decisions.jsonl</code> and restart the server.
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-300">
          Failed to load decisions: {String(error)}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-slate-700">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 bg-slate-800/60">
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400 whitespace-nowrap">Time</th>
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400">Symbol</th>
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400">Bias</th>
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400">Conf</th>
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400">Vol Regime</th>
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400">Action</th>
              <th className="px-3 py-2.5 text-left text-xs font-medium text-slate-400">Reason</th>
              <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-400">Score</th>
              <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-400">Dir</th>
              <th className="px-3 py-2.5 text-right text-xs font-medium text-slate-400">Trigs</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && decisions.length === 0 ? (
              // Loading skeleton rows
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i} className="border-b border-slate-800 animate-pulse">
                  {Array.from({ length: 10 }).map((__, j) => (
                    <td key={j} className="px-3 py-2">
                      <div className="h-3 rounded bg-slate-700 w-16" />
                    </td>
                  ))}
                </tr>
              ))
            ) : decisions.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-3 py-8 text-center text-slate-500">
                  {data?.log_path !== null
                    ? "No decisions logged yet — waiting for first evaluate() call."
                    : "Log path not configured."}
                </td>
              </tr>
            ) : (
              decisions.map((entry: any, i: number) => {
                const ctx = entry.context ?? {};
                const dec = entry.decision ?? {};
                const trigCount = Array.isArray(entry.triggers) ? entry.triggers.length : 0;
                const isTradeRow = dec.action === "trade";
                return (
                  <tr
                    key={i}
                    className={`border-b border-slate-800/60 transition-colors ${
                      isTradeRow
                        ? "bg-emerald-950/20 hover:bg-emerald-950/40"
                        : "hover:bg-slate-800/40"
                    }`}
                  >
                    <td className="px-3 py-2 text-xs text-slate-400 whitespace-nowrap font-mono">
                      {fmtTime(entry.ts)}
                    </td>
                    <td className="px-3 py-2 text-xs font-medium text-slate-200">
                      {entry.symbol ?? "—"}
                    </td>
                    <td className={`px-3 py-2 text-xs ${biasColor(ctx.swing_bias ?? "")}`}>
                      {ctx.swing_bias ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-300 tabular-nums">
                      {ctx.confidence != null ? ctx.confidence.toFixed(3) : "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-400">
                      {ctx.volatility_regime ?? "—"}
                    </td>
                    <td className={`px-3 py-2 text-xs ${actionColor(dec.action ?? "")}`}>
                      {dec.action ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-300">
                      {reasonLabel(dec.reason ?? "")}
                    </td>
                    <td className="px-3 py-2 text-xs text-right tabular-nums text-slate-300">
                      {dec.score != null ? dec.score.toFixed(3) : "—"}
                    </td>
                    <td className={`px-3 py-2 text-xs text-right font-medium ${
                      dec.direction === "long"
                        ? "text-emerald-400"
                        : dec.direction === "short"
                        ? "text-red-400"
                        : "text-slate-500"
                    }`}>
                      {dec.direction ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-right tabular-nums text-slate-400">
                      {trigCount}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Trigger detail for trade rows */}
      {decisions.some((d: any) => d.decision?.action === "trade" && Array.isArray(d.triggers) && d.triggers.length > 0) && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-slate-300">Trade Signal Details</h2>
          {decisions
            .filter((d: any) => d.decision?.action === "trade" && Array.isArray(d.triggers) && d.triggers.length > 0)
            .map((entry: any, i: number) => (
              <div key={i} className="bg-emerald-950/30 border border-emerald-800/50 rounded-lg p-3">
                <div className="flex items-center gap-3 mb-2 text-xs text-slate-300">
                  <span className="font-mono text-slate-400">{fmtTime(entry.ts)}</span>
                  <span className="font-medium text-slate-200">{entry.symbol}</span>
                  <span className="text-emerald-400">
                    {entry.decision.direction} via {entry.decision.route}
                    {entry.decision.score != null && ` (score ${entry.decision.score.toFixed(3)})`}
                  </span>
                </div>
                <div className="space-y-1">
                  {entry.triggers.map((t: any, j: number) => (
                    <div key={j} className="text-xs text-slate-400 pl-2 border-l border-emerald-700">
                      <span className="text-cyan-400">[{t.source}]</span>{" "}
                      <span>{t.reason}</span>{" "}
                      <span className="text-slate-500">(str {(t.strength ?? 0).toFixed(2)})</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
