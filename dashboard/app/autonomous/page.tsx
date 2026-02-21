// ============================================================
// autonomous/page.tsx - Autonomous trading system page
// ============================================================

"use client";

import { useState } from "react";
import useSWR from "swr";
import StatusBadge from "@/components/StatusBadge";
import MetricCard from "@/components/MetricCard";
import ErrorBanner from "@/components/ErrorBanner";
import { MetricsSkeleton, CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function AutonomousPage() {
  const { data: status, error: statusErr, isLoading, mutate: mutateStatus } = useSWR(
    "/auto-status",
    () => api.getAutonomousStatus(),
    { refreshInterval: 30000 }
  );
  const { data: events, error: eventsErr } = useSWR(
    "/auto-events",
    () => api.getAutonomousEvents(),
    { refreshInterval: 30000 }
  );
  const { data: alerts, mutate: mutateAlerts } = useSWR(
    "/auto-alerts",
    () => api.getAlerts(true),
    { refreshInterval: 30000 }
  );

  const [confirming, setConfirming] = useState<string | null>(null);

  const s = (status as Record<string, any>) || {};
  const isHalted = s.state === "halted";
  const unackedAlerts = alerts?.alerts || [];
  const error = statusErr || eventsErr;

  const handleHalt = async () => {
    if (confirming !== "halt") { setConfirming("halt"); return; }
    await api.emergencyHalt();
    setConfirming(null);
    mutateStatus();
  };

  const handleResume = async () => {
    await api.emergencyResume();
    mutateStatus();
  };

  const handleForceClose = async () => {
    if (confirming !== "close") { setConfirming("close"); return; }
    await api.forceCloseAll();
    setConfirming(null);
  };

  return (
    <div className="space-y-6">
      {/* Header with state badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Autonomous System</h1>
          <StatusBadge state={s.state || "unknown"} />
        </div>

        {/* Kill switch controls */}
        <div className="flex gap-2">
          {isHalted ? (
            <button
              onClick={handleResume}
              className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-sm font-medium"
            >
              Resume Trading
            </button>
          ) : (
            <button
              onClick={handleHalt}
              className={`px-4 py-2 rounded text-sm font-medium ${
                confirming === "halt"
                  ? "bg-red-700 animate-pulse"
                  : "bg-red-600 hover:bg-red-700"
              }`}
            >
              {confirming === "halt" ? "Confirm HALT?" : "Emergency Halt"}
            </button>
          )}
          <button
            onClick={handleForceClose}
            className={`px-4 py-2 rounded text-sm font-medium ${
              confirming === "close"
                ? "bg-orange-700 animate-pulse"
                : "bg-orange-600 hover:bg-orange-700"
            }`}
          >
            {confirming === "close" ? "Confirm Close All?" : "Force Close All"}
          </button>
        </div>
      </div>

      <ErrorBanner error={error} onRetry={() => mutateStatus()} />

      {isLoading && !status ? (
        <div className="space-y-4">
          <MetricsSkeleton />
          <CardListSkeleton count={5} />
        </div>
      ) : (
        <>
          {/* Alerts */}
          {unackedAlerts.length > 0 && (
            <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-semibold text-red-400">
                  Active Alerts ({unackedAlerts.length})
                </h3>
                <button
                  onClick={async () => { await api.acknowledgeAlerts(); mutateAlerts(); }}
                  className="text-xs text-gray-400 hover:text-white"
                >
                  Acknowledge All
                </button>
              </div>
              {unackedAlerts.map((a: any, i: number) => (
                <div key={i} className="flex gap-3 items-center text-sm">
                  <span className={`w-2 h-2 rounded-full ${
                    a.severity === "critical" ? "bg-red-500" : "bg-orange-500"
                  }`} />
                  <span className="text-gray-400 text-xs">
                    {new Date(a.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="text-gray-200">{a.message}</span>
                </div>
              ))}
            </div>
          )}

          {/* Metrics row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Daily PnL"
              value={`$${(s.daily_pnl || 0).toFixed(2)}`}
              color={(s.daily_pnl || 0) >= 0 ? "green" : "red"}
            />
            <MetricCard
              label="Loss Streak"
              value={String(s.consecutive_losses || 0)}
              color={(s.consecutive_losses || 0) >= 3 ? "red" : "default"}
            />
            <MetricCard
              label="Decisions"
              value={String(s.total_autonomous_decisions || 0)}
            />
            <MetricCard label="State" value={s.state || "unknown"} color="yellow" />
          </div>

          {/* Event log */}
          <div>
            <h2 className="text-lg font-semibold mb-3">Recent Events</h2>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-2">
              {(events?.events || []).length === 0 ? (
                <p className="text-gray-500">No events yet.</p>
              ) : (
                (events?.events as any[])
                  .slice()
                  .reverse()
                  .map((e: any, i: number) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 py-1 border-b border-gray-700/50 last:border-0"
                    >
                      <span className="text-xs text-gray-500">
                        {new Date(e.timestamp).toLocaleString()}
                      </span>
                      <span className="text-xs font-medium text-blue-400">
                        [{e.type}]
                      </span>
                      <span className="text-sm text-gray-300">
                        {e.description}
                      </span>
                    </div>
                  ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
