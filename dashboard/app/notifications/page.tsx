// ============================================================
// notifications/page.tsx — Notification & Alert History (v7.0)
// ============================================================

"use client";

import useSWR from "swr";
import ErrorBanner from "@/components/ErrorBanner";
import { CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

const typeColors: Record<string, string> = {
  trade_open: "text-green-400",
  trade_close: "text-blue-400",
  error: "text-red-400",
  state_change: "text-yellow-400",
  startup: "text-purple-400",
  daily_summary: "text-cyan-400",
  large_loss: "text-red-500",
};

const channelBadge: Record<string, string> = {
  telegram: "bg-blue-900/50 text-blue-300 border-blue-700",
  discord: "bg-purple-900/50 text-purple-300 border-purple-700",
  email: "bg-gray-700 text-gray-300 border-gray-600",
};

export default function NotificationsPage() {
  const { data, error, isLoading, mutate } = useSWR(
    "/notifications", () => api.getNotifications(200), { refreshInterval: 30000 }
  );
  const { data: modules } = useSWR("/system/modules", () =>
    api.getSystemModules()
  );

  const notifs = data?.notifications || [];
  const noChannels = modules && !modules.notifications?.enabled;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Notifications</h1>
        <span className="text-sm text-gray-500">{notifs.length} events</span>
      </div>

      <ErrorBanner error={error} onRetry={() => mutate()} />

      {/* Channel status */}
      {modules?.notifications && (
        <div className="flex gap-2">
          {Object.entries(modules.notifications.channels || {}).map(
            ([ch, enabled]: [string, any]) => (
              <span
                key={ch}
                className={`px-3 py-1 rounded-full text-xs font-medium border ${
                  enabled
                    ? "bg-green-900/50 text-green-400 border-green-700"
                    : "bg-gray-800 text-gray-500 border-gray-700"
                }`}
              >
                {ch}: {enabled ? "ON" : "OFF"}
              </span>
            )
          )}
        </div>
      )}

      {isLoading && !data ? (
        <CardListSkeleton count={6} />
      ) : noChannels ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
          <p className="text-lg">No notification channels configured</p>
          <p className="text-sm mt-2">
            Add TELEGRAM_BOT_TOKEN or DISCORD_WEBHOOK_URL in .env
          </p>
        </div>
      ) : notifs.length === 0 ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
          <p>No notifications yet — they will appear as the agent sends alerts</p>
        </div>
      ) : (
        <div className="space-y-2">
          {[...notifs].reverse().map((n: any, i: number) => (
            <div
              key={i}
              className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex items-start gap-3"
            >
              <div className="flex-shrink-0 mt-0.5">
                <span
                  className={`text-xs font-mono font-bold ${
                    typeColors[n.type] || "text-gray-400"
                  }`}
                >
                  {(n.type || "info").toUpperCase()}
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-200 break-words">
                  {n.message || "—"}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {n.timestamp
                    ? new Date(n.timestamp).toLocaleString()
                    : "—"}
                </p>
              </div>

              <div className="flex-shrink-0 flex items-center gap-2">
                {n.channel && (
                  <span
                    className={`px-2 py-0.5 rounded text-xs border ${
                      channelBadge[n.channel] ||
                      "bg-gray-700 text-gray-400 border-gray-600"
                    }`}
                  >
                    {n.channel}
                  </span>
                )}
                {n.success === false && (
                  <span className="px-2 py-0.5 rounded text-xs bg-red-900/50 text-red-400 border border-red-700">
                    FAILED
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
