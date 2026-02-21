// ============================================================
// ConnectionStatus.tsx - WebSocket connection indicator
// ============================================================
// Shows a small dot in the sidebar indicating the WebSocket
// connection state: green (connected), yellow (reconnecting),
// red (disconnected).

"use client";

import type { ConnectionState } from "@/lib/types";

const stateConfig: Record<ConnectionState, { color: string; label: string }> = {
  connected: { color: "bg-green-500", label: "Live" },
  reconnecting: { color: "bg-yellow-500 animate-pulse", label: "Reconnecting" },
  disconnected: { color: "bg-red-500", label: "Offline" },
};

export default function ConnectionStatus({ state }: { state: ConnectionState }) {
  const { color, label } = stateConfig[state];

  return (
    <div className="flex items-center gap-2 px-3 py-2 text-xs text-gray-400">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      <span>{label}</span>
    </div>
  );
}
