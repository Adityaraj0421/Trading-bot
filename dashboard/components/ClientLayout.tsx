// ============================================================
// ClientLayout.tsx - Client-side layout wrapper
// ============================================================
// Wraps the dashboard content with the WebSocket connection
// and passes connection state to the sidebar. This is needed
// because the root layout.tsx is a server component and
// can't use React hooks.

"use client";

import { useAgentWebSocket } from "@/lib/useWebSocket";
import Sidebar from "./Sidebar";
import ConnectionStatus from "./ConnectionStatus";

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { connectionState } = useAgentWebSocket();

  return (
    <div className="flex">
      <aside className="w-56 bg-gray-900 border-r border-gray-800 min-h-screen flex flex-col">
        <Sidebar />
        <div className="mt-auto border-t border-gray-800">
          <ConnectionStatus state={connectionState} />
        </div>
      </aside>
      <main className="flex-1 p-6 overflow-auto min-h-screen">{children}</main>
    </div>
  );
}
