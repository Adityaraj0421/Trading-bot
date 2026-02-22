// ============================================================
// useWebSocket.ts - Real-time WebSocket connection hook
// ============================================================
// Manages a WebSocket connection to the agent's API server.
// On each message, injects data into SWR's cache so all
// components using useSWR see instant updates.
//
// Auto-reconnects with exponential backoff (1s -> 2s -> 4s -> max 30s).
// Falls back to SWR polling when disconnected.

"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useSWRConfig } from "swr";
import type { ConnectionState, WebSocketMessage } from "./types";

const WS_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
  .replace(/^http/, "ws");
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
const MAX_RECONNECT_DELAY = 30000;
const PING_INTERVAL = 30000;

/** Map WebSocket snapshot-replace event types to SWR cache keys. */
const SNAPSHOT_EVENTS: Record<string, string> = {
  snapshot: "/status",
};

export function useAgentWebSocket() {
  const { mutate } = useSWRConfig();
  const [state, setState] = useState<ConnectionState>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(1000);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const pingTimer = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    // No API key in URL — auth happens via first message after connecting
    const url = `${WS_URL}/ws`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;

        // Send auth as first message (keeps key out of URL/logs/history)
        if (API_KEY) {
          ws.send(JSON.stringify({ type: "auth", api_key: API_KEY }));
        }

        setState("connected");
        reconnectDelay.current = 1000; // Reset backoff on success

        // Start ping keepalive
        pingTimer.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, PING_INTERVAL);
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const msg: WebSocketMessage = JSON.parse(event.data);

          // Snapshot events: replace the whole SWR cache entry
          const snapshotKey = SNAPSHOT_EVENTS[msg.type];
          if (snapshotKey) {
            mutate(snapshotKey, msg.data, false);
            return;
          }

          // Append events: merge new item into the existing cached collection
          // so the shape always matches what the REST API returns
          if (msg.type === "equity") {
            mutate(
              "/equity",
              (current: { equity: object[]; total_points: number } | undefined) => ({
                equity: [...(current?.equity ?? []), msg.data],
                total_points: (current?.total_points ?? 0) + 1,
              }),
              false
            );
          } else if (msg.type === "trade") {
            mutate(
              "/trades",
              (current: { trades: object[]; total: number } | undefined) => ({
                trades: [msg.data, ...(current?.trades ?? [])],
                total: (current?.total ?? 0) + 1,
              }),
              false
            );
          } else if (msg.type === "event") {
            mutate(
              "/autonomous/events",
              (current: { events: object[]; count: number } | undefined) => ({
                events: [...(current?.events ?? []), msg.data],
                count: (current?.count ?? 0) + 1,
              }),
              false
            );
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        cleanup();
        if (!mountedRef.current) return;
        setState("reconnecting");
        scheduleReconnect();
      };

      ws.onerror = () => {
        // onclose will fire after onerror
      };
    } catch {
      setState("reconnecting");
      scheduleReconnect();
    }
  }, [mutate]);

  const cleanup = useCallback(() => {
    if (pingTimer.current) {
      clearInterval(pingTimer.current);
      pingTimer.current = null;
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current) return; // Already scheduled
    reconnectTimer.current = setTimeout(() => {
      reconnectTimer.current = null;
      // Exponential backoff: 1s, 2s, 4s, 8s, ... max 30s
      reconnectDelay.current = Math.min(
        reconnectDelay.current * 2,
        MAX_RECONNECT_DELAY
      );
      connect();
    }, reconnectDelay.current);
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      cleanup();
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect, cleanup]);

  return { connectionState: state };
}
