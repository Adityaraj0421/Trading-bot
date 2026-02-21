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

/** Map WebSocket event types to SWR cache keys. */
const EVENT_TO_SWR_KEY: Record<string, string> = {
  snapshot: "/status",
  equity: "/equity",
  trade: "/trades",
  event: "/autonomous/events",
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

    // Build URL with auth
    const url = API_KEY ? `${WS_URL}/ws?api_key=${API_KEY}` : `${WS_URL}/ws`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
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
          const swrKey = EVENT_TO_SWR_KEY[msg.type];
          if (swrKey) {
            // Inject into SWR cache (false = don't revalidate via HTTP)
            mutate(swrKey, msg.data, false);
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
