// ============================================================
// types.ts - TypeScript interfaces for the trading agent data
// ============================================================
// These types define the shape of data returned by the Python
// FastAPI backend. Using types helps catch bugs early and gives
// your editor autocomplete support.

/**
 * A full snapshot of the agent's current state.
 * Returned by the /status endpoint.
 */
export interface Snapshot {
  cycle: number;
  price: number;
  pair: string;
  trading_pair: string;
  capital: number;
  total_pnl: number;
  daily_pnl: number;
  total_fees: number;
  win_rate: number;
  total_trades: number;
  open_positions: number;
  regime: string;
  positions: Position[];
  autonomous: AutonomousStatus;
  updated_at: string;
}

/**
 * A single open position the agent is holding.
 */
export interface Position {
  symbol: string;
  side: string;
  entry_price: number;
  quantity: number;
  unrealized_pnl: number;
  stop_loss: number;
  take_profit: number;
  trailing_stop: number;
  strategy?: string;
  strategy_name?: string;
}

/**
 * A completed (closed) trade with entry and exit details.
 */
export interface Trade {
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl_gross: number;
  pnl_net: number;
  fees_paid: number;
  entry_time: string;
  exit_time: string;
  exit_reason: string;
  strategy?: string;
  strategy_name?: string;
  hold_bars: number;
}

/**
 * A single point on the equity curve (portfolio value over time).
 */
export interface EquityPoint {
  equity: number;
  timestamp: string;
}

/**
 * Status of the autonomous trading system (risk management).
 */
export interface AutonomousStatus {
  state: string;
  daily_pnl: number;
  consecutive_losses: number;
  total_autonomous_decisions: number;
  recent_events: AutonomousEvent[];
}

/**
 * An event logged by the autonomous system (e.g., "reduced position size").
 */
export interface AutonomousEvent {
  type: string;
  description: string;
  timestamp: string;
}

// ============================================================
// v7.0 Types
// ============================================================

/**
 * A notification sent by the agent (Telegram, Discord, Email).
 */
export interface Notification {
  type: string;           // "trade_open" | "trade_close" | "error" | "state_change" | "startup" | "daily_summary"
  channel: string;        // "telegram" | "discord" | "email"
  message: string;
  timestamp: string;
  success: boolean;
}

/**
 * Status of a single v7.0 system module.
 */
export interface SystemModules {
  websocket: { enabled: boolean; status: string };
  notifications: {
    enabled: boolean;
    channels: { telegram: boolean; discord: boolean; email: boolean };
  };
  trade_db: { enabled: boolean; status: string };
  rate_limiter: {
    max_requests_per_minute: number;
    max_orders_per_minute: number;
  };
  intelligence: {
    enabled: boolean;
    providers_enabled: number;
  };
}

/**
 * Rate limiter usage stats.
 */
export interface RateLimiterStats {
  max_requests_per_minute: number;
  max_orders_per_minute: number;
  requests_used: number;
  orders_used: number;
}

// ============================================================
// v7.1 WebSocket Types
// ============================================================

/** A message received from the WebSocket server. */
export interface WebSocketMessage {
  type: "snapshot" | "equity" | "trade" | "event" | "alert" | "pong";
  data: Record<string, any>;
  ts: string;
}

/** WebSocket connection state for the dashboard. */
export type ConnectionState = "connected" | "reconnecting" | "disconnected";

/** Portfolio overview data (nested in Snapshot). */
export interface PortfolioOverview {
  total_exposure: number;
  concentration: number;
  corr_risk: string;
  pair_exposure: Record<string, number>;
}
