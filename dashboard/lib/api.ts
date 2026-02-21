// ============================================================
// api.ts - Functions to call the Python FastAPI backend (v2.0)
// ============================================================
// v2.0: Added request timeouts, API key authentication,
//       and structured error handling.
//
// The API_URL defaults to http://localhost:8000 (where the
// Python server runs). You can override it by setting the
// NEXT_PUBLIC_API_URL environment variable.
// Set NEXT_PUBLIC_API_KEY for authenticated access.

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
const DEFAULT_TIMEOUT_MS = 10000; // 10 second timeout

/**
 * Create an AbortController with timeout for fetch requests.
 * Prevents dashboard from hanging when the API is unresponsive.
 */
function createTimeoutSignal(timeoutMs: number = DEFAULT_TIMEOUT_MS): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), timeoutMs);
  return controller.signal;
}

/**
 * Build common headers including API key authentication.
 */
function getHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }
  return headers;
}

/**
 * Sleep helper for retry backoff.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Generic fetch helper with retry logic.
 * Retries up to 2 times with exponential backoff (1s, 2s).
 * Includes timeout, API key auth, and structured error handling.
 */
async function fetchAPI<T>(path: string, timeoutMs?: number): Promise<T> {
  const maxRetries = 2;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(`${API_URL}${path}`, {
        signal: createTimeoutSignal(timeoutMs),
        headers: getHeaders(),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      return res.json();
    } catch (err) {
      if (attempt === maxRetries) throw err;
      await sleep(1000 * Math.pow(2, attempt));
    }
  }
  throw new Error("Unreachable");
}

async function postAPI<T>(path: string, body: Record<string, any> = {}, timeoutMs?: number): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: getHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
    signal: createTimeoutSignal(timeoutMs),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

const api = {
  // Health check
  getHealth: () =>
    fetchAPI<{ status: string; agent_running: boolean }>("/health"),

  // Full status snapshot (capital, PnL, regime, etc.)
  getStatus: () => fetchAPI<Record<string, any>>("/status"),

  // Agent configuration
  getConfig: () => fetchAPI<Record<string, any>>("/config"),

  // Completed trades
  getTrades: (limit = 100) =>
    fetchAPI<{ trades: any[]; total: number }>(`/trades?limit=${limit}`),

  // Equity curve data points
  getEquity: (limit = 0) =>
    fetchAPI<{ equity: any[]; total_points: number }>(`/equity?limit=${limit}`),

  // Currently open positions
  getPositions: () =>
    fetchAPI<{ positions: any[]; count: number }>("/positions"),

  // Autonomous system status
  getAutonomousStatus: () =>
    fetchAPI<Record<string, any>>("/autonomous/status"),

  // Autonomous system event log
  getAutonomousEvents: (limit = 50) =>
    fetchAPI<{ events: any[]; count: number }>(
      `/autonomous/events?limit=${limit}`
    ),

  // Backtest (longer timeout for heavy operations)
  runBacktest: (params: { scenario?: string; pair?: string; periods?: number; mode?: string }) =>
    postAPI<{ status: string; message: string }>("/backtest/run", params, 30000),
  getBacktestResults: () =>
    fetchAPI<{ results: any[]; total: number }>("/backtest/results"),
  getBacktestScenarios: () =>
    fetchAPI<{ scenarios: string[] }>("/backtest/scenarios"),
  clearBacktestResults: () =>
    postAPI<{ status: string }>("/backtest/clear"),

  // Intelligence signals
  getIntelligence: () =>
    fetchAPI<Record<string, any>>("/intelligence/signals"),
  getIntelligenceProviders: () =>
    fetchAPI<{ providers: any[] }>("/intelligence/providers"),

  // Arbitrage
  getArbitrageOpportunities: () =>
    fetchAPI<Record<string, any>>("/arbitrage/opportunities"),

  // Risk simulation (longer timeout)
  getRiskSimulation: () =>
    fetchAPI<Record<string, any>>("/risk/simulation"),
  runMonteCarlo: (params: { n_simulations?: number; n_days?: number }) =>
    postAPI<{ status: string; message: string }>("/risk/monte-carlo", params, 30000),
  getStressTests: () =>
    fetchAPI<{ scenarios: any[] }>("/risk/stress-tests"),

  // Production safeguards
  emergencyHalt: (reason?: string) =>
    postAPI<{ status: string }>("/autonomous/halt", { reason: reason || "Manual kill switch" }),
  emergencyResume: () =>
    postAPI<{ status: string }>("/autonomous/resume"),
  forceCloseAll: () =>
    postAPI<{ status: string }>("/autonomous/force-close"),
  getAlerts: (unacknowledged = false) =>
    fetchAPI<{ alerts: any[]; count: number }>(`/autonomous/alerts?unacknowledged=${unacknowledged}`),
  acknowledgeAlerts: () =>
    postAPI<{ status: string }>("/autonomous/alerts/acknowledge"),

  // v7.0 — System modules
  getSystemModules: () =>
    fetchAPI<Record<string, any>>("/system/modules"),
  getRateLimiterStats: () =>
    fetchAPI<Record<string, any>>("/system/rate-limiter"),

  // v7.0 — Notifications
  getNotifications: (limit = 100) =>
    fetchAPI<{ notifications: any[]; count: number }>(`/notifications?limit=${limit}`),

  // v7.0 — PnL Summary (by pair, by strategy, cumulative curve)
  getPnlSummary: () =>
    fetchAPI<{
      by_pair: Record<string, { trades: number; pnl: number; wins: number; losses: number }>;
      by_strategy: Record<string, { trades: number; pnl: number; wins: number; losses: number }>;
      cumulative_pnl: { pnl: number; timestamp: string; symbol: string }[];
      total_closed: number;
    }>("/pnl-summary"),
};

export default api;
