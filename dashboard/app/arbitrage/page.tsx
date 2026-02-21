"use client";

import useSWR from "swr";
import ErrorBanner from "@/components/ErrorBanner";
import { MetricsSkeleton, CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function ArbitragePage() {
  const { data, error, isLoading, mutate } = useSWR(
    "/arbitrage/opportunities", () => api.getArbitrageOpportunities(),
    { refreshInterval: 30000 }
  );

  const isDisabled = data?.status === "not_enabled";
  const opportunities = data?.opportunities || [];
  const status = data || {};

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Multi-Exchange Arbitrage</h1>

      <ErrorBanner error={error} onRetry={() => mutate()} />

      {isLoading && !data ? (
        <div className="space-y-4">
          <MetricsSkeleton count={2} />
          <CardListSkeleton count={3} />
        </div>
      ) : isDisabled ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
          <p className="text-lg">Arbitrage is disabled</p>
          <p className="text-sm mt-2">
            Set ARBITRAGE_ENABLED=true in .env and configure exchange API keys
          </p>
        </div>
      ) : (
        <>
          {/* Status bar */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center gap-6 text-sm flex-wrap">
              <div>
                <span className="text-gray-400">Scans:</span>{" "}
                <span className="font-mono">{status.scan_count || 0}</span>
              </div>
              <div>
                <span className="text-gray-400">Active opportunities:</span>{" "}
                <span className="font-mono text-green-400">
                  {status.active_opportunities || 0}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Min spread:</span>{" "}
                <span className="font-mono">
                  {status.min_spread_pct?.toFixed(3) || "0.300"}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Exchanges:</span>{" "}
                <span className="font-mono">
                  {(status.exchanges_monitored || []).join(", ") || "none"}
                </span>
              </div>
            </div>
          </div>

          {/* Opportunities list */}
          {opportunities.length > 0 ? (
            <div className="space-y-3">
              {opportunities.map((opp: any, i: number) => (
                <div
                  key={i}
                  className="bg-gray-800 rounded-lg border border-gray-700 p-4"
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">
                      {opp.buy_exchange} &rarr; {opp.sell_exchange}
                    </span>
                    <span className="text-green-400 font-mono text-sm">
                      +{opp.net_profit_pct?.toFixed(4)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-gray-400">
                    <div>Buy: ${opp.buy_price?.toLocaleString()}</div>
                    <div>Sell: ${opp.sell_price?.toLocaleString()}</div>
                    <div>Spread: {opp.spread_pct?.toFixed(4)}%</div>
                    <div>Fees: {opp.fees_pct?.toFixed(4)}%</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
              <p>No arbitrage opportunities detected</p>
              <p className="text-sm mt-2">
                The scanner checks for spreads exceeding the minimum threshold
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
