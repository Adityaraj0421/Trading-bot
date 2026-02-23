// ============================================================
// trades/page.tsx - Full trade history + PnL breakdown (v7.0)
// ============================================================

"use client";

import useSWR from "swr";
import TradeTable from "@/components/TradeTable";
import PnlBreakdown from "@/components/PnlBreakdown";
import ErrorBanner from "@/components/ErrorBanner";
import PageSkeleton from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function TradesPage() {
  const { data: trades, error: tradesErr, isLoading, mutate: mutateTrades } = useSWR(
    "/trades", () => api.getTrades(500), { refreshInterval: 30000 }
  );
  const { data: pnl, error: pnlErr, mutate: mutatePnl } = useSWR(
    "/pnl-summary", () => api.getPnlSummary(), { refreshInterval: 30000 }
  );

  const error = tradesErr || pnlErr;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Trade History</h1>

      <ErrorBanner error={error} onRetry={() => { mutateTrades(); mutatePnl(); }} />

      {isLoading && !trades ? (
        <PageSkeleton variant="table" />
      ) : (
        <>
          {/* PnL breakdown charts */}
          <PnlBreakdown data={pnl || null} />

          {/* Total trade count */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">All Trades</h2>
            <p className="text-sm text-gray-400">
              Total: {trades?.total || 0} trades
            </p>
          </div>

          {/* Trades table */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <TradeTable trades={(trades?.trades || []) as any[]} />
          </div>
        </>
      )}
    </div>
  );
}
