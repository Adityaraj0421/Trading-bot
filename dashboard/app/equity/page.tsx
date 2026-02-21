// ============================================================
// equity/page.tsx - Full equity curve page
// ============================================================

"use client";

import useSWR from "swr";
import EquityChart from "@/components/EquityChart";
import ErrorBanner from "@/components/ErrorBanner";
import PageSkeleton from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function EquityPage() {
  const { data, error, isLoading, mutate } = useSWR(
    "/equity-full", () => api.getEquity(), { refreshInterval: 30000 }
  );

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Equity Curve</h1>

      <ErrorBanner error={error} onRetry={() => mutate()} />

      {isLoading && !data ? (
        <PageSkeleton variant="chart" />
      ) : (
        <>
          <EquityChart data={data?.equity || []} height={500} />

          <p className="text-sm text-gray-500">
            Total data points: {data?.total_points || 0}
          </p>
        </>
      )}
    </div>
  );
}
