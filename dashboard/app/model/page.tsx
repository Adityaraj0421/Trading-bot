"use client";

import useSWR from "swr";
import ErrorBanner from "@/components/ErrorBanner";
import { MetricsSkeleton, CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function ModelPage() {
  const { data, error, isLoading, mutate } = useSWR(
    "/model/feature-importance",
    () => api.getFeatureImportance(),
    { refreshInterval: 60000 }
  );

  const isNotTrained = data?.status === "not_trained";
  const hasImportance =
    data?.feature_importance && Object.keys(data.feature_importance).length > 0;

  // Normalise bars against the highest importance value
  const importanceEntries: [string, number][] = hasImportance
    ? (Object.entries(data!.feature_importance!) as [string, number][])
    : [];
  const maxImportance = importanceEntries.length
    ? Math.max(...importanceEntries.map(([, v]) => v))
    : 1;

  const topSet = new Set(data?.top_features ?? []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">ML Model</h1>

      <ErrorBanner error={error} onRetry={() => mutate()} />

      {isLoading && !data ? (
        <div className="space-y-4">
          <MetricsSkeleton count={4} />
          <CardListSkeleton count={6} />
        </div>
      ) : isNotTrained ? (
        <>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center text-gray-500">
            <p className="text-lg">Model not yet trained</p>
            <p className="text-sm mt-2">
              {data?.message ?? "Waiting for first training cycle..."}
            </p>
            {data?.n_features != null && (
              <p className="text-xs mt-1 text-gray-600">
                {data.n_features} features configured
              </p>
            )}
          </div>
          {/* Show configured feature columns even before training */}
          {data?.feature_cols && data.feature_cols.length > 0 && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Configured Feature Columns ({data.feature_cols.length})
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.feature_cols.map((col) => (
                  <span
                    key={col}
                    className="px-2 py-1 rounded text-xs font-mono bg-gray-700 text-gray-400"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      ) : !hasImportance ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
          <p>Feature importance data unavailable</p>
          <p className="text-sm mt-1">Requires XGBoost tier (Tier 2 or Tier 1 ensemble).</p>
        </div>
      ) : (
        <>
          {/* Summary row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <p className="text-sm text-gray-400">Tier</p>
              <p className="text-2xl font-bold font-mono">{data?.tier ?? "—"}</p>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <p className="text-sm text-gray-400">Active Features</p>
              <p className="text-2xl font-bold font-mono">{data?.n_features ?? "—"}</p>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <p className="text-sm text-gray-400">Top-K Highlighted</p>
              <p className="text-2xl font-bold font-mono text-blue-400">
                {data?.top_features?.length ?? "—"}
              </p>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <p className="text-sm text-gray-400">Status</p>
              <p className="text-xl font-bold text-green-400 capitalize">
                {data?.status ?? "—"}
              </p>
            </div>
          </div>

          {/* Feature importance chart */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
              Feature Importance (XGBoost)
            </h2>
            <div className="space-y-2">
              {importanceEntries.map(([feature, importance]) => {
                const barPct = maxImportance > 0 ? (importance / maxImportance) * 100 : 0;
                const isTop = topSet.has(feature);
                return (
                  <div key={feature} className="flex items-center gap-3">
                    {/* Feature name */}
                    <span
                      className={`w-36 text-xs font-mono truncate shrink-0 ${
                        isTop ? "text-blue-400 font-semibold" : "text-gray-400"
                      }`}
                      title={feature}
                    >
                      {isTop ? "★ " : "  "}
                      {feature}
                    </span>
                    {/* Bar */}
                    <div className="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          isTop ? "bg-blue-500" : "bg-gray-500"
                        }`}
                        style={{ width: `${barPct.toFixed(1)}%` }}
                      />
                    </div>
                    {/* Value */}
                    <span className="w-14 text-xs font-mono text-right text-gray-400 shrink-0">
                      {(importance * 100).toFixed(2)}%
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-gray-600 mt-4">
              ★ highlighted = top-k features (pruning candidates). Blue bar = included if pruning enabled.
            </p>
          </div>

          {/* Feature columns list */}
          {data?.feature_cols && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Active Feature Columns ({data.feature_cols.length})
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.feature_cols.map((col) => (
                  <span
                    key={col}
                    className={`px-2 py-1 rounded text-xs font-mono ${
                      topSet.has(col)
                        ? "bg-blue-900/50 text-blue-300 border border-blue-700"
                        : "bg-gray-700 text-gray-400"
                    }`}
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
