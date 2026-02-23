"use client";

import useSWR from "swr";
import ErrorBanner from "@/components/ErrorBanner";
import { MetricsSkeleton, CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function IntelligencePage() {
  const { data: signals, error: signalsErr, isLoading, mutate: mutateSignals } = useSWR(
    "/intelligence/signals", () => api.getIntelligence(),
    { refreshInterval: 30000 }
  );
  const { data: providers, error: providersErr, mutate: mutateProviders } = useSWR(
    "/intelligence/providers", () => api.getIntelligenceProviders()
  );

  const error = signalsErr || providersErr;
  const isDisabled = signals?.status === "not_enabled";
  const isWaiting = signals?.status === "awaiting_first_cycle";
  const hasData = signals?.signals && Array.isArray(signals.signals);

  const biasColor: Record<string, string> = {
    bullish: "text-green-400",
    bearish: "text-red-400",
    neutral: "text-gray-400",
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Intelligence Signals</h1>

      <ErrorBanner error={error} onRetry={() => { mutateSignals(); mutateProviders(); }} />

      {isLoading && !signals ? (
        <div className="space-y-4">
          <MetricsSkeleton count={3} />
          <CardListSkeleton count={4} />
        </div>
      ) : (
        <>
          {/* Provider status */}
          {providers?.providers && (
            <div className="flex gap-2 flex-wrap">
              {providers.providers.map((p: any) => (
                <span
                  key={p.name}
                  className={`px-3 py-1 rounded-full text-xs font-medium ${
                    p.enabled
                      ? "bg-green-900/50 text-green-400 border border-green-700"
                      : "bg-gray-800 text-gray-500 border border-gray-700"
                  }`}
                >
                  {p.name}: {p.enabled ? "ON" : "OFF"}
                </span>
              ))}
            </div>
          )}

          {isDisabled ? (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
              <p className="text-lg">Intelligence signals are disabled</p>
              <p className="text-sm mt-2">
                Enable providers in .env (e.g., ENABLE_NEWS_NLP=true)
              </p>
            </div>
          ) : isWaiting ? (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
              <p>Waiting for agent to complete first cycle...</p>
            </div>
          ) : hasData ? (
            <>
              {/* Aggregate summary */}
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
                <div className="flex items-center gap-6 flex-wrap">
                  <div>
                    <p className="text-sm text-gray-400">Overall Bias</p>
                    <p className={`text-xl font-bold ${biasColor[signals.bias] || "text-gray-400"}`}>
                      {signals.bias?.toUpperCase()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Adjustment Factor</p>
                    <p className="text-xl font-mono">{signals.adjustment_factor?.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Net Score</p>
                    <p className={`text-xl font-mono ${(signals.net_score || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {signals.net_score >= 0 ? "+" : ""}{signals.net_score?.toFixed(3)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Bullish</p>
                    <p className="text-xl font-mono text-green-400">
                      +{signals.bullish_score?.toFixed(3) || "0.000"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Bearish</p>
                    <p className="text-xl font-mono text-red-400">
                      -{signals.bearish_score?.toFixed(3) || "0.000"}
                    </p>
                  </div>
                </div>
              </div>

              {/* Individual signals with v7.0 expanded details */}
              <div className="space-y-3">
                {signals.signals.map((sig: any, i: number) => {
                  const hasError = sig.data?.error;
                  const dataKeys = sig.data ? Object.keys(sig.data).filter(k => k !== "error") : [];
                  return (
                    <div key={i} className={`bg-gray-800 rounded-lg border p-4 ${
                      hasError ? "border-red-800/50" : "border-gray-700"
                    }`}>
                      <div className="flex justify-between items-center">
                        <span className="font-medium">{sig.source}</span>
                        <div className="flex gap-3 items-center">
                          <span className={`text-sm ${biasColor[sig.signal] || "text-gray-400"}`}>
                            {sig.signal}
                          </span>
                          <span className="text-sm text-gray-400 font-mono">
                            str: {sig.strength?.toFixed(2)}
                          </span>
                        </div>
                      </div>
                      {hasError && (
                        <p className="text-xs text-red-400 mt-2 font-mono">
                          Error: {sig.data.error}
                        </p>
                      )}
                      {dataKeys.length > 0 && (
                        <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                          {dataKeys.slice(0, 8).map((key) => {
                            const val = sig.data[key];
                            let display: string;
                            if (val == null) {
                              display = "\u2014";
                            } else if (typeof val === "number") {
                              display = val.toLocaleString(undefined, { maximumFractionDigits: 4 });
                            } else if (typeof val === "boolean") {
                              display = val ? "true" : "false";
                            } else if (Array.isArray(val)) {
                              display = `[${val.length} items]`;
                            } else if (typeof val === "object") {
                              display = JSON.stringify(val).slice(0, 40);
                            } else {
                              display = String(val).slice(0, 30);
                            }
                            return (
                              <div key={key} className="bg-gray-900 rounded px-2 py-1">
                                <span className="text-gray-500">{key}: </span>
                                <span className="text-gray-300 font-mono">{display}</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
              <p>No intelligence data available</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
