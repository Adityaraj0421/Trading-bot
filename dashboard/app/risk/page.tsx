"use client";

import { useState } from "react";
import useSWR from "swr";
import ErrorBanner from "@/components/ErrorBanner";
import { MetricsSkeleton, CardListSkeleton } from "@/components/PageSkeleton";
import api from "@/lib/api";

export default function RiskPage() {
  const { data: simulation, error: simErr, isLoading, mutate } = useSWR(
    "/risk/simulation", () => api.getRiskSimulation(),
    { refreshInterval: 30000 }
  );
  const { data: stressTests, error: stressErr } = useSWR(
    "/risk/stress-tests", () => api.getStressTests()
  );

  const [running, setRunning] = useState(false);
  const [nSims, setNSims] = useState(10000);
  const [nDays, setNDays] = useState(30);

  const error = simErr || stressErr;

  const runSimulation = async () => {
    setRunning(true);
    try {
      await api.runMonteCarlo({ n_simulations: nSims, n_days: nDays });
      setTimeout(() => mutate(), 5000);
    } finally {
      setRunning(false);
    }
  };

  const isNotRun = simulation?.status === "not_run";
  const isCompleted = simulation?.status === "completed";
  const mc = simulation?.monte_carlo || {};
  const stressResults = simulation?.stress_tests || [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Risk Simulation</h1>

      <ErrorBanner error={error} onRetry={() => mutate()} />

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 flex gap-4 items-end flex-wrap">
        <div>
          <label className="block text-sm text-gray-400 mb-1">Simulations</label>
          <input
            type="number"
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm w-28 font-mono"
            value={nSims}
            onChange={(e) => setNSims(Number(e.target.value))}
            min={100}
            max={100000}
          />
        </div>
        <div>
          <label className="block text-sm text-gray-400 mb-1">Days</label>
          <input
            type="number"
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm w-20 font-mono"
            value={nDays}
            onChange={(e) => setNDays(Number(e.target.value))}
            min={1}
            max={365}
          />
        </div>
        <button
          onClick={runSimulation}
          disabled={running}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded text-sm font-medium"
        >
          {running ? "Running..." : "Run Monte Carlo"}
        </button>
      </div>

      {isLoading && !simulation ? (
        <div className="space-y-4">
          <MetricsSkeleton />
          <CardListSkeleton />
        </div>
      ) : isCompleted && mc ? (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Monte Carlo Results</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="VaR (95%)" value={formatDollar(mc.var_95)} color="red" />
            <StatCard label="VaR (99%)" value={formatDollar(mc.var_99)} color="red" />
            <StatCard label="CVaR (95%)" value={formatDollar(mc.cvar_95)} color="red" />
            <StatCard label="Prob. of Ruin" value={formatPct(mc.probability_of_ruin)} color="red" />
            <StatCard label="Median Equity" value={formatDollar(mc.median_final_equity)} color="green" />
            <StatCard label="Max DD (95th)" value={formatPct(mc.max_drawdown_95th)} color="red" />
            <StatCard label="Simulations" value={mc.n_simulations?.toLocaleString() || "—"} />
            <StatCard label="Days Simulated" value={mc.n_days?.toString() || "—"} />
          </div>

          {/* Percentile distribution */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">Equity Percentiles</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
              {[
                { label: "Worst 5%", val: mc.percentile_5 },
                { label: "25th", val: mc.percentile_25 },
                { label: "Median", val: mc.median_final_equity },
                { label: "75th", val: mc.percentile_75 },
                { label: "Best 5%", val: mc.percentile_95 },
              ].map(({ label, val }) => (
                <div key={label} className="text-center">
                  <div className="text-gray-400">{label}</div>
                  <div className={`font-mono ${(val || 0) >= (mc.initial_equity || 0) ? "text-green-400" : "text-red-400"}`}>
                    {formatDollar(val)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Interpretation */}
          {mc.interpretation && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-2">
              <h3 className="text-sm font-semibold text-gray-300 mb-2">Plain English</h3>
              {Object.entries(mc.interpretation as Record<string, string>)
                .filter(([k]) => k !== "risk_level")
                .map(([key, val]) => (
                  <p key={key} className="text-sm text-gray-400">{val}</p>
                ))
              }
            </div>
          )}
        </div>
      ) : (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center text-gray-500">
          <p>{isNotRun ? "No simulation results yet" : "Loading simulation..."}</p>
          <p className="text-sm mt-2">
            Run a Monte Carlo simulation to see VaR, CVaR, and return distributions
          </p>
        </div>
      )}

      {/* Stress Test Results */}
      {stressResults.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Stress Test Results</h2>
          {stressResults.map((s: any, i: number) => (
            <div key={i} className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium">{s.scenario || s.name}</span>
                <span className={`text-sm font-mono ${(s.impact || s.pnl || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {formatPct(s.impact || s.pnl)}
                </span>
              </div>
              {s.description && (
                <p className="text-sm text-gray-400">{s.description}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Available Scenarios */}
      {stressTests?.scenarios && stressTests.scenarios.length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Available Stress Scenarios</h3>
          <div className="flex flex-wrap gap-2">
            {stressTests.scenarios.map((s: any, i: number) => (
              <span key={i} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300">
                {typeof s === "string" ? s : s.name || s.scenario}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color?: string }) {
  const colorClass = color === "red" ? "text-red-400" : color === "green" ? "text-green-400" : "text-white";
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-lg font-mono ${colorClass}`}>{value}</div>
    </div>
  );
}

function formatPct(val: number | undefined | null): string {
  if (val === undefined || val === null) return "—";
  return `${(val * 100).toFixed(2)}%`;
}

function formatDollar(val: number | undefined | null): string {
  if (val === undefined || val === null) return "—";
  return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
