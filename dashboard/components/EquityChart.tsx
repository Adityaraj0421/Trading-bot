// ============================================================
// EquityChart.tsx - Area chart showing portfolio equity over time
// ============================================================

"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface EquityChartProps {
  data: { equity: number; timestamp: string }[];
  height?: number;
}

export default function EquityChart({ data, height = 300 }: EquityChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 bg-slate-900 rounded-xl border border-white/5">
        <div className="w-8 h-8 rounded-full border-2 border-cyan-500/30 border-t-cyan-400 animate-spin mb-3" />
        <p className="text-slate-500 text-sm">Waiting for first trade…</p>
        <p className="text-slate-700 text-xs mt-1">Phase 9 is watching the market</p>
      </div>
    );
  }

  const formatted = data.map((d) => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  return (
    <div className="bg-slate-900 rounded-xl border border-white/5 p-2 sm:p-4">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={formatted} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="time"
            stroke="#334155"
            tick={{ fontSize: 10, fill: "#64748b" }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            stroke="#334155"
            tick={{ fontSize: 10, fill: "#64748b" }}
            width={55}
            tickFormatter={(v: number) =>
              v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v}`
            }
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              border: "1px solid rgba(255,255,255,0.05)",
              borderRadius: "0.75rem",
              fontSize: "0.75rem",
            }}
            labelStyle={{ color: "#64748b" }}
            formatter={(value: number | string | undefined) => [
              `$${Number(value ?? 0).toLocaleString()}`,
              "Equity",
            ]}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#22d3ee"
            fill="url(#equityGrad)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
