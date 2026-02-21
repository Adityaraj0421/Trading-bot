// ============================================================
// EquityChart.tsx - Area chart showing portfolio equity over time
// ============================================================
// Uses the "recharts" library to render a responsive area chart.
// The "use client" directive is required because recharts uses
// browser APIs (DOM) that don't work during server-side rendering.

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
  height?: number; // Chart height in pixels (default: 300)
}

export default function EquityChart({ data, height = 300 }: EquityChartProps) {
  // Show a placeholder if there's no data yet
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-800 rounded-lg border border-gray-700">
        <p className="text-gray-500">
          No equity data yet. Waiting for agent cycles...
        </p>
      </div>
    );
  }

  // Convert ISO timestamps to compact HH:MM format for the X axis
  const formatted = data.map((d) => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-2 sm:p-4">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={formatted} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>

          {/* X axis — preserveStartEnd prevents label overlap on mobile */}
          <XAxis
            dataKey="time"
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
            minTickGap={40}
          />

          {/* Y axis — compact $ format, narrower width for mobile */}
          <YAxis
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            width={55}
            tickFormatter={(v: number) =>
              v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v}`
            }
          />

          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
              fontSize: "0.75rem",
            }}
            labelStyle={{ color: "#9ca3af" }}
            formatter={(value: number | string | undefined) => [`$${Number(value ?? 0).toLocaleString()}`, "Equity"]}
          />

          <Area
            type="monotone"
            dataKey="equity"
            stroke="#3b82f6"
            fill="url(#equityGrad)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
