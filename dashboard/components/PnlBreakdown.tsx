// ============================================================
// PnlBreakdown.tsx - PnL summary by pair and strategy (v7.0)
// ============================================================

"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
} from "recharts";

interface PnlGroup {
  trades: number;
  pnl: number;
  wins: number;
  losses: number;
}

interface PnlSummary {
  by_pair: Record<string, PnlGroup>;
  by_strategy: Record<string, PnlGroup>;
  cumulative_pnl: { pnl: number; timestamp: string; symbol: string }[];
  total_closed: number;
}

export default function PnlBreakdown({ data }: { data: PnlSummary | null }) {
  if (!data || data.total_closed === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No closed trades yet — PnL breakdown will appear here.
      </div>
    );
  }

  const pairData = Object.entries(data.by_pair).map(([name, g]) => ({
    name: name.split("/")[0],
    pnl: parseFloat(g.pnl.toFixed(2)),
    trades: g.trades,
    winRate: g.trades > 0 ? ((g.wins / g.trades) * 100).toFixed(0) : "0",
  }));

  const stratData = Object.entries(data.by_strategy)
    .sort((a, b) => b[1].trades - a[1].trades)
    .slice(0, 8)
    .map(([name, g]) => ({
      name: name.length > 15 ? name.slice(0, 15) + "…" : name,
      pnl: parseFloat(g.pnl.toFixed(2)),
      trades: g.trades,
      winRate: g.trades > 0 ? ((g.wins / g.trades) * 100).toFixed(0) : "0",
    }));

  const cumPnl = data.cumulative_pnl.map((d) => ({
    ...d,
    time: d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : "",
  }));

  return (
    <div className="space-y-6">
      {/* Cumulative PnL Curve */}
      {cumPnl.length > 1 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Cumulative PnL
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={cumPnl}>
              <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 10 }} />
              <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                }}
              />
              <Line
                type="monotone"
                dataKey="pnl"
                stroke={
                  cumPnl.length > 0 && cumPnl[cumPnl.length - 1].pnl >= 0
                    ? "#10b981"
                    : "#ef4444"
                }
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* PnL by Pair */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            PnL by Pair
          </h3>
          {pairData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={pairData}>
                  <XAxis dataKey="name" stroke="#6b7280" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1f2937",
                      border: "1px solid #374151",
                    }}
                  />
                  <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
                    {pairData.map((d, i) => (
                      <Cell
                        key={i}
                        fill={d.pnl >= 0 ? "#10b981" : "#ef4444"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-2 space-y-1">
                {pairData.map((d) => (
                  <div
                    key={d.name}
                    className="flex justify-between text-xs text-gray-400"
                  >
                    <span>{d.name}</span>
                    <span>
                      {d.trades} trades · {d.winRate}% WR ·{" "}
                      <span
                        className={
                          d.pnl >= 0 ? "text-green-400" : "text-red-400"
                        }
                      >
                        ${d.pnl}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-gray-500 text-sm">No pair data</p>
          )}
        </div>

        {/* PnL by Strategy */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            PnL by Strategy
          </h3>
          {stratData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={stratData} layout="vertical">
                  <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 10 }} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    stroke="#6b7280"
                    tick={{ fontSize: 10 }}
                    width={100}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1f2937",
                      border: "1px solid #374151",
                    }}
                  />
                  <Bar dataKey="pnl" radius={[0, 4, 4, 0]}>
                    {stratData.map((d, i) => (
                      <Cell
                        key={i}
                        fill={d.pnl >= 0 ? "#10b981" : "#ef4444"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-2 space-y-1">
                {stratData.map((d) => (
                  <div
                    key={d.name}
                    className="flex justify-between text-xs text-gray-400"
                  >
                    <span>{d.name}</span>
                    <span>
                      {d.trades} trades · {d.winRate}% WR
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-gray-500 text-sm">No strategy data</p>
          )}
        </div>
      </div>
    </div>
  );
}
