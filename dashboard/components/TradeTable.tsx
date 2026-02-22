// ============================================================
// TradeTable.tsx - Table showing completed trades (v7.0)
// ============================================================
// Displays trade history with symbol, entry/exit prices, PnL,
// strategy, and other details. Trades are shown newest-first.

interface Trade {
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  pnl_net: number;
  fees_paid: number;
  exit_reason: string;
  strategy?: string;
  strategy_name?: string;
  hold_bars: number;
  exit_time: string;
  entry_time: string;
}

// Symbol → short color-coded badge
const pairColors: Record<string, string> = {
  "BTC/USDT": "bg-orange-900/50 text-orange-400 border-orange-700",
  "ETH/USDT": "bg-blue-900/50 text-blue-400 border-blue-700",
  "SOL/USDT": "bg-purple-900/50 text-purple-400 border-purple-700",
};

export default function TradeTable({ trades }: { trades: Trade[] }) {
  if (trades.length === 0) {
    return <div className="text-center py-8 text-gray-500">No trades yet.</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left py-2 px-3">Time</th>
            <th className="text-left py-2 px-3">Pair</th>
            <th className="text-left py-2 px-3">Side</th>
            <th className="text-right py-2 px-3">Entry</th>
            <th className="text-right py-2 px-3">Exit</th>
            <th className="text-right py-2 px-3">PnL</th>
            <th className="text-left py-2 px-3">Strategy</th>
            <th className="text-left py-2 px-3">Reason</th>
            <th className="text-right py-2 px-3">Bars</th>
          </tr>
        </thead>

        <tbody>
          {trades
            .slice()
            .reverse()
            .map((t, i) => {
              const pair = t.symbol || "BTC/USDT";
              const colorClass = pairColors[pair] || "bg-gray-700 text-gray-300 border-gray-600";
              return (
                <tr
                  key={i}
                  className="border-b border-gray-800 hover:bg-gray-800/50"
                >
                  <td className="py-2 px-3 text-gray-400 whitespace-nowrap">
                    {t.exit_time
                      ? new Date(t.exit_time).toLocaleString()
                      : t.entry_time
                        ? new Date(t.entry_time).toLocaleString()
                        : "-"}
                  </td>

                  {/* Pair badge */}
                  <td className="py-2 px-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium border ${colorClass}`}>
                      {pair.split("/")[0]}
                    </span>
                  </td>

                  <td
                    className={`py-2 px-3 font-medium ${
                      t.side === "long" ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {t.side.toUpperCase()}
                  </td>

                  <td className="py-2 px-3 text-right font-mono">
                    ${t.entry_price?.toLocaleString()}
                  </td>

                  <td className="py-2 px-3 text-right font-mono">
                    {t.exit_price ? `$${t.exit_price.toLocaleString()}` : "-"}
                  </td>

                  <td
                    className={`py-2 px-3 text-right font-medium font-mono ${
                      (t.pnl_net || 0) >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {t.pnl_net != null ? `$${t.pnl_net.toFixed(2)}` : "-"}
                  </td>

                  <td className="py-2 px-3 text-gray-300 text-xs max-w-[120px] truncate">
                    {t.strategy_name ?? t.strategy}
                  </td>

                  <td className="py-2 px-3 text-gray-400 text-xs">
                    {t.exit_reason}
                  </td>

                  <td className="py-2 px-3 text-right text-gray-400">
                    {t.hold_bars}
                  </td>
                </tr>
              );
            })}
        </tbody>
      </table>
    </div>
  );
}
