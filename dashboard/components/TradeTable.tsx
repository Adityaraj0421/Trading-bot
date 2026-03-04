// ============================================================
// TradeTable.tsx - Table showing completed trades
// ============================================================

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

const pairColors: Record<string, string> = {
  "BTC/USDT": "bg-orange-950/50 text-orange-400 border-orange-800",
  "ETH/USDT": "bg-cyan-950/50 text-cyan-400 border-cyan-800",
  "SOL/USDT": "bg-violet-950/50 text-violet-400 border-violet-800",
};

export default function TradeTable({ trades }: { trades: Trade[] }) {
  if (trades.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <div className="text-3xl mb-3">🤖</div>
        <p className="text-slate-400 font-medium">No trades yet</p>
        <p className="text-slate-600 text-sm mt-1">
          Phase 9 is watching the market
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-500 border-b border-slate-800">
            <th className="text-left py-2 px-3 font-medium">Time</th>
            <th className="text-left py-2 px-3 font-medium">Pair</th>
            <th className="text-left py-2 px-3 font-medium">Side</th>
            <th className="text-right py-2 px-3 font-medium">Entry</th>
            <th className="text-right py-2 px-3 font-medium">Exit</th>
            <th className="text-right py-2 px-3 font-medium">PnL</th>
            <th className="text-left py-2 px-3 font-medium">Strategy</th>
            <th className="text-left py-2 px-3 font-medium">Reason</th>
            <th className="text-right py-2 px-3 font-medium">Bars</th>
          </tr>
        </thead>
        <tbody>
          {trades
            .slice()
            .reverse()
            .map((t, i) => {
              const pair = t.symbol || "BTC/USDT";
              const colorClass =
                pairColors[pair] ||
                "bg-slate-800 text-slate-300 border-slate-700";
              return (
                <tr
                  key={i}
                  className="border-b border-slate-800/60 hover:bg-slate-800/30 transition-colors"
                >
                  <td className="py-2 px-3 text-slate-500 whitespace-nowrap text-xs">
                    {t.exit_time
                      ? new Date(t.exit_time).toLocaleString()
                      : t.entry_time
                      ? new Date(t.entry_time).toLocaleString()
                      : "—"}
                  </td>
                  <td className="py-2 px-3">
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-medium border ${colorClass}`}
                    >
                      {pair.split("/")[0]}
                    </span>
                  </td>
                  <td
                    className={`py-2 px-3 font-medium text-xs ${
                      t.side === "long" ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {t.side.toUpperCase()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-slate-300">
                    ${t.entry_price?.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-slate-300">
                    {t.exit_price ? `$${t.exit_price.toLocaleString()}` : "—"}
                  </td>
                  <td
                    className={`py-2 px-3 text-right font-medium font-mono ${
                      (t.pnl_net || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {t.pnl_net != null ? `$${t.pnl_net.toFixed(2)}` : "—"}
                  </td>
                  <td className="py-2 px-3 text-slate-400 text-xs max-w-[120px] truncate">
                    {t.strategy_name ?? t.strategy}
                  </td>
                  <td className="py-2 px-3 text-slate-500 text-xs">
                    {t.exit_reason}
                  </td>
                  <td className="py-2 px-3 text-right text-slate-500 text-xs">
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
