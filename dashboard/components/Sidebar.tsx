"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import api from "@/lib/api";

const navItems = [
  { href: "/", label: "Overview", icon: "⬡" },
  { href: "/equity", label: "Equity Curve", icon: "📈" },
  { href: "/trades", label: "Trade History", icon: "🔁" },
  { href: "/autonomous", label: "Autonomous", icon: "🤖" },
  { href: "/notifications", label: "Notifications", icon: "🔔" },
  { href: "/backtest", label: "Backtesting", icon: "🧪" },
  { href: "/intelligence", label: "Intelligence", icon: "🧠" },
  { href: "/arbitrage", label: "Arbitrage", icon: "⚡" },
  { href: "/risk", label: "Risk Sim", icon: "🛡️" },
  { href: "/model", label: "Model", icon: "🧩" },
  { href: "/decisions", label: "Decisions", icon: "🔍" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { data: status } = useSWR(
    "/status-sidebar",
    () => api.getStatus(),
    { refreshInterval: 10000 }
  );
  const s = (status as Record<string, any>) || {};
  const mode = s.trading_mode || "paper";
  const cycle = s.cycle;

  return (
    <div className="p-4 flex-1 flex flex-col">
      {/* Brand */}
      <div className="mb-6">
        <h1 className="text-base font-bold bg-gradient-to-r from-cyan-400 to-teal-400 bg-clip-text text-transparent tracking-wide">
          ◈ CRYPTO AGENT
        </h1>
        <div className="flex items-center gap-1.5 mt-1">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              mode === "live" ? "bg-emerald-400" : "bg-amber-400"
            }`}
          />
          <span className="text-xs text-slate-400 capitalize">{mode} mode</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="space-y-0.5 flex-1">
        {navItems.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all ${
                active
                  ? "bg-cyan-950/60 text-cyan-300 border-l-2 border-cyan-400"
                  : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 border-l-2 border-transparent"
              }`}
            >
              <span className="text-base leading-none">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      {cycle && (
        <div className="mt-4 pt-3 border-t border-slate-800">
          <div className="flex items-center gap-1.5 px-1">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs text-slate-500">
              Cycle #{cycle} · Phase 9
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
