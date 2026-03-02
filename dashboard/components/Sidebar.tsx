// ============================================================
// Sidebar.tsx - Navigation sidebar for the dashboard
// ============================================================
// Shows a list of pages the user can navigate to. The current
// page is highlighted. Uses Next.js <Link> for client-side
// navigation (no full page reload).

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// All the pages in the dashboard
const navItems = [
  { href: "/", label: "Overview", icon: "O" },
  { href: "/equity", label: "Equity Curve", icon: "E" },
  { href: "/trades", label: "Trade History", icon: "T" },
  { href: "/autonomous", label: "Autonomous", icon: "A" },
  { href: "/notifications", label: "Notifications", icon: "N" },
  { href: "/backtest", label: "Backtesting", icon: "B" },
  { href: "/intelligence", label: "Intelligence", icon: "I" },
  { href: "/arbitrage", label: "Arbitrage", icon: "X" },
  { href: "/risk", label: "Risk Sim", icon: "R" },
  { href: "/model", label: "Model", icon: "M" },
];

export default function Sidebar() {
  // usePathname() returns the current URL path (e.g., "/trades")
  const pathname = usePathname();

  return (
    <div className="p-4 flex-1">
      {/* App title */}
      <h1 className="text-lg font-bold text-blue-400 mb-6">Crypto Agent v7</h1>

      {/* Navigation links */}
      <nav className="space-y-1">
        {navItems.map((item) => {
          // Check if this link is the currently active page
          const active = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-blue-900/50 text-blue-300 border border-blue-800"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              }`}
            >
              {/* Icon letter in a small circle */}
              <span className="w-5 h-5 flex items-center justify-center rounded bg-gray-700 text-xs font-bold">
                {item.icon}
              </span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
