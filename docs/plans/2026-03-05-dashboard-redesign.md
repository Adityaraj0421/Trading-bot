# Dashboard Full Visual Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the crypto trading dashboard with a teal/violet crypto-native aesthetic, better empty states, Phase 9 decision panel, and clear all stale data.

**Architecture:** Modify 7 existing React/TSX files (no new dependencies, no API changes). Clear 6 data files so agent restarts fresh. All styling via Tailwind CSS utility classes — no new CSS files needed beyond globals.css tweak.

**Tech Stack:** Next.js 15, React, Tailwind CSS v4, Recharts, SWR

---

### Task 1: Clear all stale data

**Files:**
- Delete: `data/trades.db`
- Delete: `data/agent_state.json`
- Delete: `data/agent_state_autonomous.json`
- Delete: `data/phase9_decisions.jsonl`
- Delete: `data/agent.log`
- Delete: `data/equity.json` (if exists)

**Step 1: Stop the API server**
Run: `pkill -f "uvicorn api.server"` (or use preview_stop)

**Step 2: Delete data files**
```bash
rm -f data/trades.db data/agent_state.json data/agent_state_autonomous.json \
       data/phase9_decisions.jsonl data/agent.log data/equity.json \
       data/agent_state_model.pkl
```

**Step 3: Verify clean**
```bash
ls data/
```
Expected: empty or only `data/` directory exists

---

### Task 2: Update globals.css and layout.tsx

**Files:**
- Modify: `dashboard/app/globals.css`
- Modify: `dashboard/app/layout.tsx`

**Step 1: Update globals.css**
Replace entire file with:
```css
@import "tailwindcss";

/* Custom scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #020617; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
```

**Step 2: Update layout.tsx**
- Change title: `"Crypto Trading Agent v7.0"` → `"Crypto Agent — Phase 9"`
- Change body class: `"bg-gray-950"` → `"bg-slate-950"`

---

### Task 3: Redesign Sidebar.tsx

**Files:**
- Modify: `dashboard/components/Sidebar.tsx`

**Step 1: Replace entire file**
New nav items use emoji icons. Active state: teal left border + teal text. Brand is teal gradient. Footer shows "Phase 9" tag.

```tsx
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
  { href: "/risk", label: "Risk Sim", icon: "🛡" },
  { href: "/model", label: "Model", icon: "🧩" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { data: status } = useSWR("/status-sidebar", () => api.getStatus(), { refreshInterval: 10000 });
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
          <span className={`w-1.5 h-1.5 rounded-full ${mode === "live" ? "bg-emerald-400" : "bg-amber-400"}`} />
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
```

---

### Task 4: Redesign MetricCard.tsx

**Files:**
- Modify: `dashboard/components/MetricCard.tsx`

**Step 1: Replace entire file**
Adds glow support, teal for "blue" color, emerald/rose instead of green/red.

```tsx
interface MetricCardProps {
  label: string;
  value: string;
  subtext?: string;
  color?: "green" | "red" | "blue" | "yellow" | "violet" | "default";
  glow?: boolean;
}

const colorMap: Record<string, string> = {
  green: "text-emerald-400",
  red: "text-rose-400",
  blue: "text-cyan-400",
  yellow: "text-amber-400",
  violet: "text-violet-400",
  default: "text-white",
};

const glowMap: Record<string, string> = {
  green: "drop-shadow-[0_0_8px_rgba(52,211,153,0.6)]",
  red: "drop-shadow-[0_0_8px_rgba(251,113,133,0.6)]",
  blue: "drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]",
  yellow: "drop-shadow-[0_0_8px_rgba(251,191,36,0.5)]",
  violet: "drop-shadow-[0_0_8px_rgba(167,139,250,0.5)]",
  default: "",
};

const borderMap: Record<string, string> = {
  blue: "border-cyan-800/60",
  green: "border-emerald-900/50",
  red: "border-rose-900/50",
  yellow: "border-amber-900/50",
  violet: "border-violet-900/50",
  default: "border-white/5",
};

export default function MetricCard({
  label,
  value,
  subtext,
  color = "default",
  glow = false,
}: MetricCardProps) {
  const shouldGlow = glow || color !== "default";
  return (
    <div className={`bg-slate-900 border rounded-xl p-4 ${borderMap[color] || "border-white/5"}`}>
      <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">{label}</p>
      <p className={`text-2xl font-bold mt-1 font-mono ${colorMap[color]} ${shouldGlow ? glowMap[color] : ""}`}>
        {value}
      </p>
      {subtext && <p className="text-xs text-slate-600 mt-1">{subtext}</p>}
    </div>
  );
}
```

---

### Task 5: Update ClientLayout.tsx

**Files:**
- Modify: `dashboard/components/ClientLayout.tsx`

**Step 1: Update sidebar and main colors**
Change `bg-gray-900 border-r border-gray-800` → `bg-slate-900 border-r border-white/5`
Change `bg-gray-800` → `bg-slate-800`

Full replacement:
```tsx
"use client";

import { useAgentWebSocket } from "@/lib/useWebSocket";
import Sidebar from "./Sidebar";
import ConnectionStatus from "./ConnectionStatus";

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const { connectionState } = useAgentWebSocket();

  return (
    <div className="flex">
      <aside className="w-56 bg-slate-900 border-r border-white/5 min-h-screen flex flex-col">
        <Sidebar />
        <div className="border-t border-white/5">
          <ConnectionStatus state={connectionState} />
        </div>
      </aside>
      <main className="flex-1 p-6 overflow-auto min-h-screen">{children}</main>
    </div>
  );
}
```

---

### Task 6: Update EquityChart.tsx

**Files:**
- Modify: `dashboard/components/EquityChart.tsx`

**Step 1: Better empty state + teal colors**
- Empty state: animated pulse line visual + better text
- Chart: cyan instead of blue

Replace `data.length === 0` block and gradient/stroke colors:
- `stopColor="#3b82f6"` → `stopColor="#22d3ee"` (both stops)
- `stroke="#3b82f6"` (Area) → `stroke="#22d3ee"`
- Empty state: styled with slate-900 bg + animated pulse dot

---

### Task 7: Update TradeTable.tsx empty state

**Files:**
- Modify: `dashboard/components/TradeTable.tsx`

**Step 1: Better empty state, hover row**
- Empty state text: `"No trades yet — Phase 9 is watching the market"`
- Row hover: `hover:bg-slate-800/40` instead of gray
- Header border: `border-slate-700`
- Body border: `border-slate-800`

---

### Task 8: Update StatusBadge.tsx

**Files:**
- Modify: `dashboard/components/StatusBadge.tsx`

**Step 1: Match new palette**
- normal: `bg-emerald-950 text-emerald-300 border-emerald-800`
- cautious: `bg-amber-950 text-amber-300 border-amber-800`
- defensive: `bg-orange-950 text-orange-300 border-orange-800`
- halted: `bg-rose-950 text-rose-300 border-rose-800`

---

### Task 9: Update Overview page (app/page.tsx)

**Files:**
- Modify: `dashboard/app/page.tsx`

**Step 1: Add Phase 9 decision panel**
After the metrics grid, add a violet-bordered panel that reads from `/autonomous/events` API (already available via `api.getAutonomousEvents()`). Shows last event: type, description, timestamp.

**Step 2: Win Rate shows "—" when 0 trades**
```tsx
value={s.total_trades ? `${((s.win_rate || 0) * 100).toFixed(1)}%` : "—"}
```

**Step 3: Better System Modules section**
Add Phase 9 indicator chip alongside existing ones.

---

### Task 10: Restart servers and verify

**Step 1: Restart API (clean start)**
```bash
preview_start api
```

**Step 2: Restart dashboard**
```bash
preview_start dashboard
```
Or trigger hot-reload by saving a file.

**Step 3: Take screenshot and verify**
- Sidebar shows emoji icons + teal active state
- MetricCards show slate-900 bg + colored borders
- PnL shows $0.00 (fresh data)
- Win Rate shows "—" (no trades yet)
- Equity chart shows empty state

**Step 4: Commit**
```bash
git add dashboard/ docs/
git commit -m "feat: full dashboard visual redesign (Phase 9 aesthetic)"
```
