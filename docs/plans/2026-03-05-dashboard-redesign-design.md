# Dashboard Full Visual Redesign — Design Doc
**Date**: 2026-03-05
**Status**: Approved
**Scope**: Full visual overhaul of the Next.js dashboard + data reset for fresh start

---

## Goals
1. Replace dated "Crypto Agent v7" aesthetic with a modern, crypto-native design
2. Show clean empty states on fresh start (no confusing zeros)
3. Surface Phase 9 decision data on the Overview page (it's the primary engine)
4. Update branding to reflect current version (Phase 9/10)

---

## Visual Language

### Color Palette
| Token | Value | Use |
|-------|-------|-----|
| Background | `slate-950` / `slate-900` | Page + card backgrounds |
| Primary | `cyan-400` / `teal-500` | Active nav, key metrics, borders |
| AI/Phase9 | `violet-500` / `purple-400` | Phase 9 panel, Autonomous page |
| Gain | `emerald-400` | Positive PnL, wins |
| Loss | `rose-400` | Negative PnL, losses |
| Warning | `amber-400` | Regime chip (high_volatility, etc.) |
| Border | `white/5` to `white/10` | Glass-morphism card borders |

### Effects
- Glow on PnL number: green glow when positive, red glow when negative
- Glowing teal left-bar on active nav item
- Gradient text on brand name
- Gradient border on Phase 9 panel

---

## Components Changed

### Sidebar (`Sidebar.tsx`)
- Brand: "◈ CRYPTO AGENT" with teal gradient text (replace "Crypto Agent v7")
- Mode badge: "Paper" or "Live" pill below brand name
- Nav icons: emoji icons replacing letter circles
- Active state: teal left border + teal text + subtle bg
- Footer: last cycle heartbeat ("Cycle #N · Xm ago")

### MetricCard (`MetricCard.tsx`)
- Gradient border on "Capital" card (teal)
- Trend arrow on PnL card (↑ / ↓)
- Show "—" instead of "0.0%" for Win Rate when no trades
- Subtle glow on value text based on color prop

### Overview page (`app/page.tsx`)
- New **Phase 9 Last Decision** panel (violet border) showing:
  - Symbol, action, reason, swing bias, score, time ago
  - Reads from `/autonomous/events` API
- Better **System Modules** chips (add Phase 9 + WS Feed indicators)
- Empty state for equity curve: animated pulse + "Waiting for first trade…"
- Empty state for trades table: centered message "No trades yet — Phase 9 is watching the market"

### Global layout (`app/globals.css` / `layout.tsx`)
- Background: `bg-slate-950`
- Sidebar: `bg-slate-900` with `border-r border-white/5`
- Scrollbar styling

---

## Data Reset Plan
Delete the following files to get a clean slate:
- `data/trades.db` — SQLite trade history
- `data/agent_state.json` — capital / PnL counters
- `data/agent_state_autonomous.json` — autonomous event log
- `data/phase9_decisions.jsonl` — Phase 9 decision audit log
- `data/agent.log` — agent cycle log
- `data/equity.json` (if present) — equity snapshots

Agent must be restarted after data clear so it re-initializes from `INITIAL_CAPITAL` in `.env`.

---

## Out of Scope
- Chart library changes (Recharts stays as-is)
- API changes
- Mobile responsive overhaul
- Dark/light mode toggle
