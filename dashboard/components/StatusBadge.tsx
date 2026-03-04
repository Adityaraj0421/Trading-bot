// ============================================================
// StatusBadge.tsx - Shows the autonomous system state as a pill
// ============================================================

interface StatusBadgeProps {
  state: string;
}

const stateStyles: Record<string, string> = {
  normal: "bg-emerald-950 text-emerald-300 border-emerald-800",
  cautious: "bg-amber-950 text-amber-300 border-amber-800",
  defensive: "bg-orange-950 text-orange-300 border-orange-800",
  halted: "bg-rose-950 text-rose-300 border-rose-800",
};

export default function StatusBadge({ state }: StatusBadgeProps) {
  const style =
    stateStyles[state] || "bg-slate-800 text-slate-300 border-slate-700";

  return (
    <span
      className={`px-2 py-1 rounded-full text-xs font-medium border ${style}`}
    >
      {state.toUpperCase()}
    </span>
  );
}
