// ============================================================
// StatusBadge.tsx - Shows the autonomous system state as a pill
// ============================================================
// The autonomous trading system has states like:
//   "normal"    -> green  (everything fine)
//   "cautious"  -> yellow (reducing risk)
//   "defensive" -> orange (high alert)
//   "halted"    -> red    (trading stopped)

interface StatusBadgeProps {
  state: string;
}

// Map each state to background, text, and border colors
const stateStyles: Record<string, string> = {
  normal: "bg-green-900 text-green-300 border-green-700",
  cautious: "bg-yellow-900 text-yellow-300 border-yellow-700",
  defensive: "bg-orange-900 text-orange-300 border-orange-700",
  halted: "bg-red-900 text-red-300 border-red-700",
};

export default function StatusBadge({ state }: StatusBadgeProps) {
  // Fall back to gray if the state is unknown
  const style =
    stateStyles[state] || "bg-gray-700 text-gray-300 border-gray-600";

  return (
    <span
      className={`px-2 py-1 rounded-full text-xs font-medium border ${style}`}
    >
      {state.toUpperCase()}
    </span>
  );
}
