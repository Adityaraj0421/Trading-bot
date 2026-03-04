// ============================================================
// MetricCard.tsx - A card displaying a single metric
// ============================================================
// Supports teal/emerald/rose color scheme with optional glow.

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
    <div
      className={`bg-slate-900 border rounded-xl p-4 ${
        borderMap[color] || "border-white/5"
      }`}
    >
      <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">
        {label}
      </p>
      <p
        className={`text-2xl font-bold mt-1 font-mono ${colorMap[color]} ${
          shouldGlow ? glowMap[color] : ""
        }`}
      >
        {value}
      </p>
      {subtext && <p className="text-xs text-slate-600 mt-1">{subtext}</p>}
    </div>
  );
}
