// ============================================================
// MetricCard.tsx - A small card that shows a single metric
// ============================================================
// Used on the dashboard to display Capital, PnL, Win Rate, etc.
// Each card has a label, a big value, an optional subtext, and
// an optional color to highlight positive/negative numbers.

interface MetricCardProps {
  label: string;        // Small label at the top (e.g., "Capital")
  value: string;        // Big number (e.g., "$10,000")
  subtext?: string;     // Optional small text below the value
  color?: "green" | "red" | "blue" | "yellow" | "default";
}

// Map color names to Tailwind CSS classes
const colorMap: Record<string, string> = {
  green: "text-green-400",
  red: "text-red-400",
  blue: "text-blue-400",
  yellow: "text-yellow-400",
  default: "text-white",
};

export default function MetricCard({
  label,
  value,
  subtext,
  color = "default",
}: MetricCardProps) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      {/* Label */}
      <p className="text-xs text-gray-400 uppercase tracking-wide">{label}</p>

      {/* Main value - colored based on the "color" prop */}
      <p className={`text-2xl font-bold mt-1 ${colorMap[color]}`}>{value}</p>

      {/* Optional subtext */}
      {subtext && <p className="text-xs text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}
