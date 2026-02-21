// ============================================================
// PageSkeleton.tsx - Reusable loading skeletons for pages
// ============================================================

const pulse = "bg-gray-800 rounded-lg border border-gray-700 animate-pulse";

function SkeletonBlock({ className }: { className?: string }) {
  return <div className={`${pulse} ${className || ""}`} />;
}

export function MetricsSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {[...Array(count)].map((_, i) => (
        <SkeletonBlock key={i} className="h-24" />
      ))}
    </div>
  );
}

export function ChartSkeleton({ height = "h-64" }: { height?: string }) {
  return <SkeletonBlock className={height} />;
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      <SkeletonBlock className="h-10" />
      {[...Array(rows)].map((_, i) => (
        <SkeletonBlock key={i} className="h-12" />
      ))}
    </div>
  );
}

export function CardListSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-3">
      {[...Array(count)].map((_, i) => (
        <SkeletonBlock key={i} className="h-32" />
      ))}
    </div>
  );
}

export default function PageSkeleton({ variant = "default" }: {
  variant?: "default" | "table" | "chart" | "cards";
}) {
  switch (variant) {
    case "table":
      return (
        <div className="space-y-4">
          <MetricsSkeleton count={2} />
          <TableSkeleton />
        </div>
      );
    case "chart":
      return (
        <div className="space-y-4">
          <ChartSkeleton height="h-96" />
        </div>
      );
    case "cards":
      return (
        <div className="space-y-4">
          <MetricsSkeleton />
          <CardListSkeleton />
        </div>
      );
    default:
      return (
        <div className="space-y-4">
          <MetricsSkeleton />
          <ChartSkeleton />
          <TableSkeleton />
        </div>
      );
  }
}
