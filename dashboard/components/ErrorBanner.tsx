// ============================================================
// ErrorBanner.tsx - Reusable API error display with retry
// ============================================================

"use client";

interface ErrorBannerProps {
  error: Error | undefined;
  onRetry?: () => void;
}

export default function ErrorBanner({ error, onRetry }: ErrorBannerProps) {
  if (!error) return null;

  return (
    <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 flex items-center justify-between">
      <div className="text-sm text-red-300">
        Failed to fetch data from API. Retrying...
        <span className="text-red-500 ml-2 text-xs">{error.message}</span>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="text-xs text-red-400 hover:text-red-200 border border-red-700 px-3 py-1 rounded transition-colors"
        >
          Retry Now
        </button>
      )}
    </div>
  );
}
