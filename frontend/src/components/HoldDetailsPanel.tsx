/**
 * HoldDetailsPanel component
 *
 * Displays detailed statistics for a selected hold position.
 */

import type { HoldStats } from "../types/analytics";

interface HoldDetailsPanelProps {
  /** The hold position (e.g., "F7") */
  position: string;
  /** Statistics for this hold */
  stats: HoldStats;
}

/**
 * Get color class based on normalized ratio
 * >1 = over-represented (warmer colors), <1 = under-represented (cooler colors)
 */
function getNormalizedColor(ratio: number): string {
  if (ratio >= 2.0) return "bg-orange-500";
  if (ratio >= 1.5) return "bg-amber-500";
  if (ratio >= 1.2) return "bg-yellow-500";
  if (ratio >= 0.8) return "bg-blue-500";
  if (ratio >= 0.5) return "bg-sky-500";
  return "bg-cyan-500";
}

/**
 * Get badge style for normalized indicator
 */
function getNormalizedBadge(
  ratio: number
): { text: string; className: string } | null {
  if (ratio >= 2.0)
    return { text: `${ratio.toFixed(1)}×`, className: "bg-orange-500/20 text-orange-300" };
  if (ratio >= 1.5)
    return { text: `${ratio.toFixed(1)}×`, className: "bg-amber-500/20 text-amber-300" };
  if (ratio <= 0.5)
    return { text: `${ratio.toFixed(1)}×`, className: "bg-cyan-500/20 text-cyan-300" };
  return null;
}

/**
 * Simple bar chart for grade distribution with normalized indicators
 */
function GradeDistributionChart({
  distribution,
  distributionNormalized,
}: {
  distribution: Record<string, number>;
  distributionNormalized: Record<string, number>;
}) {
  // Sort grades by frequency and take top entries
  const sortedGrades = Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);

  const maxCount = Math.max(...sortedGrades.map(([, count]) => count));

  if (sortedGrades.length === 0) return null;

  return (
    <div className="space-y-1.5">
      {sortedGrades.map(([grade, count]) => {
        const normalizedRatio = distributionNormalized[grade] ?? 1;
        const barColor = getNormalizedColor(normalizedRatio);
        const badge = getNormalizedBadge(normalizedRatio);

        return (
          <div key={grade} className="flex items-center gap-2 text-sm">
            <span className="w-10 text-gray-400 font-mono">{grade}</span>
            <div className="flex-1 h-4 bg-gray-700 rounded overflow-hidden relative group">
              <div
                className={`h-full ${barColor} rounded transition-all duration-300`}
                style={{ width: `${(count / maxCount) * 100}%` }}
              />
              {/* Tooltip showing normalized ratio */}
              <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-xs text-white font-medium bg-gray-900/80 px-1.5 py-0.5 rounded">
                  {normalizedRatio.toFixed(1)}× vs dataset
                </span>
              </div>
            </div>
            <span className="w-10 text-right text-gray-400">{count}</span>
            {/* Badge for significant over/under-representation */}
            {badge ? (
              <span
                className={`text-xs px-1.5 py-0.5 rounded font-medium ${badge.className}`}
              >
                {badge.text}
              </span>
            ) : (
              <span className="w-10" /> // Spacer for alignment
            )}
          </div>
        );
      })}
      {/* Legend */}
      <div className="flex items-center gap-3 mt-3 pt-2 border-t border-gray-700 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded bg-orange-500" />
          Over-rep
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded bg-blue-500" />
          Expected
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded bg-cyan-500" />
          Under-rep
        </span>
      </div>
    </div>
  );
}

/**
 * Usage breakdown pie/bar display
 */
function UsageBreakdown({
  asStart,
  asMiddle,
  asEnd,
}: {
  asStart: number;
  asMiddle: number;
  asEnd: number;
}) {
  const total = asStart + asMiddle + asEnd;
  if (total === 0) return null;

  const startPct = Math.round((asStart / total) * 100);
  const middlePct = Math.round((asMiddle / total) * 100);
  const endPct = Math.round((asEnd / total) * 100);

  return (
    <div className="space-y-2">
      {/* Stacked bar */}
      <div className="h-3 bg-gray-700 rounded-full overflow-hidden flex">
        {startPct > 0 && (
          <div
            className="h-full bg-green-500"
            style={{ width: `${startPct}%` }}
            title={`Start: ${startPct}%`}
          />
        )}
        {middlePct > 0 && (
          <div
            className="h-full bg-blue-500"
            style={{ width: `${middlePct}%` }}
            title={`Middle: ${middlePct}%`}
          />
        )}
        {endPct > 0 && (
          <div
            className="h-full bg-red-500"
            style={{ width: `${endPct}%` }}
            title={`End: ${endPct}%`}
          />
        )}
      </div>
      {/* Legend */}
      <div className="flex justify-between text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 bg-green-500 rounded-full" />
          Start {startPct}%
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 bg-blue-500 rounded-full" />
          Middle {middlePct}%
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 bg-red-500 rounded-full" />
          End {endPct}%
        </span>
      </div>
    </div>
  );
}

export default function HoldDetailsPanel({
  position,
  stats,
}: HoldDetailsPanelProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-5 shadow-lg">
      {/* Header */}
      <div className="flex items-baseline justify-between mb-4">
        <h3 className="text-2xl font-bold text-white">Hold {position}</h3>
        <span className="text-gray-400 text-sm">
          {stats.frequency.toLocaleString()} problems
        </span>
      </div>

      {/* Grade Summary */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="bg-gray-700/50 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-400 mb-1">Mean</div>
          <div className="text-xl font-bold text-white">{stats.meanGrade}</div>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-400 mb-1">Median</div>
          <div className="text-xl font-bold text-white">{stats.medianGrade}</div>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-400 mb-1">Easiest</div>
          <div className="text-xl font-bold text-green-400">{stats.minGrade}</div>
        </div>
      </div>

      {/* Usage Breakdown */}
      <div className="mb-5">
        <h4 className="text-sm font-semibold text-gray-300 mb-2">
          Usage Breakdown
        </h4>
        <UsageBreakdown
          asStart={stats.asStart}
          asMiddle={stats.asMiddle}
          asEnd={stats.asEnd}
        />
      </div>

      {/* Grade Distribution */}
      <div>
        <h4 className="text-sm font-semibold text-gray-300 mb-2">
          Grade Distribution
        </h4>
        <GradeDistributionChart
          distribution={stats.gradeDistribution}
          distributionNormalized={stats.gradeDistributionNormalized}
        />
      </div>
    </div>
  );
}

