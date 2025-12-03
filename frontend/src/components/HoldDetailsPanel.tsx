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
 * Simple bar chart for grade distribution
 */
function GradeDistributionChart({
  distribution,
}: {
  distribution: Record<string, number>;
}) {
  // Sort grades by frequency and take top entries
  const sortedGrades = Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);

  const maxCount = Math.max(...sortedGrades.map(([, count]) => count));

  if (sortedGrades.length === 0) return null;

  return (
    <div className="space-y-1.5">
      {sortedGrades.map(([grade, count]) => (
        <div key={grade} className="flex items-center gap-2 text-sm">
          <span className="w-10 text-gray-400 font-mono">{grade}</span>
          <div className="flex-1 h-4 bg-gray-700 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded transition-all duration-300"
              style={{ width: `${(count / maxCount) * 100}%` }}
            />
          </div>
          <span className="w-10 text-right text-gray-400">{count}</span>
        </div>
      ))}
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
        <GradeDistributionChart distribution={stats.gradeDistribution} />
      </div>
    </div>
  );
}

