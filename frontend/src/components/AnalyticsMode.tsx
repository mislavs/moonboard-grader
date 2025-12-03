/**
 * AnalyticsMode component
 *
 * Main container for the Board Analytics tab, displaying hold difficulty
 * heatmaps and detailed statistics.
 */

import { useState, useEffect } from "react";
import { fetchBoardAnalytics } from "../services/api";
import type {
  BoardAnalyticsResponse,
  HoldStats,
  HeatmapMetric,
} from "../types/analytics";
import { BOARD_CONFIG } from "../config/board";
import DifficultyHeatmap from "./DifficultyHeatmap";
import HoldDetailsPanel from "./HoldDetailsPanel";
import LoadingSpinner from "./LoadingSpinner";
import ErrorMessage from "./ErrorMessage";
import moonboardImage from "../assets/moonboard.jpg";

const METRIC_OPTIONS: { value: HeatmapMetric; label: string }[] = [
  { value: "meanGrade", label: "Mean Grade" },
  { value: "minGrade", label: "Min Grade" },
  { value: "frequency", label: "Frequency" },
];

/**
 * Color scale legend component
 */
function ColorScaleLegend({ metric }: { metric: HeatmapMetric }) {
  const labels = metric === "frequency" 
    ? { low: "Rarely Used", high: "Often Used" }
    : { low: "Easier", high: "Harder" };

  return (
    <div className="flex items-center gap-3 text-sm text-gray-400">
      <span>{labels.low}</span>
      <div className="w-32 h-3 rounded-full bg-gradient-to-r from-blue-500 via-white to-red-500" />
      <span>{labels.high}</span>
    </div>
  );
}

/**
 * Metric selector dropdown
 */
function MetricSelector({
  value,
  onChange,
}: {
  value: HeatmapMetric;
  onChange: (metric: HeatmapMetric) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-gray-400 text-sm">Show:</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as HeatmapMetric)}
        className="bg-gray-700 text-white rounded-lg px-3 py-2 text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
      >
        {METRIC_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

/**
 * Stats summary panel
 */
function StatsSummary({ meta }: { meta: BoardAnalyticsResponse["meta"] }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 mb-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-2">Dataset Info</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="text-gray-400">Problems analyzed:</div>
        <div className="text-white font-medium">
          {meta.totalProblems.toLocaleString()}
        </div>
        <div className="text-gray-400">Min repeats filter:</div>
        <div className="text-white font-medium">{meta.minRepeatsFilter}+</div>
      </div>
    </div>
  );
}

export default function AnalyticsMode() {
  const [analytics, setAnalytics] = useState<BoardAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<HeatmapMetric>("meanGrade");
  const [selectedHold, setSelectedHold] = useState<string | null>(null);
  const [hoveredHold, setHoveredHold] = useState<string | null>(null);

  // Fetch analytics data on mount
  useEffect(() => {
    async function loadAnalytics() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchBoardAnalytics();
        setAnalytics(data);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load analytics";
        setError(message);
        console.error("Failed to load analytics:", err);
      } finally {
        setLoading(false);
      }
    }

    loadAnalytics();
  }, []);

  // Get the current heatmap data based on selected metric
  const currentHeatmap = analytics?.heatmaps[selectedMetric] ?? null;

  // Get stats for the selected or hovered hold
  const displayHold = hoveredHold ?? selectedHold;
  const displayStats: HoldStats | null =
    displayHold && analytics?.holds[displayHold]
      ? analytics.holds[displayHold]
      : null;

  const { width, height } = BOARD_CONFIG;

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner message="Loading board analytics..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center py-12">
        <ErrorMessage message={error} />
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="flex justify-center py-12">
        <ErrorMessage message="No analytics data available" />
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Main Content */}
      <div className="flex flex-row gap-8 items-start justify-center w-full px-4">
        {/* Left Panel: Controls and Details */}
        <div className="w-80 flex flex-col gap-4">
          {/* Metric Selector */}
          <div className="bg-gray-800 rounded-lg p-4">
            <MetricSelector value={selectedMetric} onChange={setSelectedMetric} />
            <div className="mt-3">
              <ColorScaleLegend metric={selectedMetric} />
            </div>
          </div>

          {/* Stats Summary */}
          <StatsSummary meta={analytics.meta} />

          {/* Hold Details */}
          {displayStats && displayHold ? (
            <HoldDetailsPanel position={displayHold} stats={displayStats} />
          ) : (
            <div className="bg-gray-800 rounded-lg p-5 text-center text-gray-400">
              <p className="mb-2">Click on a hold to see details</p>
              <p className="text-sm">
                Hover over holds to preview their statistics
              </p>
            </div>
          )}
        </div>

        {/* Right Panel: Board with Heatmap */}
        <div className="flex flex-col gap-4">
          <div className="relative bg-gray-800 rounded-lg shadow-2xl p-4">
            <div
              className="relative border-4 border-gray-700 rounded overflow-hidden"
              style={{ width, height }}
            >
              {/* Background Image */}
              <img
                src={moonboardImage}
                alt="Moonboard"
                className="absolute inset-0 w-full h-full object-cover"
                style={{ width, height }}
              />

              {/* Heatmap Overlay */}
              {currentHeatmap && (
                <DifficultyHeatmap
                  heatmapData={currentHeatmap}
                  selectedHold={selectedHold}
                  onHoldClick={setSelectedHold}
                  onHoldHover={setHoveredHold}
                  opacity={0.55}
                />
              )}
            </div>
          </div>

          {/* Instructions */}
          <p className="text-center text-gray-500 text-sm">
            Click on any hold position to see detailed statistics
          </p>
        </div>
      </div>
    </div>
  );
}

