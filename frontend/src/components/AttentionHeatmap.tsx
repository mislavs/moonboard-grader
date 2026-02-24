/**
 * AttentionHeatmap component
 *
 * Renders a heatmap overlay on the Moonboard showing which areas
 * the model considers most important for grade prediction.
 */

import { useMemo, type ReactElement } from "react";
import { BOARD_CONFIG } from "../config/board";
import { getHeatmapColor } from "../constants/heatmapColors";

const GRID_ROWS = 18;
const GRID_COLS = 11;

interface AttentionHeatmapProps {
  /** 18x11 array of attention weights (0-1), where [0][0] is A1 (bottom-left) */
  attentionMap: number[][];
  /** Opacity of the heatmap overlay (0-1) */
  opacity?: number;
}

/**
 * Normalize attention values to use full 0-1 range for better contrast
 */
function normalizeAttentionMap(attentionMap: number[][]): number[][] {
  // Find min and max values
  let min = Infinity;
  let max = -Infinity;

  for (const row of attentionMap) {
    for (const val of row) {
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }

  // Avoid division by zero
  const range = max - min;
  if (range === 0) return attentionMap;

  // Normalize to 0-1 range
  return attentionMap.map((row) => row.map((val) => (val - min) / range));
}

export default function AttentionHeatmap({
  attentionMap,
  opacity = 0.7,
}: AttentionHeatmapProps) {
  const { width, height, columns, rows, margins } = BOARD_CONFIG;

  // Calculate grid dimensions (same as gridParser.ts)
  const gridWidth = width - margins.left - margins.right;
  const gridHeight = height - margins.top - margins.bottom;
  const colSpacing = gridWidth / (columns.length - 1);
  const rowSpacing = gridHeight / (rows - 1);

  // Cell dimensions (centered on grid intersection points)
  const cellWidth = colSpacing;
  const cellHeight = rowSpacing;

  // Check if attention map is valid
  const isValid =
    attentionMap &&
    attentionMap.length === GRID_ROWS &&
    attentionMap[0]?.length === GRID_COLS;

  // Memoize normalized map (returns empty array if invalid)
  const normalizedMap = useMemo(
    () => (isValid ? normalizeAttentionMap(attentionMap) : []),
    [attentionMap, isValid]
  );

  // Memoize cells (returns empty array if invalid)
  const cells = useMemo(() => {
    if (!isValid) return [];

    const result: ReactElement[] = [];

    for (let tensorRow = 0; tensorRow < GRID_ROWS; tensorRow++) {
      for (let col = 0; col < GRID_COLS; col++) {
        const value = normalizedMap[tensorRow][col];
        const centerX = margins.left + col * colSpacing;
        const centerY = margins.top + gridHeight - tensorRow * rowSpacing;
        const rectX = centerX - cellWidth / 2;
        const rectY = centerY - cellHeight / 2;

        result.push(
          <rect
            key={`${tensorRow}-${col}`}
            x={rectX}
            y={rectY}
            width={cellWidth}
            height={cellHeight}
            fill={getHeatmapColor(value)}
            opacity={opacity}
          />
        );
      }
    }

    return result;
  }, [isValid, normalizedMap, margins, colSpacing, rowSpacing, gridHeight, cellWidth, cellHeight, opacity]);

  // Early return after all hooks
  if (!isValid) return null;

  return (
    <svg
      width={width}
      height={height}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 1 }}
    >
      <defs>
        <filter id="heatmap-blur" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="12" />
        </filter>
      </defs>
      <g filter="url(#heatmap-blur)">
        {cells}
      </g>
    </svg>
  );
}
