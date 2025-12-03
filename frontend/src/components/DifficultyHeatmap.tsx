/**
 * DifficultyHeatmap component
 *
 * Renders an interactive heatmap overlay on the Moonboard showing
 * hold difficulty based on various metrics.
 */

import { useMemo, type ReactElement } from "react";
import { BOARD_CONFIG } from "../config/board";

const GRID_ROWS = 18;
const GRID_COLS = 11;
const COLUMNS = "ABCDEFGHIJK";

interface CellData {
  position: string;
  x: number;
  y: number;
  color: string;
  isSelected: boolean;
}

interface DifficultyHeatmapProps {
  /** 18x11 array of normalized values (0-1), where [0][0] is row 1 */
  heatmapData: number[][];
  /** Opacity of the heatmap overlay (0-1) */
  opacity?: number;
  /** Currently selected hold position (e.g., "F7") */
  selectedHold?: string | null;
  /** Callback when a hold is clicked */
  onHoldClick?: (position: string) => void;
  /** Callback when hovering over a hold */
  onHoldHover?: (position: string | null) => void;
}

/**
 * Convert normalized value (0-1) to a color on a blue-white-red scale
 * Blue = easy (low values), Red = hard (high values)
 */
function getHeatmapColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));

  if (v < 0.5) {
    // Blue (59, 130, 246) to White (255, 255, 255)
    const t = v * 2;
    const r = Math.round(59 + (255 - 59) * t);
    const g = Math.round(130 + (255 - 130) * t);
    const b = Math.round(246 + (255 - 246) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    // White (255, 255, 255) to Red (239, 68, 68)
    const t = (v - 0.5) * 2;
    const r = Math.round(255 + (239 - 255) * t);
    const g = Math.round(255 + (68 - 255) * t);
    const b = Math.round(255 + (68 - 255) * t);
    return `rgb(${r},${g},${b})`;
  }
}

/**
 * Convert grid position to hold position string (e.g., "F7")
 */
function gridToPosition(row: number, col: number): string {
  return `${COLUMNS[col]}${row + 1}`;
}

export default function DifficultyHeatmap({
  heatmapData,
  opacity = 0.6,
  selectedHold,
  onHoldClick,
  onHoldHover,
}: DifficultyHeatmapProps) {
  const { width, height, columns, rows, margins } = BOARD_CONFIG;

  // Calculate grid dimensions
  const gridWidth = width - margins.left - margins.right;
  const gridHeight = height - margins.top - margins.bottom;
  const colSpacing = gridWidth / (columns.length - 1);
  const rowSpacing = gridHeight / (rows - 1);

  // Cell dimensions
  const cellWidth = colSpacing;
  const cellHeight = rowSpacing;

  // Check if heatmap data is valid
  const isValid =
    heatmapData &&
    heatmapData.length === GRID_ROWS &&
    heatmapData[0]?.length === GRID_COLS;

  // Memoize cell data
  const cellsData = useMemo(() => {
    if (!isValid) return [];

    const result: CellData[] = [];

    for (let tensorRow = 0; tensorRow < GRID_ROWS; tensorRow++) {
      for (let col = 0; col < GRID_COLS; col++) {
        const value = heatmapData[tensorRow][col];
        const position = gridToPosition(tensorRow, col);
        const isSelected = selectedHold === position;

        // Calculate position (row 0 in data = row 1 on board = bottom)
        const centerX = margins.left + col * colSpacing;
        const centerY = margins.top + gridHeight - tensorRow * rowSpacing;
        const rectX = centerX - cellWidth / 2;
        const rectY = centerY - cellHeight / 2;

        result.push({
          position,
          x: rectX,
          y: rectY,
          color: getHeatmapColor(value),
          isSelected,
        });
      }
    }

    return result;
  }, [
    isValid,
    heatmapData,
    selectedHold,
    margins,
    colSpacing,
    rowSpacing,
    gridHeight,
    cellWidth,
    cellHeight,
  ]);

  // Memoize rendered cells
  const cells = useMemo(() => {
    return cellsData.map((cell): ReactElement => (
      <g key={cell.position}>
        {/* Heatmap cell */}
        <rect
          x={cell.x}
          y={cell.y}
          width={cellWidth}
          height={cellHeight}
          fill={cell.color}
          opacity={opacity}
          className="transition-opacity duration-150"
        />
        {/* Interactive overlay */}
        <rect
          x={cell.x}
          y={cell.y}
          width={cellWidth}
          height={cellHeight}
          fill="transparent"
          stroke={cell.isSelected ? "#ffffff" : "transparent"}
          strokeWidth={cell.isSelected ? 3 : 0}
          className="cursor-pointer hover:stroke-white hover:stroke-2"
          onClick={() => onHoldClick?.(cell.position)}
          onMouseEnter={() => onHoldHover?.(cell.position)}
          onMouseLeave={() => onHoldHover?.(null)}
        />
      </g>
    ));
  }, [cellsData, cellWidth, cellHeight, opacity, onHoldClick, onHoldHover]);

  // Memoize blurred background cells
  const blurredCells = useMemo(() => {
    return cellsData.map((cell): ReactElement => (
      <rect
        key={`bg-${cell.position}`}
        x={cell.x}
        y={cell.y}
        width={cellWidth}
        height={cellHeight}
        fill={cell.color}
        opacity={opacity * 0.5}
      />
    ));
  }, [cellsData, cellWidth, cellHeight, opacity]);

  if (!isValid) return null;

  return (
    <svg
      width={width}
      height={height}
      className="absolute inset-0"
      style={{ zIndex: 1 }}
    >
      <defs>
        <filter id="difficulty-blur" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="8" />
        </filter>
      </defs>
      {/* Blurred background layer */}
      <g filter="url(#difficulty-blur)" style={{ pointerEvents: "none" }}>
        {blurredCells}
      </g>
      {/* Interactive layer */}
      <g>{cells}</g>
    </svg>
  );
}

