import type { GridPosition } from '../types/problem';
import { BOARD_CONFIG } from '../constants/boardConfig';

/**
 * Parse a move description (e.g., "J4", "A16") into grid coordinates
 */
export function parsePosition(description: string): GridPosition {
  const col = description[0]; // A-K
  const row = parseInt(description.slice(1)); // 1-18
  return { col, row };
}

/**
 * Convert grid position to pixel coordinates for rendering
 * The moonboard has 11 columns (A-K) and 18 rows (1-18)
 */
export function gridToPixel(
  position: GridPosition,
  boardWidth: number,
  boardHeight: number
): { x: number; y: number } {
  const { columns, rows, margins } = BOARD_CONFIG;
  const colIndex = columns.indexOf(position.col as typeof columns[number]);
  
  const gridWidth = boardWidth - margins.left - margins.right;
  const gridHeight = boardHeight - margins.top - margins.bottom;
  
  const colSpacing = gridWidth / (columns.length - 1);
  const rowSpacing = gridHeight / (rows - 1);
  
  const x = margins.left + (colSpacing * colIndex);
  const y = margins.top + gridHeight - (rowSpacing * (position.row - 1));
  
  return { x, y };
}



