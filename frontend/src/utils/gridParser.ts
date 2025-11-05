import type { GridPosition } from '../types/problem';
import { BOARD_CONFIG } from '../config/board';

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

/**
 * Convert pixel coordinates to the nearest grid position
 * Returns null if the click is too far from any hold
 */
export function pixelToGrid(
  pixelX: number,
  pixelY: number,
  boardWidth: number,
  boardHeight: number
): GridPosition | null {
  const { columns, rows, margins } = BOARD_CONFIG;
  
  const gridWidth = boardWidth - margins.left - margins.right;
  const gridHeight = boardHeight - margins.top - margins.bottom;
  
  const colSpacing = gridWidth / (columns.length - 1);
  const rowSpacing = gridHeight / (rows - 1);
  
  // Find nearest column
  const colIndex = Math.round((pixelX - margins.left) / colSpacing);
  if (colIndex < 0 || colIndex >= columns.length) return null;
  
  // Find nearest row (remember: y increases downward, but row numbers increase upward)
  const rowIndex = Math.round((boardHeight - margins.bottom - pixelY) / rowSpacing);
  if (rowIndex < 0 || rowIndex >= rows) return null;
  
  const col = columns[colIndex];
  const row = rowIndex + 1; // rows are 1-indexed
  
  // Check if click is reasonably close to the calculated position
  const calculatedPos = gridToPixel({ col, row }, boardWidth, boardHeight);
  const distance = Math.sqrt(
    Math.pow(pixelX - calculatedPos.x, 2) + Math.pow(pixelY - calculatedPos.y, 2)
  );
  
  // Only register click if within 40 pixels of a hold
  if (distance > 40) return null;
  
  return { col, row };
}

/**
 * Format a grid position back into a description string (e.g., "J4")
 */
export function formatPosition(position: GridPosition): string {
  return `${position.col}${position.row}`;
}



