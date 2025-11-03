import type { GridPosition } from '../types/problem';

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
  const cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'];
  const colIndex = cols.indexOf(position.col);
  
  const leftMargin = 65;
  const topMargin = 80;
  const rightMargin = 30;
  const bottomMargin = 55;
  
  const gridWidth = boardWidth - leftMargin - rightMargin;
  const gridHeight = boardHeight - topMargin - bottomMargin;
  
  const colSpacing = gridWidth / (cols.length - 1);
  const rowSpacing = gridHeight / (18 - 1);
  
  const x = leftMargin + (colSpacing * colIndex);
  const y = topMargin + gridHeight - (rowSpacing * (position.row - 1));
  
  return { x, y };
}



