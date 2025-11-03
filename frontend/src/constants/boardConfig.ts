/**
 * Moonboard configuration constants
 */
export const BOARD_CONFIG = {
  width: 550,
  height: 900,
  holdRadius: 22,
  columns: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'] as const,
  rows: 18,
  margins: {
    left: 65,
    top: 80,
    right: 30,
    bottom: 55,
  },
} as const;

export const HOLD_COLORS = {
  start: '#22c55e',     // green
  intermediate: '#3b82f6', // blue
  end: '#ef4444',        // red
} as const;

export type HoldType = keyof typeof HOLD_COLORS;

