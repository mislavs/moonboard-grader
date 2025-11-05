import type { Move } from '../types/problem';
import { HOLD_COLORS } from '../config/board';

/**
 * Determine the color of a hold based on its properties
 */
export function getHoldColor(move: Move): string {
  if (move.isStart) return HOLD_COLORS.start;
  if (move.isEnd) return HOLD_COLORS.end;
  return HOLD_COLORS.intermediate;
}

