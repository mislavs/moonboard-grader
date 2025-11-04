import type { Move } from '../types/problem';

/**
 * Handles the state transitions when clicking on a hold in create mode
 * Cycle: none -> intermediate -> start -> end -> removed
 */
export function getNextMoveState(
  moves: Move[],
  description: string
): Move[] {
  const existingMoveIndex = moves.findIndex(m => m.description === description);

  if (existingMoveIndex === -1) {
    // Add new intermediate hold
    return [...moves, { description, isStart: false, isEnd: false }];
  }

  const existingMove = moves[existingMoveIndex];
  const newMoves = [...moves];

  if (!existingMove.isStart && !existingMove.isEnd) {
    // Intermediate -> Start
    newMoves[existingMoveIndex] = { ...existingMove, isStart: true };
  } else if (existingMove.isStart && !existingMove.isEnd) {
    // Start -> End
    newMoves[existingMoveIndex] = { ...existingMove, isStart: false, isEnd: true };
  } else {
    // End -> Remove
    return moves.filter((_, i) => i !== existingMoveIndex);
  }

  return newMoves;
}

