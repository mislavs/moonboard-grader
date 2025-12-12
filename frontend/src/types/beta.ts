/**
 * Types for the beta solver API responses
 */

export interface BetaMove {
  /** Hold description e.g., "J4" */
  hold: string;
  /** Which hand to use */
  hand: "LH" | "RH";
}

export interface StartHand {
  /** Which hand (LH or RH) */
  hand: "LH" | "RH";
  /** Hold description e.g., "J4" */
  description: string;
}

export interface BetaResponse {
  /** Starting hand positions */
  startHands: StartHand[];
  /** Sequence of moves */
  moves: BetaMove[];
}
