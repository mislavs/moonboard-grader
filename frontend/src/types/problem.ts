export interface Move {
  description: string; // e.g., "J4", "G6"
  isStart: boolean;
  isEnd: boolean;
}

export interface Problem {
  id: number;
  name: string;
  grade: string;
  moves: Move[];
}

export interface GridPosition {
  col: string; // A-K
  row: number; // 1-18
}
