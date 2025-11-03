export interface Move {
  problemId: number;
  description: string; // e.g., "J4", "G6"
  isStart: boolean;
  isEnd: boolean;
}

export interface HoldSetup {
  description: string;
  holdsets: null | unknown;
  apiId: number;
}

export interface HoldSet {
  description: string;
  locations: null | unknown;
  apiId: number;
}

export interface Problem {
  name: string;
  grade: string;
  userGrade: string;
  setbyId: string;
  setby: string;
  method: string;
  userRating: number;
  repeats: number;
  holdsetup: HoldSetup;
  isBenchmark: boolean;
  isMaster: boolean;
  upgraded: boolean;
  downgraded: boolean;
  moves: Move[];
  holdsets: HoldSet[];
  hasBetaVideo: boolean;
  moonBoardConfigurationId: number;
  apiId: number;
  dateInserted: string;
  dateUpdated: string;
  dateDeleted: string | null;
}

export interface GridPosition {
  col: string; // A-K
  row: number; // 1-18
}
