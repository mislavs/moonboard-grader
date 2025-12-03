/**
 * Types for board analytics data
 */

export interface HoldStats {
  minGrade: string;
  minGradeIndex: number;
  meanGrade: string;
  medianGrade: string;
  frequency: number;
  asStart: number;
  asMiddle: number;
  asEnd: number;
  gradeDistribution: Record<string, number>;
}

export interface AnalyticsMeta {
  totalProblems: number;
  totalProblemsUnfiltered: number;
  minRepeatsFilter: number;
}

export interface AnalyticsHeatmaps {
  meanGrade: number[][];
  minGrade: number[][];
  frequency: number[][];
}

export interface BoardAnalyticsResponse {
  holds: Record<string, HoldStats>;
  heatmaps: AnalyticsHeatmaps;
  meta: AnalyticsMeta;
}

export type HeatmapMetric = 'meanGrade' | 'minGrade' | 'frequency';

