/**
 * Types related to grade prediction
 */

import type { Move } from './problem';

export interface TopKPrediction {
  grade: string;
  probability: number;
}

export interface PredictionResponse {
  predicted_grade: string;
  confidence: number;
  top_k_predictions: TopKPrediction[];
}

export interface PredictionRequest {
  moves: Move[];
  top_k?: number;
}

