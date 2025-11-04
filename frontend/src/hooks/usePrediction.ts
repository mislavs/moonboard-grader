/**
 * Custom hook for managing grade prediction state and logic
 */

import { useState } from 'react';
import { predictGrade, ApiError } from '../services/api';
import type { Move } from '../types/problem';
import type { PredictionResponse } from '../types/prediction';
import { PREDICTION_TOP_K, ERROR_MESSAGES } from '../config/constants';

export function usePrediction() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = async (moves: Move[]) => {
    if (moves.length === 0) {
      return;
    }

    try {
      setPredicting(true);
      setError(null);
      const result = await predictGrade(moves, PREDICTION_TOP_K);
      setPrediction(result);
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : ERROR_MESSAGES.PREDICTION_FAILED;
      setError(errorMessage);
      console.error('Failed to predict grade:', err);
    } finally {
      setPredicting(false);
    }
  };

  const reset = () => {
    setPrediction(null);
    setError(null);
  };

  return {
    prediction,
    predicting,
    error,
    predict,
    reset,
  };
}

