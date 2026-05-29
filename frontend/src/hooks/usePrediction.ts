/**
 * Custom hook for managing grade prediction state and logic
 */

import { useState, useCallback, useRef } from 'react';
import { predictGrade, ApiError, type BoardSetupParams } from '../services/api';
import type { Move } from '../types/problem';
import type { PredictionResponse } from '../types/prediction';
import { PREDICTION_TOP_K, ERROR_MESSAGES } from '../config/api';

export function usePrediction() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  const predict = useCallback(async (moves: Move[], setupParams?: BoardSetupParams) => {
    if (moves.length === 0) {
      return;
    }

    const requestId = ++requestIdRef.current;

    try {
      setPredicting(true);
      setError(null);
      const result = await predictGrade(moves, PREDICTION_TOP_K, setupParams);
      if (requestId === requestIdRef.current) {
        setPrediction(result);
      }
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : ERROR_MESSAGES.PREDICTION_FAILED;
      if (requestId === requestIdRef.current) {
        setError(errorMessage);
      }
      console.error('Failed to predict grade:', err);
    } finally {
      if (requestId === requestIdRef.current) {
        setPredicting(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    requestIdRef.current += 1;
    setPrediction(null);
    setPredicting(false);
    setError(null);
  }, []);

  return {
    prediction,
    predicting,
    error,
    predict,
    reset,
  };
}

