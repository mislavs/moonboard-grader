/**
 * Custom hook for managing beta solving state and logic
 */

import { useState, useCallback, useRef } from 'react';
import { solveBeta, ApiError, type BoardSetupParams } from '../services/api';
import type { Move } from '../types/problem';
import type { BetaResponse } from '../types/beta';

export function useBeta() {
  const [beta, setBeta] = useState<BetaResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  const fetchBeta = useCallback(async (
    moves: Move[],
    setupParams?: BoardSetupParams
  ) => {
    if (moves.length === 0) {
      return;
    }

    const requestId = ++requestIdRef.current;

    try {
      setLoading(true);
      setError(null);
      const result = await solveBeta(moves, setupParams);
      if (requestId === requestIdRef.current) {
        setBeta(result);
      }
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : 'Failed to solve beta';
      if (requestId === requestIdRef.current) {
        setError(errorMessage);
      }
      console.error('Failed to solve beta:', err);
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    requestIdRef.current += 1;
    setBeta(null);
    setLoading(false);
    setError(null);
  }, []);

  return {
    beta,
    loading,
    error,
    fetchBeta,
    reset,
  };
}
