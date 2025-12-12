/**
 * Custom hook for managing beta solving state and logic
 */

import { useState, useCallback } from 'react';
import { solveBeta, ApiError } from '../services/api';
import type { Move } from '../types/problem';
import type { BetaResponse } from '../types/beta';

export function useBeta() {
  const [beta, setBeta] = useState<BetaResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchBeta = useCallback(async (moves: Move[]) => {
    if (moves.length === 0) {
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await solveBeta(moves);
      setBeta(result);
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : 'Failed to solve beta';
      setError(errorMessage);
      console.error('Failed to solve beta:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setBeta(null);
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
