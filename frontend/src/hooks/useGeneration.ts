/**
 * Custom hook for managing problem generation state and logic
 */

import { useState, useCallback, useRef } from 'react';
import {
  generateProblem,
  ApiError,
  type GenerateResponse,
  type BoardSetupParams,
} from '../services/api';
import { ERROR_MESSAGES } from '../config/api';

export function useGeneration() {
  const [generated, setGenerated] = useState<GenerateResponse | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  const generate = useCallback(async (
    grade: string = '6A+',
    temperature: number = 1.0,
    setupParams?: BoardSetupParams
  ) => {
    const requestId = ++requestIdRef.current;

    try {
      setGenerating(true);
      setError(null);
      const result = await generateProblem(grade, temperature, setupParams);
      if (requestId !== requestIdRef.current) {
        return null;
      }

      setGenerated(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : ERROR_MESSAGES.SERVER_CONNECTION_FAILED;
      if (requestId === requestIdRef.current) {
        setError(errorMessage);
      }
      console.error('Failed to generate problem:', err);
      return null;
    } finally {
      if (requestId === requestIdRef.current) {
        setGenerating(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    requestIdRef.current += 1;
    setGenerated(null);
    setGenerating(false);
    setError(null);
  }, []);

  return {
    generated,
    generating,
    error,
    generate,
    reset,
  };
}

