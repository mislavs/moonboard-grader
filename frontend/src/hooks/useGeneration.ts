/**
 * Custom hook for managing problem generation state and logic
 */

import { useState } from 'react';
import { generateProblem, ApiError, type GenerateResponse } from '../services/api';
import { ERROR_MESSAGES } from '../config/api';

export function useGeneration() {
  const [generated, setGenerated] = useState<GenerateResponse | null>(null);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async (grade: string = '6A+', temperature: number = 1.0) => {
    try {
      setGenerating(true);
      setError(null);
      const result = await generateProblem(grade, temperature);
      setGenerated(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : ERROR_MESSAGES.SERVER_CONNECTION_FAILED;
      setError(errorMessage);
      console.error('Failed to generate problem:', err);
      return null;
    } finally {
      setGenerating(false);
    }
  };

  const reset = () => {
    setGenerated(null);
    setError(null);
  };

  return {
    generated,
    generating,
    error,
    generate,
    reset,
  };
}

