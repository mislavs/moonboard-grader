/**
 * Custom hook for managing duplicate check state and logic
 */

import { useState, useCallback } from 'react';
import { checkDuplicate, fetchProblem, ApiError } from '../services/api';
import type { Move } from '../types/problem';

interface DuplicateInfo {
  exists: boolean;
  problemName?: string;
  problemGrade?: string;
}

export function useDuplicateCheck() {
  const [duplicate, setDuplicate] = useState<DuplicateInfo | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkForDuplicate = useCallback(async (moves: Move[]) => {
    if (moves.length === 0) {
      return;
    }

    try {
      setChecking(true);
      setError(null);
      
      // Check for duplicate
      const result = await checkDuplicate(moves);
      
      if (result.exists && result.problem_id !== null) {
        // Fetch the problem details to get name and grade
        const problem = await fetchProblem(result.problem_id);
        setDuplicate({
          exists: true,
          problemName: problem.name,
          problemGrade: problem.grade,
        });
      } else {
        setDuplicate({
          exists: false,
        });
      }
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : 'Failed to check for duplicates';
      setError(errorMessage);
      console.error('Failed to check duplicate:', err);
    } finally {
      setChecking(false);
    }
  }, []);

  const reset = useCallback(() => {
    setDuplicate(null);
    setError(null);
  }, []);

  return {
    duplicate,
    checking,
    error,
    checkForDuplicate,
    reset,
  };
}

