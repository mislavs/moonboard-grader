/**
 * Custom hook for managing duplicate check state and logic
 */

import { useState, useCallback, useRef } from 'react';
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
  const requestIdRef = useRef(0);

  const checkForDuplicate = useCallback(async (moves: Move[]) => {
    if (moves.length === 0) {
      return;
    }

    const requestId = ++requestIdRef.current;

    try {
      setChecking(true);
      setError(null);
      
      // Check for duplicate
      const result = await checkDuplicate(moves);
      
      if (result.exists && result.problem_id !== null) {
        // Fetch the problem details to get name and grade
        const problem = await fetchProblem(result.problem_id);
        if (requestId === requestIdRef.current) {
          setDuplicate({
            exists: true,
            problemName: problem.name,
            problemGrade: problem.grade,
          });
        }
      } else {
        if (requestId === requestIdRef.current) {
          setDuplicate({
            exists: false,
          });
        }
      }
    } catch (err) {
      const errorMessage = err instanceof ApiError 
        ? err.message 
        : 'Failed to check for duplicates';
      if (requestId === requestIdRef.current) {
        setError(errorMessage);
      }
      console.error('Failed to check duplicate:', err);
    } finally {
      if (requestId === requestIdRef.current) {
        setChecking(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    requestIdRef.current += 1;
    setDuplicate(null);
    setChecking(false);
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

