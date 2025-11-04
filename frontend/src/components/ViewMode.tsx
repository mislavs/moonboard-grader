import { useEffect, useState } from 'react';
import MoonBoard from './MoonBoard';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import { fetchProblem, ApiError } from '../services/api';
import type { Problem } from '../types/problem';
import { DEFAULT_PROBLEM_ID, ERROR_MESSAGES } from '../config/constants';

export default function ViewMode() {
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadProblem() {
      try {
        setLoading(true);
        setError(null);
        const problem = await fetchProblem(DEFAULT_PROBLEM_ID);
        setProblem(problem);
      } catch (err) {
        const errorMessage = err instanceof ApiError 
          ? err.message 
          : ERROR_MESSAGES.PROBLEM_LOAD_FAILED;
        setError(errorMessage);
        console.error('Failed to load problem:', err);
      } finally {
        setLoading(false);
      }
    }

    loadProblem();
  }, []);

  return (
    <>
      {loading && <LoadingSpinner message="Loading problem..." />}
      {error && <ErrorMessage message={error} />}
      {problem && !loading && !error && (
        <MoonBoard problem={problem} mode="view" />
      )}
    </>
  );
}

