import { useEffect, useState } from 'react';
import MoonBoard from './MoonBoard';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import ProblemNavigator from './ProblemNavigator';
import { fetchProblem, ApiError } from '../services/api';
import type { Problem } from '../types/problem';
import { ERROR_MESSAGES } from '../config/api';

export default function ViewMode() {
  const [selectedProblemId, setSelectedProblemId] = useState<number | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadProblem() {
      if (selectedProblemId === null) {
        return;
      }
      
      try {
        setLoading(true);
        setError(null);
        const problem = await fetchProblem(selectedProblemId);
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
  }, [selectedProblemId]);

  const handleProblemSelect = (problemId: number) => {
    setSelectedProblemId(problemId);
  };

  const handleFirstProblemLoaded = (problemId: number) => {
    if (selectedProblemId === null) {
      setSelectedProblemId(problemId);
    }
  };

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Main Content: Navigator on Left, MoonBoard on Right */}
      <div className="flex flex-row gap-8 items-start justify-center w-full px-4">
        {/* Left Panel: Problem Navigator */}
        <div className="w-96 flex">
          <ProblemNavigator
            selectedProblemId={selectedProblemId}
            onProblemSelect={handleProblemSelect}
            onFirstProblemLoaded={handleFirstProblemLoaded}
          />
        </div>

        {/* Right Panel: MoonBoard */}
        <div className="flex flex-col relative">
          {error && <ErrorMessage message={error} />}
          {problem && !error && (
            <MoonBoard problem={problem} mode="view" />
          )}
          
          {/* Loading Overlay */}
          {loading && (
            <div className="absolute inset-0 bg-black/40 rounded-lg flex items-center justify-center z-10">
              <LoadingSpinner message="Loading problem..." />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

