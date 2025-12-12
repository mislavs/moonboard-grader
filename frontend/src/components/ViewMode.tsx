import { useEffect, useState } from 'react';
import MoonBoard from './MoonBoard';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import ProblemNavigator from './ProblemNavigator';
import CruxHighlightToggle from './CruxHighlightToggle';
import BetaToggle from './BetaToggle';
import { fetchProblem, ApiError } from '../services/api';
import { usePrediction } from '../hooks/usePrediction';
import { useBeta } from '../hooks/useBeta';
import type { Problem } from '../types/problem';
import { ERROR_MESSAGES } from '../config/api';

export default function ViewMode() {
  const [selectedProblemId, setSelectedProblemId] = useState<number | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAttention, setShowAttention] = useState(false);
  const [showBeta, setShowBeta] = useState(false);
  
  const { prediction, predicting, predict, reset: resetPrediction } = usePrediction();
  const { beta, loading: betaLoading, fetchBeta, reset: resetBeta } = useBeta();

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

  // Reset prediction and beta when problem changes
  useEffect(() => {
    resetPrediction();
    resetBeta();
  }, [selectedProblemId, resetPrediction, resetBeta]);

  // Auto-fetch heatmap when problem loads and toggle is on
  useEffect(() => {
    if (showAttention && problem && !predicting && !prediction?.attention_map) {
      predict(problem.moves);
    }
  }, [problem, showAttention, predicting, prediction?.attention_map, predict]);

  // Auto-fetch beta when problem loads and toggle is on
  useEffect(() => {
    if (showBeta && problem && !betaLoading && !beta && !loading) {
      fetchBeta(problem.moves);
    }
  }, [problem, showBeta, betaLoading, beta, fetchBeta, loading]);

  // Fetch attention map when toggle is turned on
  const handleToggleAttention = async (checked: boolean) => {
    setShowAttention(checked);
    if (checked && problem && !prediction?.attention_map) {
      await predict(problem.moves);
    }
  };

  // Fetch beta when toggle is turned on
  const handleToggleBeta = async (checked: boolean) => {
    setShowBeta(checked);
    if (checked && problem && !beta) {
      await fetchBeta(problem.moves);
    }
  };

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
        <div className="flex flex-col gap-4">
          <div className="relative">
            {error && <ErrorMessage message={error} />}
            {problem && !error && (
              <MoonBoard
                problem={problem}
                mode="view"
                attentionMap={prediction?.attention_map}
                showAttention={showAttention}
                beta={beta ?? undefined}
                showBeta={showBeta}
              />
            )}
            
            {/* Loading Overlay */}
            {(loading || predicting || betaLoading) && (
              <div className="absolute inset-0 bg-black/40 rounded-lg flex items-center justify-center z-10">
                <LoadingSpinner message={betaLoading ? "Loading beta..." : predicting ? "Loading heatmap..." : "Loading problem..."} />
              </div>
            )}
          </div>

          {/* Crux Highlight Toggle */}
          {problem && !error && (
            <CruxHighlightToggle
              checked={showAttention}
              onChange={handleToggleAttention}
            />
          )}

          {/* Beta Toggle */}
          {problem && !error && (
            <BetaToggle
              checked={showBeta}
              onChange={handleToggleBeta}
            />
          )}
        </div>
      </div>
    </div>
  );
}

