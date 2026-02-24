import { useCallback, useEffect, useState } from 'react';
import MoonBoard from './MoonBoard';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import ProblemNavigator from './ProblemNavigator';
import CruxHighlightToggle from './CruxHighlightToggle';
import BetaToggle from './BetaToggle';
import { fetchProblem, ApiError } from '../services/api';
import { usePrediction } from '../hooks/usePrediction';
import { useBeta } from '../hooks/useBeta';
import { useBoardSetupParams } from '../contexts/BoardSetupContext';
import type { Problem } from '../types/problem';
import { ERROR_MESSAGES } from '../config/api';

export default function ViewMode() {
  const [selectedProblemId, setSelectedProblemId] = useState<number | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAttention, setShowAttention] = useState(false);
  const [showBeta, setShowBeta] = useState(false);
  const setupParams = useBoardSetupParams();
  
  const { prediction, predicting, predict, reset: resetPrediction } = usePrediction();
  const { beta, loading: betaLoading, fetchBeta, reset: resetBeta } = useBeta();

  useEffect(() => {
    let isCancelled = false;

    async function loadProblem() {
      if (selectedProblemId === null) {
        setProblem(null);
        setLoading(false);
        return;
      }
      
      try {
        setLoading(true);
        setError(null);
        const loadedProblem = await fetchProblem(selectedProblemId);
        if (!isCancelled) {
          setProblem(loadedProblem);
        }
      } catch (err) {
        if (isCancelled) {
          return;
        }
        const errorMessage = err instanceof ApiError 
          ? err.message 
          : ERROR_MESSAGES.PROBLEM_LOAD_FAILED;
        setError(errorMessage);
        console.error('Failed to load problem:', err);
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    }

    loadProblem();

    return () => {
      isCancelled = true;
    };
  }, [selectedProblemId]);

  // Reset prediction and beta when problem changes
  useEffect(() => {
    resetPrediction();
    resetBeta();
  }, [selectedProblemId, setupParams, resetPrediction, resetBeta]);

  // Auto-fetch heatmap when problem loads and toggle is on
  useEffect(() => {
    if (showAttention && problem && !predicting && !prediction?.attention_map) {
      predict(problem.moves, setupParams);
    }
  }, [problem, showAttention, predicting, prediction?.attention_map, predict, setupParams]);

  // Auto-fetch beta when problem loads and toggle is on
  useEffect(() => {
    if (showBeta && problem && !betaLoading && !beta && !loading) {
      fetchBeta(problem.moves, setupParams);
    }
  }, [problem, showBeta, betaLoading, beta, fetchBeta, loading, setupParams]);

  // Fetch attention map when toggle is turned on
  const handleToggleAttention = async (checked: boolean) => {
    setShowAttention(checked);
    if (checked && problem && !prediction?.attention_map) {
      await predict(problem.moves, setupParams);
    }
  };

  // Fetch beta when toggle is turned on
  const handleToggleBeta = async (checked: boolean) => {
    setShowBeta(checked);
    if (checked && problem && !beta) {
      await fetchBeta(problem.moves, setupParams);
    }
  };

  const handleProblemSelect = (problemId: number) => {
    setSelectedProblemId((current) => (current === problemId ? current : problemId));
  };

  const handleFirstProblemLoaded = useCallback((problemId: number) => {
    setSelectedProblemId((current) => current ?? problemId);
  }, []);

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
            setupParams={setupParams}
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
            {(predicting || betaLoading) && (
              <div className="absolute inset-0 bg-black/40 rounded-lg flex items-center justify-center z-10">
                <LoadingSpinner message={betaLoading ? "Loading beta..." : "Loading heatmap..."} />
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

