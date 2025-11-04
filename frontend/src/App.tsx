import { useEffect, useState } from 'react';
import MoonBoard from './components/MoonBoard';
import TabNavigation from './components/TabNavigation';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import CreateModeControls from './components/CreateModeControls';
import { fetchProblem, ApiError } from './services/api';
import type { Problem, Move } from './types/problem';

type Mode = 'view' | 'create';

function App() {
  const [mode, setMode] = useState<Mode>('view');
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [createdMoves, setCreatedMoves] = useState<Move[]>([]);

  useEffect(() => {
    async function loadProblem() {
      try {
        setLoading(true);
        setError(null);
        const problem = await fetchProblem(305461);
        setProblem(problem);
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err.message);
        } else {
          setError('An unexpected error occurred');
        }
        console.error('Failed to load problem:', err);
      } finally {
        setLoading(false);
      }
    }

    if (mode === 'view') {
      loadProblem();
    } else {
      setLoading(false);
    }
  }, [mode]);

  const handleClearAll = () => {
    setCreatedMoves([]);
  };

  const handlePredictGrade = () => {
    console.log('Predicting grade for moves:', createdMoves);
    // TODO: Call API to predict grade
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-5xl font-bold text-white mb-2">
          Moonboard Grader
        </h1>
        <p className="text-lg text-gray-400">
          AI-powered climbing grade prediction
        </p>
      </div>

      <TabNavigation activeMode={mode} onModeChange={setMode} />

      {/* View Mode */}
      {mode === 'view' && (
        <>
          {loading && <LoadingSpinner message="Loading problem..." />}
          {error && <ErrorMessage message={error} />}
          {problem && !loading && !error && (
            <MoonBoard problem={problem} mode="view" />
          )}
        </>
      )}

      {/* Create Mode */}
      {mode === 'create' && (
        <div className="flex flex-col items-center gap-6">
          <CreateModeControls
            movesCount={createdMoves.length}
            onClearAll={handleClearAll}
            onPredictGrade={handlePredictGrade}
          />
          <MoonBoard
            moves={createdMoves}
            mode="create"
            onMovesChange={setCreatedMoves}
          />
        </div>
      )}
    </div>
  );
}

export default App;
