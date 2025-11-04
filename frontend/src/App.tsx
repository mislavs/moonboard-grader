import { useEffect, useState } from 'react';
import MoonBoard from './components/MoonBoard';
import { fetchProblem, ApiError } from './services/api';
import type { Problem } from './types/problem';

function App() {
  const [problem, setProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

    loadProblem();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="text-center mb-8">
        <h1 className="text-5xl font-bold text-white mb-2">
          Moonboard Grader
        </h1>
        <p className="text-lg text-gray-400">
          AI-powered climbing grade prediction
        </p>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-20">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4"></div>
            <p className="text-white text-lg">Loading problem...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="max-w-2xl mx-auto px-4">
          <div className="bg-red-900 border border-red-700 text-red-100 px-6 py-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Oops! Something went wrong</h3>
            <p>{error}</p>
            <p className="mt-2 text-sm text-red-200">
              Please check your connection and try refreshing the page.
            </p>
          </div>
        </div>
      )}

      {problem && !loading && !error && (
        <MoonBoard problem={problem} />
      )}
    </div>
  );
}

export default App;
