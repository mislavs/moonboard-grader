import { lazy, Suspense, useState } from 'react';
import TabNavigation from './components/TabNavigation';
import LoadingSpinner from './components/LoadingSpinner';
import BoardSetupSelector from './components/BoardSetupSelector';
import { BoardSetupProvider } from './contexts/BoardSetupContext';

type Mode = 'view' | 'create' | 'analytics';
const ViewMode = lazy(() => import('./components/ViewMode'));
const CreateMode = lazy(() => import('./components/CreateMode'));
const AnalyticsMode = lazy(() => import('./components/AnalyticsMode'));

function App() {
  const [mode, setMode] = useState<Mode>('view');

  return (
    <BoardSetupProvider>
      <div className="min-h-screen bg-gray-900 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">
            Moonboard Grader
          </h1>
          <p className="text-lg text-gray-400">
            AI-powered climbing grade prediction
          </p>
          <div className="mt-4">
            <BoardSetupSelector />
          </div>
        </div>

        <TabNavigation activeMode={mode} onModeChange={setMode} />

        <Suspense
          fallback={(
            <div className="flex justify-center py-12">
              <LoadingSpinner message="Loading mode..." />
            </div>
          )}
        >
          {mode === 'view' && <ViewMode />}
          {mode === 'create' && <CreateMode />}
          {mode === 'analytics' && <AnalyticsMode />}
        </Suspense>
      </div>
    </BoardSetupProvider>
  );
}

export default App;
