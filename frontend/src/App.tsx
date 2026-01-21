import { useState } from 'react';
import TabNavigation from './components/TabNavigation';
import ViewMode from './components/ViewMode';
import CreateMode from './components/CreateMode';
import AnalyticsMode from './components/AnalyticsMode';
import BoardSetupSelector from './components/BoardSetupSelector';
import { BoardSetupProvider } from './contexts/BoardSetupContext';

type Mode = 'view' | 'create' | 'analytics';

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

        {mode === 'view' && <ViewMode />}
        {mode === 'create' && <CreateMode />}
        {mode === 'analytics' && <AnalyticsMode />}
      </div>
    </BoardSetupProvider>
  );
}

export default App;
