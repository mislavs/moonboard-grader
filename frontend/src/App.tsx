import { useState } from 'react';
import TabNavigation from './components/TabNavigation';
import ViewMode from './components/ViewMode';
import CreateMode from './components/CreateMode';

type Mode = 'view' | 'create';

function App() {
  const [mode, setMode] = useState<Mode>('view');

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

      {mode === 'view' ? <ViewMode /> : <CreateMode />}
    </div>
  );
}

export default App;
