import MoonBoard from './components/MoonBoard';
import { sampleProblem } from './data/sampleProblems';

function App() {
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
      <MoonBoard problem={sampleProblem} />
    </div>
  );
}

export default App;
