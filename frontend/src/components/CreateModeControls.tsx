import SpinnerIcon from './SpinnerIcon';
import GradeSelector from './GradeSelector';

interface CreateModeControlsProps {
  movesCount: number;
  onClearAll: () => void;
  onPredictGrade: () => void;
  onCheckDuplicate: () => void;
  onGenerate: () => void;
  selectedGrade: string;
  onGradeChange: (grade: string) => void;
  isLoading?: boolean;
  isCheckingDuplicate?: boolean;
  isGenerating?: boolean;
}

export default function CreateModeControls({ 
  movesCount, 
  onClearAll, 
  onPredictGrade,
  onCheckDuplicate,
  onGenerate,
  selectedGrade,
  onGradeChange,
  isLoading = false,
  isCheckingDuplicate = false,
  isGenerating = false
}: CreateModeControlsProps) {
  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Instructions */}
      <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
        <h3 className="text-white font-semibold mb-2">How to Create</h3>
        <p className="text-gray-300 text-sm leading-relaxed">
          Click on holds to add them to your problem:
        </p>
        <ul className="mt-3 space-y-2 text-sm">
          <li className="flex items-center gap-2">
            <span className="text-blue-400 font-semibold">1st click:</span>
            <span className="text-gray-300">Intermediate hold</span>
          </li>
          <li className="flex items-center gap-2">
            <span className="text-green-400 font-semibold">2nd click:</span>
            <span className="text-gray-300">Start hold</span>
          </li>
          <li className="flex items-center gap-2">
            <span className="text-red-400 font-semibold">3rd click:</span>
            <span className="text-gray-300">End hold</span>
          </li>
          <li className="flex items-center gap-2">
            <span className="text-gray-400 font-semibold">4th click:</span>
            <span className="text-gray-300">Remove</span>
          </li>
        </ul>
      </div>

      {/* Grade Selection */}
      <GradeSelector
        selectedGrade={selectedGrade}
        onGradeChange={onGradeChange}
      />

      {/* Action Buttons */}
      <div className="flex flex-col gap-3 w-full">
        <button
          onClick={onGenerate}
          disabled={isGenerating}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg transition-colors duration-200"
        >
          {isGenerating ? (
            <span className="flex items-center justify-center gap-2">
              <SpinnerIcon />
              Generating...
            </span>
          ) : (
            'Generate Boulder'
          )}
        </button>
        <button
          onClick={onPredictGrade}
          disabled={movesCount === 0 || isLoading}
          className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg transition-colors duration-200"
        >
          {isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <SpinnerIcon />
              Predicting...
            </span>
          ) : (
            'Predict Grade'
          )}
        </button>
        <button
          onClick={onCheckDuplicate}
          disabled={movesCount === 0 || isCheckingDuplicate}
          className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg transition-colors duration-200"
        >
          {isCheckingDuplicate ? (
            <span className="flex items-center justify-center gap-2">
              <SpinnerIcon />
              Checking...
            </span>
          ) : (
            'Check Duplicate'
          )}
        </button>
        <button
          onClick={onClearAll}
          disabled={isLoading || isCheckingDuplicate}
          className="w-full px-6 py-3 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg transition-colors duration-200"
        >
          Clear All
        </button>
      </div>
    </div>
  );
}
