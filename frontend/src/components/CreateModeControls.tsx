interface CreateModeControlsProps {
  movesCount: number;
  onClearAll: () => void;
  onPredictGrade: () => void;
}

export default function CreateModeControls({ 
  movesCount, 
  onClearAll, 
  onPredictGrade 
}: CreateModeControlsProps) {
  return (
    <>
      {/* Instructions */}
      <div className="max-w-2xl px-4 text-center">
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
          <p className="text-white text-sm">
            Click on holds to add them to your problem.
            <span className="block mt-2">
              <span className="text-blue-400">1st click</span> = intermediate hold,
              <span className="text-green-400 ml-2">2nd click</span> = start hold,
              <span className="text-red-400 ml-2">3rd click</span> = end hold,
              <span className="text-gray-400 ml-2">4th click</span> = remove
            </span>
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={onClearAll}
          className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg shadow-lg transition-colors duration-200"
        >
          Clear All
        </button>
        <button
          onClick={onPredictGrade}
          disabled={movesCount === 0}
          className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg transition-colors duration-200"
        >
          Predict Grade
        </button>
      </div>
    </>
  );
}

