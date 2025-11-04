import type { PredictionResponse } from '../types/prediction';

interface PredictionDisplayProps {
  prediction: PredictionResponse;
}

export default function PredictionDisplay({ prediction }: PredictionDisplayProps) {
  return (
    <div className="w-full max-w-md bg-gray-800 rounded-lg p-6 shadow-2xl border-2 border-green-500">
      {/* Main Prediction */}
      <div className="text-center mb-6">
        <h2 className="text-gray-400 text-sm uppercase tracking-wider mb-2">
          Predicted Grade
        </h2>
        <div className="text-6xl font-bold text-green-400 mb-3">
          {prediction.predicted_grade}
        </div>
        <div className="text-xl text-gray-300">
          Confidence: <span className="font-semibold text-green-400">
            {(prediction.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Divider */}
      {prediction.top_k_predictions.length > 1 && (
        <div className="border-t border-gray-700 my-4"></div>
      )}

      {/* Alternative Predictions (excluding the main prediction) */}
      {prediction.top_k_predictions.length > 1 && (
        <div>
          <h3 className="text-gray-400 text-sm uppercase tracking-wider mb-3">
            Alternative Predictions
          </h3>
          <div className="space-y-2">
            {prediction.top_k_predictions.slice(1).map((pred, index) => (
              <div
                key={index}
                className="flex items-center justify-between bg-gray-700 rounded px-4 py-2"
              >
                <div className="flex items-center gap-3">
                  <span className="text-gray-500 font-mono text-sm">
                    #{index + 2}
                  </span>
                  <span className="text-white font-semibold">
                    {pred.grade}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-600 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-green-500 h-full transition-all duration-300"
                      style={{ width: `${pred.probability * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-gray-300 text-sm font-mono w-12 text-right">
                    {(pred.probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

