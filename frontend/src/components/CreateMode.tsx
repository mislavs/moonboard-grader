import { useState, useEffect } from 'react';
import MoonBoard from './MoonBoard';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import CreateModeControls from './CreateModeControls';
import PredictionDisplay from './PredictionDisplay';
import DuplicateCheckResult from './DuplicateCheckResult';
import { usePrediction } from '../hooks/usePrediction';
import { useDuplicateCheck } from '../hooks/useDuplicateCheck';
import { useBackendHealth } from '../hooks/useBackendHealth';
import { useGeneration } from '../hooks/useGeneration';
import type { Move } from '../types/problem';
import { ERROR_MESSAGES } from '../config/api';
import { AVAILABLE_GRADES } from '../constants/grades';

export default function CreateMode() {
  const [createdMoves, setCreatedMoves] = useState<Move[]>([]);
  const [selectedGrade, setSelectedGrade] = useState<string>(AVAILABLE_GRADES[0]);
  const { prediction, predicting, error, predict, reset } = usePrediction();
  const { duplicate, checking, error: duplicateError, checkForDuplicate, reset: resetDuplicate } = useDuplicateCheck();
  const { generating, error: generateError, generate, reset: resetGenerate } = useGeneration();
  const backendHealthy = useBackendHealth();

  // Clear duplicate result when moves change
  useEffect(() => {
    resetDuplicate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createdMoves]);

  const handleClearAll = () => {
    setCreatedMoves([]);
    reset();
    resetDuplicate();
    resetGenerate();
  };

  const handlePredictGrade = async () => {
    await predict(createdMoves);
  };

  const handleCheckDuplicate = async () => {
    await checkForDuplicate(createdMoves);
  };

  const handleGenerate = async () => {
    // Clear existing state
    reset();
    resetDuplicate();
    resetGenerate();
    
    // Generate new problem with selected grade
    const result = await generate(selectedGrade);
    if (result) {
      setCreatedMoves(result.moves);
    }
  };

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Backend Health Warning */}
      {backendHealthy === false && (
        <ErrorMessage message={ERROR_MESSAGES.BACKEND_UNREACHABLE} />
      )}

      {/* Main Content: Controls on Left, MoonBoard on Right */}
      <div className="flex flex-row gap-8 items-start justify-center w-full px-4">
        {/* Left Panel: Controls and Results */}
        <div className="flex flex-col gap-6 w-96">
          <CreateModeControls
            movesCount={createdMoves.length}
            onClearAll={handleClearAll}
            onPredictGrade={handlePredictGrade}
            onCheckDuplicate={handleCheckDuplicate}
            onGenerate={handleGenerate}
            selectedGrade={selectedGrade}
            onGradeChange={setSelectedGrade}
            isLoading={predicting}
            isCheckingDuplicate={checking}
            isGenerating={generating}
          />

          {generateError && !generating && (
            <ErrorMessage message={generateError} />
          )}

          {generating && (
            <LoadingSpinner message="Generating boulder..." />
          )}

          {duplicateError && !checking && (
            <ErrorMessage message={duplicateError} />
          )}

          {duplicate && !checking && (
            <DuplicateCheckResult
              exists={duplicate.exists}
              problemName={duplicate.problemName}
              problemGrade={duplicate.problemGrade}
            />
          )}

          {checking && (
            <LoadingSpinner message="Checking for duplicates..." />
          )}

          {/* Prediction Error */}
          {error && !predicting && (
            <ErrorMessage message={error} />
          )}

          {/* Prediction Display */}
          {prediction && !predicting && (
            <PredictionDisplay prediction={prediction} />
          )}

          {/* Loading State for Prediction */}
          {predicting && (
            <LoadingSpinner message="Predicting grade..." />
          )}
        </div>

        {/* Right Panel: MoonBoard */}
        <div>
          <MoonBoard
            moves={createdMoves}
            mode="create"
            onMovesChange={setCreatedMoves}
          />
        </div>
      </div>
    </div>
  );
}

