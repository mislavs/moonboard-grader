import { useState, useEffect } from "react";
import MoonBoard from "./MoonBoard";
import LoadingSpinner from "./LoadingSpinner";
import ErrorMessage from "./ErrorMessage";
import CreateModeControls from "./CreateModeControls";
import PredictionDisplay from "./PredictionDisplay";
import DuplicateCheckResult from "./DuplicateCheckResult";
import CruxHighlightToggle from "./CruxHighlightToggle";
import BetaToggle from "./BetaToggle";
import { usePrediction } from "../hooks/usePrediction";
import { useBeta } from "../hooks/useBeta";
import { useDuplicateCheck } from "../hooks/useDuplicateCheck";
import { useBackendHealth } from "../hooks/useBackendHealth";
import { useGeneration } from "../hooks/useGeneration";
import type { Move } from "../types/problem";
import { ERROR_MESSAGES } from "../config/api";
import { AVAILABLE_GRADES } from "../constants/grades";

export default function CreateMode() {
  const [createdMoves, setCreatedMoves] = useState<Move[]>([]);
  const [selectedGrade, setSelectedGrade] = useState<string>(
    AVAILABLE_GRADES[0]
  );
  const [showAttention, setShowAttention] = useState(false);
  const [showBeta, setShowBeta] = useState(false);
  const { prediction, predicting, error, predict, reset } = usePrediction();
  const { beta, loading: betaLoading, fetchBeta, reset: resetBeta } = useBeta();
  const {
    duplicate,
    checking,
    error: duplicateError,
    checkForDuplicate,
    reset: resetDuplicate,
  } = useDuplicateCheck();
  const {
    generating,
    error: generateError,
    generate,
    reset: resetGenerate,
  } = useGeneration();
  const backendHealthy = useBackendHealth();

  // Clear duplicate result and beta when moves change
  useEffect(() => {
    resetDuplicate();
    resetBeta();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createdMoves]);

  const handleClearAll = () => {
    setCreatedMoves([]);
    reset();
    resetDuplicate();
    resetGenerate();
    resetBeta();
  };

  const handleToggleBeta = async (checked: boolean) => {
    setShowBeta(checked);
    if (checked && createdMoves.length > 0 && !beta) {
      await fetchBeta(createdMoves);
    }
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

          {generating && <LoadingSpinner message="Generating boulder..." />}

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

          {checking && <LoadingSpinner message="Checking for duplicates..." />}

          {/* Prediction Error */}
          {error && !predicting && <ErrorMessage message={error} />}

          {/* Prediction Display */}
          {prediction && !predicting && (
            <PredictionDisplay prediction={prediction} />
          )}

          {/* Loading State for Prediction */}
          {predicting && <LoadingSpinner message="Predicting grade..." />}
        </div>

        {/* Right Panel: MoonBoard */}
        <div className="flex flex-col gap-4">
          <MoonBoard
            moves={createdMoves}
            mode="create"
            onMovesChange={setCreatedMoves}
            attentionMap={prediction?.attention_map}
            showAttention={showAttention}
            beta={beta ?? undefined}
            showBeta={showBeta}
          />

          {/* Crux Highlight Toggle */}
          {prediction?.attention_map && (
            <CruxHighlightToggle
              checked={showAttention}
              onChange={setShowAttention}
            />
          )}

          {/* Beta Toggle */}
          {createdMoves.length > 0 && (
            <BetaToggle
              checked={showBeta}
              onChange={handleToggleBeta}
            />
          )}

          {/* Beta Loading */}
          {betaLoading && (
            <LoadingSpinner message="Loading beta..." />
          )}
        </div>
      </div>
    </div>
  );
}
