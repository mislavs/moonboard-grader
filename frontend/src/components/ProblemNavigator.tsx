import ErrorMessage from './ErrorMessage';
import LoadingSpinner from './LoadingSpinner';
import ProblemListItem from './ProblemListItem';
import PaginationControls from './PaginationControls';
import BenchmarkFilter from './BenchmarkFilter';
import GradeFilter from './GradeFilter';
import { useProblems } from '../hooks/useProblems';

interface ProblemNavigatorProps {
  selectedProblemId: number | null;
  onProblemSelect: (problemId: number) => void;
  onFirstProblemLoaded?: (problemId: number) => void;
}

export default function ProblemNavigator({
  selectedProblemId,
  onProblemSelect,
  onFirstProblemLoaded,
}: ProblemNavigatorProps) {
  const {
    problems,
    loading,
    error,
    page,
    totalPages,
    goToNextPage,
    goToPreviousPage,
    canGoNext,
    canGoPrevious,
    benchmarkFilter,
    setBenchmarkFilter,
    gradeFrom,
    gradeTo,
    setGradeFrom,
    setGradeTo,
  } = useProblems(onFirstProblemLoaded);

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg flex flex-col relative" style={{ height: '976px' }}>
      {error && <ErrorMessage message={error} />}

      {!error && (
        <>
          <GradeFilter
            gradeFrom={gradeFrom}
            gradeTo={gradeTo}
            onGradeFromChange={setGradeFrom}
            onGradeToChange={setGradeTo}
          />

          <BenchmarkFilter
            benchmarkFilter={benchmarkFilter}
            onFilterChange={setBenchmarkFilter}
          />

          <div className="space-y-2 overflow-y-auto mb-4 flex-1 min-h-0">
            {problems.map((problem) => (
              <ProblemListItem
                key={problem.id}
                problem={problem}
                isSelected={selectedProblemId === problem.id}
                onSelect={onProblemSelect}
              />
            ))}
          </div>

          <PaginationControls
            currentPage={page}
            totalPages={totalPages}
            onPrevious={goToPreviousPage}
            onNext={goToNextPage}
            canGoPrevious={canGoPrevious}
            canGoNext={canGoNext}
          />
        </>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 bg-black/40 rounded-lg flex items-center justify-center z-10">
          <LoadingSpinner message="Loading problems..." />
        </div>
      )}
    </div>
  );
}
