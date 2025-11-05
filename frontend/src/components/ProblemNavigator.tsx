import ErrorMessage from './ErrorMessage';
import LoadingSpinner from './LoadingSpinner';
import ProblemListItem from './ProblemListItem';
import PaginationControls from './PaginationControls';
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
  } = useProblems(onFirstProblemLoaded);

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg flex flex-col" style={{ height: '976px' }}>
      {loading && <LoadingSpinner message="Loading problems..." />}
      {error && <ErrorMessage message={error} />}

      {!loading && !error && (
        <>
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
    </div>
  );
}
