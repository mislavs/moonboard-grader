interface PaginationControlsProps {
  currentPage: number;
  totalPages: number;
  onPrevious: () => void;
  onNext: () => void;
  canGoPrevious: boolean;
  canGoNext: boolean;
}

export default function PaginationControls({
  currentPage,
  totalPages,
  onPrevious,
  onNext,
  canGoPrevious,
  canGoNext,
}: PaginationControlsProps) {
  return (
    <div className="flex items-center justify-between pt-4 border-t border-gray-700">
      <PaginationButton
        onClick={onPrevious}
        disabled={!canGoPrevious}
        label="Previous"
      />

      <span className="text-gray-300">
        Page {currentPage} of {totalPages}
      </span>

      <PaginationButton
        onClick={onNext}
        disabled={!canGoNext}
        label="Next"
      />
    </div>
  );
}

interface PaginationButtonProps {
  onClick: () => void;
  disabled: boolean;
  label: string;
}

function PaginationButton({ onClick, disabled, label }: PaginationButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 rounded font-medium transition-colors ${
        disabled
          ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
          : 'bg-blue-600 text-white hover:bg-blue-700'
      }`}
    >
      {label}
    </button>
  );
}

