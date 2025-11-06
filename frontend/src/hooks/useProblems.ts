import { useEffect, useState, useCallback } from 'react';
import { fetchProblems, type ProblemSummary, ApiError } from '../services/api';

const DEFAULT_PAGE_SIZE = 20;

interface UseProblemsReturn {
  problems: ProblemSummary[];
  loading: boolean;
  error: string | null;
  page: number;
  totalPages: number;
  goToNextPage: () => void;
  goToPreviousPage: () => void;
  canGoNext: boolean;
  canGoPrevious: boolean;
  benchmarkFilter: boolean | null;
  setBenchmarkFilter: (filter: boolean | null) => void;
}

export function useProblems(
  onFirstLoad?: (firstProblemId: number) => void
): UseProblemsReturn {
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [benchmarkFilter, setBenchmarkFilter] = useState<boolean | null>(null);

  useEffect(() => {
    async function loadProblems() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetchProblems(page, DEFAULT_PAGE_SIZE, benchmarkFilter);
        
        setProblems(response.items);
        setTotalPages(response.total_pages);

        // Auto-select first problem on initial load
        if (page === 1 && response.items.length > 0 && onFirstLoad) {
          onFirstLoad(response.items[0].id);
        }
      } catch (err) {
        const errorMessage =
          err instanceof ApiError ? err.message : 'Failed to load problems';
        setError(errorMessage);
        console.error('Failed to load problems:', err);
      } finally {
        setLoading(false);
      }
    }

    loadProblems();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, benchmarkFilter]);

  const goToNextPage = useCallback(() => {
    setPage((prev) => Math.min(prev + 1, totalPages));
  }, [totalPages]);

  const goToPreviousPage = useCallback(() => {
    setPage((prev) => Math.max(prev - 1, 1));
  }, []);

  const handleSetBenchmarkFilter = useCallback((filter: boolean | null) => {
    setBenchmarkFilter(filter);
    setPage(1); // Reset to first page when filter changes
  }, []);

  return {
    problems,
    loading,
    error,
    page,
    totalPages,
    goToNextPage,
    goToPreviousPage,
    canGoNext: page < totalPages,
    canGoPrevious: page > 1,
    benchmarkFilter,
    setBenchmarkFilter: handleSetBenchmarkFilter,
  };
}

