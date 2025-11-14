/**
 * API service for communicating with the backend
 */

import type { Problem, Move } from '../types/problem';
import type { PredictionResponse, PredictionRequest } from '../types/prediction';
import { API_BASE_URL, ERROR_MESSAGES } from '../config/api';

export class ApiError extends Error {
  statusCode?: number;
  details?: unknown;

  constructor(
    message: string,
    statusCode?: number,
    details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new ApiError(
        errorData?.detail || `Request failed: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    throw new ApiError(
      ERROR_MESSAGES.SERVER_CONNECTION_FAILED,
      undefined,
      error
    );
  }
}

/**
 * Fetch a problem by its ID from the backend
 */
export async function fetchProblem(id: number): Promise<Problem> {
  return apiFetch<Problem>(`${API_BASE_URL}/problems/${id}`);
}

/**
 * Predict the grade of a climbing problem
 */
export async function predictGrade(
  moves: Move[],
  topK: number = 3
): Promise<PredictionResponse> {
  const requestBody: PredictionRequest = {
    moves,
    top_k: topK,
  };

  return apiFetch<PredictionResponse>(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });
}

/**
 * Health check to verify backend connection
 */
export async function healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
  return apiFetch<{ status: string; model_loaded: boolean }>(`${API_BASE_URL}/health`);
}

/**
 * Problem summary (basic info for navigation)
 */
export interface ProblemSummary {
  id: number;
  name: string;
  grade: string;
  isBenchmark: boolean;
}

/**
 * Paginated problems response
 */
export interface PaginatedProblemsResponse {
  items: ProblemSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/**
 * Fetch a paginated list of problems
 */
export async function fetchProblems(
  page: number = 1,
  pageSize: number = 20,
  benchmarksOnly?: boolean | null,
  gradeFrom?: string | null,
  gradeTo?: string | null
): Promise<PaginatedProblemsResponse> {
  let url = `${API_BASE_URL}/problems?page=${page}&page_size=${pageSize}`;
  
  if (benchmarksOnly !== null && benchmarksOnly !== undefined) {
    url += `&benchmarks_only=${benchmarksOnly}`;
  }
  
  if (gradeFrom) {
    url += `&grade_from=${encodeURIComponent(gradeFrom)}`;
  }
  
  if (gradeTo) {
    url += `&grade_to=${encodeURIComponent(gradeTo)}`;
  }
  
  return apiFetch<PaginatedProblemsResponse>(url);
}

/**
 * Duplicate check response
 */
export interface DuplicateCheckResponse {
  exists: boolean;
  problem_id: number | null;
}

/**
 * Check if a problem with the same moves already exists
 */
export async function checkDuplicate(moves: Move[]): Promise<DuplicateCheckResponse> {
  return apiFetch<DuplicateCheckResponse>(`${API_BASE_URL}/problems/check-duplicate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ moves }),
  });
}

/**
 * Problem statistics from generation
 */
export interface ProblemStats {
  num_moves: number;
  num_start_holds: number;
  num_end_holds: number;
}

/**
 * Generate response
 */
export interface GenerateResponse {
  moves: Move[];
  grade: string;
  stats: ProblemStats;
}

/**
 * Generate a new climbing problem
 */
export async function generateProblem(
  grade: string = '6A+',
  temperature: number = 1.0
): Promise<GenerateResponse> {
  return apiFetch<GenerateResponse>(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ grade, temperature }),
  });
}

