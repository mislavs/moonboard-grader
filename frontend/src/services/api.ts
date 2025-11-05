/**
 * API service for communicating with the backend
 */

import type { Problem, Move } from '../types/problem';
import type { PredictionResponse, PredictionRequest } from '../types/prediction';
import { API_BASE_URL, ERROR_MESSAGES } from '../config/constants';

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
 * Problem list item (basic info for navigation)
 */
export interface ProblemListItem {
  id: number;
  name: string;
  grade: string;
}

/**
 * Paginated problems response
 */
export interface PaginatedProblemsResponse {
  items: ProblemListItem[];
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
  pageSize: number = 20
): Promise<PaginatedProblemsResponse> {
  return apiFetch<PaginatedProblemsResponse>(
    `${API_BASE_URL}/problems?page=${page}&page_size=${pageSize}`
  );
}

