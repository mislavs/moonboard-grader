/**
 * API service for communicating with the backend
 */

import type { Problem } from '../types/problem';

const API_BASE_URL = '/api';

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Fetch a problem by its ID from the backend
 */
export async function fetchProblem(id: number): Promise<Problem> {
  try {
    const response = await fetch(`${API_BASE_URL}/problems/${id}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new ApiError(
        errorData?.detail || `Failed to fetch problem: ${response.statusText}`,
        response.status,
        errorData
      );
    }
    
    const data = await response.json();
    return data as Problem;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    
    // Network errors or other issues
    throw new ApiError(
      'Failed to connect to the server. Please ensure the backend is running.',
      undefined,
      error
    );
  }
}

