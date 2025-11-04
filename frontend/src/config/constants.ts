/**
 * Application configuration constants
 */

export const API_BASE_URL = '/api';

export const DEFAULT_PROBLEM_ID = 305461;

export const PREDICTION_TOP_K = 4;

export const ERROR_MESSAGES = {
  BACKEND_UNREACHABLE: '⚠️ Backend is not reachable. Make sure the API server is running on http://localhost:8000',
  SERVER_CONNECTION_FAILED: 'Failed to connect to the server. Please ensure the backend is running.',
  PREDICTION_FAILED: 'An unexpected error occurred during prediction',
  PROBLEM_LOAD_FAILED: 'An unexpected error occurred',
} as const;

