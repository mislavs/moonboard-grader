/**
 * Font climbing grades from easiest to hardest.
 * 
 * This list matches the backend's grade_encoder.py to ensure
 * consistency across the application.
 */
export const FONT_GRADES = [
  "5+",
  "6A", "6A+", "6B", "6B+", "6C", "6C+",
  "7A", "7A+", "7B", "7B+", "7C", "7C+",
  "8A", "8A+", "8B", "8B+", "8C", "8C+"
] as const;

export type FontGrade = typeof FONT_GRADES[number];

/**
 * Available grades for problem generation (filtered subset matching the trained model).
 * These correspond to the grades the generator model was trained on (6A+ to 7C).
 */
export const AVAILABLE_GRADES = [
  "6A+", "6B", "6B+", "6C", "6C+", 
  "7A", "7A+", "7B", "7B+", "7C"
] as const;

export type AvailableGrade = typeof AVAILABLE_GRADES[number];

