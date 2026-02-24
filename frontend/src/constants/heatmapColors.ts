/**
 * Shared color constants for the crux highlight heatmap
 */

/** Color for low attention/easier areas (blue) */
export const HEATMAP_COLOR_LOW = "rgb(0,0,255)";

/** Color for medium attention (white) */
export const HEATMAP_COLOR_MID = "rgb(255,255,255)";

/** Color for high attention/harder areas (red) */
export const HEATMAP_COLOR_HIGH = "rgb(255,0,0)";

/** CSS gradient for the legend */
export const HEATMAP_GRADIENT = `linear-gradient(to right, ${HEATMAP_COLOR_LOW}, ${HEATMAP_COLOR_MID}, ${HEATMAP_COLOR_HIGH})`;

/**
 * Convert normalized value (0-1) to a color on a blue-white-red scale.
 */
export function getHeatmapColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));

  if (v < 0.5) {
    const t = v * 2;
    const r = Math.round(255 * t);
    const g = Math.round(255 * t);
    const b = 255;
    return `rgb(${r},${g},${b})`;
  }

  const t = (v - 0.5) * 2;
  const r = 255;
  const g = Math.round(255 * (1 - t));
  const b = Math.round(255 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

