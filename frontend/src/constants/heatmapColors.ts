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

