/**
 * CruxHighlightToggle component
 *
 * Checkbox to toggle the crux highlight overlay with a legend
 * explaining the color scale.
 */

import { HEATMAP_GRADIENT } from "../constants/heatmapColors";

interface CruxHighlightToggleProps {
  /** Whether the highlight is currently shown */
  checked: boolean;
  /** Callback when toggle state changes */
  onChange: (checked: boolean) => void;
}

export default function CruxHighlightToggle({
  checked,
  onChange,
}: CruxHighlightToggleProps) {
  return (
    <div className="bg-gray-800 rounded-lg px-4 py-3 border border-gray-700">
      <label className="flex items-center gap-3 text-gray-300 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-green-500 focus:ring-offset-gray-800 cursor-pointer"
        />
        <span className="text-sm font-medium">Show Crux Highlight</span>
      </label>

      {/* Legend - only shown when highlight is enabled */}
      {checked && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="text-xs text-gray-400 mb-2">
            Predicted Difficulty by Area
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">Easier</span>
            <div
              className="flex-1 h-4 rounded"
              style={{ background: HEATMAP_GRADIENT }}
            />
            <span className="text-xs text-gray-500">Harder</span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Red areas likely contain the crux moves
          </p>
        </div>
      )}
    </div>
  );
}

