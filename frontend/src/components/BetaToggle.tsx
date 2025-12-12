/**
 * BetaToggle component
 *
 * Checkbox to toggle the beta overlay showing which hand to use on each hold.
 */

interface BetaToggleProps {
  /** Whether the beta is currently shown */
  checked: boolean;
  /** Callback when toggle state changes */
  onChange: (checked: boolean) => void;
}

export default function BetaToggle({
  checked,
  onChange,
}: BetaToggleProps) {
  return (
    <div className="bg-gray-800 rounded-lg px-4 py-3 border border-gray-700">
      <label className="flex items-center gap-3 text-gray-300 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-green-500 focus:ring-green-500 focus:ring-offset-gray-800 cursor-pointer"
        />
        <span className="text-sm font-medium">Show Beta</span>
      </label>

      {/* Legend - only shown when beta is enabled */}
      {checked && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="text-xs text-gray-400 mb-2">
            Hand Assignments
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-blue-400">L</span>
              <span className="text-xs text-gray-500">Left Hand</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-orange-400">R</span>
              <span className="text-xs text-gray-500">Right Hand</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
