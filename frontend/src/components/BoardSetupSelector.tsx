/**
 * BoardSetupSelector component
 *
 * Provides two dropdowns for selecting the board setup (hold configuration)
 * and wall angle. Used in the header to allow users to switch between
 * different Moonboard configurations.
 */

import { useBoardSetup } from '../contexts/BoardSetupContext';

export default function BoardSetupSelector() {
  const {
    holdSetups,
    currentHoldSetup,
    currentAngle,
    setHoldSetup,
    setAngle,
    loading,
    error,
  } = useBoardSetup();

  // Don't render if there are no setups or only one setup with one angle
  if (loading) {
    return (
      <div className="flex items-center gap-2 text-gray-400 text-sm">
        Loading configurations...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-400 text-sm">
        {error}
      </div>
    );
  }

  // If there's only one setup with one angle, just display it as a subtle badge
  const totalAngles = holdSetups.reduce((sum, s) => sum + s.angles.length, 0);
  if (holdSetups.length === 1 && totalAngles === 1 && currentHoldSetup && currentAngle) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-800 rounded-full text-sm">
        <span className="text-gray-300">{currentHoldSetup.name}</span>
        <span className="text-gray-500">|</span>
        <span className="text-gray-400">{currentAngle.angle}°</span>
      </div>
    );
  }

  if (!currentHoldSetup || !currentAngle) {
    return null;
  }

  const availableAngles = currentHoldSetup.angles;

  return (
    <div className="flex items-center gap-4 flex-wrap justify-center">
      {/* Hold Setup Dropdown - only show if multiple setups */}
      {holdSetups.length > 1 && (
        <div className="flex items-center gap-2">
          <label className="text-gray-400 text-sm">Board:</label>
          <select
            value={currentHoldSetup.id}
            onChange={(e) => setHoldSetup(e.target.value)}
            className="bg-gray-700 text-white rounded-lg px-3 py-2 text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
          >
            {holdSetups.map((setup) => (
              <option key={setup.id} value={setup.id}>
                {setup.name}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Angle Dropdown - only show if multiple angles */}
      {availableAngles.length > 1 && (
        <div className="flex items-center gap-2">
          <label className="text-gray-400 text-sm">Angle:</label>
          <select
            value={currentAngle.angle}
            onChange={(e) => setAngle(Number(e.target.value))}
            className="bg-gray-700 text-white rounded-lg px-3 py-2 text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
          >
            {availableAngles.map((angle) => (
              <option key={angle.angle} value={angle.angle}>
                {angle.angle}°{!angle.hasModel && ' (no model)'}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Show current selection if only one of each */}
      {holdSetups.length === 1 && availableAngles.length === 1 && (
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-gray-800 rounded-full text-sm">
          <span className="text-gray-300">{currentHoldSetup.name}</span>
          <span className="text-gray-500">|</span>
          <span className="text-gray-400">{currentAngle.angle}°</span>
        </div>
      )}
    </div>
  );
}
