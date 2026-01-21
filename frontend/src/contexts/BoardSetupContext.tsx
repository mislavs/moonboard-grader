import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import type {
  HoldSetup,
  AngleConfig,
  BoardSetupsResponse,
} from '../types/boardSetup';
import { fetchBoardSetups } from '../services/api';

const STORAGE_KEY = 'moonboard-selected-setup';

interface BoardSetupContextType {
  /** All available hold setups */
  holdSetups: HoldSetup[];
  /** Currently selected hold setup */
  currentHoldSetup: HoldSetup | null;
  /** Currently selected angle configuration */
  currentAngle: AngleConfig | null;
  /** Set the current hold setup by ID */
  setHoldSetup: (setupId: string) => void;
  /** Set the current angle */
  setAngle: (angle: number) => void;
  /** Whether the context is loading */
  loading: boolean;
  /** Error message if loading failed */
  error: string | null;
}

const BoardSetupContext = createContext<BoardSetupContextType | null>(null);

interface BoardSetupProviderProps {
  children: ReactNode;
}

interface StoredSelection {
  holdSetupId: string;
  angle: number;
}

function loadStoredSelection(): StoredSelection | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch {
    // Ignore localStorage errors
  }
  return null;
}

function saveSelection(selection: StoredSelection): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
  } catch {
    // Ignore localStorage errors
  }
}

function findDefault(holdSetups: HoldSetup[]): { setup: HoldSetup; angle: AngleConfig } | null {
  for (const setup of holdSetups) {
    for (const angle of setup.angles) {
      if (angle.isDefault) {
        return { setup, angle };
      }
    }
  }
  // Fallback to first setup and first angle if no default found
  if (holdSetups.length > 0 && holdSetups[0].angles.length > 0) {
    return { setup: holdSetups[0], angle: holdSetups[0].angles[0] };
  }
  return null;
}

export function BoardSetupProvider({ children }: BoardSetupProviderProps) {
  const [holdSetups, setHoldSetups] = useState<HoldSetup[]>([]);
  const [currentHoldSetup, setCurrentHoldSetup] = useState<HoldSetup | null>(null);
  const [currentAngle, setCurrentAngle] = useState<AngleConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load board setups on mount
  useEffect(() => {
    async function loadSetups() {
      try {
        setLoading(true);
        setError(null);
        const response: BoardSetupsResponse = await fetchBoardSetups();
        setHoldSetups(response.holdSetups);

        // Try to restore previous selection from localStorage
        const stored = loadStoredSelection();
        let selectedSetup: HoldSetup | null = null;
        let selectedAngle: AngleConfig | null = null;

        if (stored) {
          const setup = response.holdSetups.find(s => s.id === stored.holdSetupId);
          if (setup) {
            const angle = setup.angles.find(a => a.angle === stored.angle);
            if (angle) {
              selectedSetup = setup;
              selectedAngle = angle;
            }
          }
        }

        // Fall back to default if stored selection not found
        if (!selectedSetup || !selectedAngle) {
          const defaultConfig = findDefault(response.holdSetups);
          if (defaultConfig) {
            selectedSetup = defaultConfig.setup;
            selectedAngle = defaultConfig.angle;
          }
        }

        setCurrentHoldSetup(selectedSetup);
        setCurrentAngle(selectedAngle);
      } catch (err) {
        console.error('Failed to load board setups:', err);
        setError('Failed to load board configurations');
      } finally {
        setLoading(false);
      }
    }

    loadSetups();
  }, []);

  // Save selection to localStorage when it changes
  useEffect(() => {
    if (currentHoldSetup && currentAngle) {
      saveSelection({
        holdSetupId: currentHoldSetup.id,
        angle: currentAngle.angle,
      });
    }
  }, [currentHoldSetup, currentAngle]);

  const setHoldSetup = useCallback((setupId: string) => {
    const setup = holdSetups.find(s => s.id === setupId);
    if (setup) {
      setCurrentHoldSetup(setup);
      // Reset angle to first available angle for this setup
      if (setup.angles.length > 0) {
        // Prefer the angle that was previously selected if available
        const prevAngle = currentAngle?.angle;
        const matchingAngle = setup.angles.find(a => a.angle === prevAngle);
        setCurrentAngle(matchingAngle || setup.angles[0]);
      } else {
        setCurrentAngle(null);
      }
    }
  }, [holdSetups, currentAngle]);

  const setAngle = useCallback((angle: number) => {
    if (currentHoldSetup) {
      const angleConfig = currentHoldSetup.angles.find(a => a.angle === angle);
      if (angleConfig) {
        setCurrentAngle(angleConfig);
      }
    }
  }, [currentHoldSetup]);

  return (
    <BoardSetupContext.Provider
      value={{
        holdSetups,
        currentHoldSetup,
        currentAngle,
        setHoldSetup,
        setAngle,
        loading,
        error,
      }}
    >
      {children}
    </BoardSetupContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useBoardSetup(): BoardSetupContextType {
  const context = useContext(BoardSetupContext);
  if (!context) {
    throw new Error('useBoardSetup must be used within a BoardSetupProvider');
  }
  return context;
}
