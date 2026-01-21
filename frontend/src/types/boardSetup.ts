/**
 * Configuration for a specific wall angle within a hold setup.
 */
export interface AngleConfig {
  /** Wall angle in degrees (e.g., 25, 40) */
  angle: number;
  /** Whether a trained model exists for this configuration */
  hasModel: boolean;
  /** Whether this is the default configuration */
  isDefault: boolean;
}

/**
 * Configuration for a hold setup (e.g., MoonBoard Masters 2017).
 */
export interface HoldSetup {
  /** Unique identifier for the hold setup */
  id: string;
  /** Display name for the hold setup */
  name: string;
  /** Available wall angles for this hold setup */
  angles: AngleConfig[];
}

/**
 * Response from the /board-setups API endpoint.
 */
export interface BoardSetupsResponse {
  /** List of available hold setups */
  holdSetups: HoldSetup[];
}

/**
 * Currently selected board configuration.
 */
export interface SelectedBoardConfig {
  /** Selected hold setup ID */
  holdSetupId: string;
  /** Selected wall angle in degrees */
  angle: number;
}
