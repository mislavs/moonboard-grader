/**
 * Custom hook for checking backend health
 */

import { useEffect, useState } from 'react';
import { healthCheck } from '../services/api';

export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    async function checkHealth() {
      try {
        await healthCheck();
        setIsHealthy(true);
      } catch (err) {
        console.error('Backend health check failed:', err);
        setIsHealthy(false);
      }
    }

    checkHealth();
  }, []);

  return isHealthy;
}

