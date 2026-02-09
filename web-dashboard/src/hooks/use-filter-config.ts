import { useState, useEffect } from 'react';
import type { FilterConfigData, FilterUpdate } from '@/types/filters';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export function useFilterConfig() {
  const [config, setConfig] = useState<FilterConfigData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updating, setUpdating] = useState(false);

  const fetchConfig = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/filters/config`);
      if (!response.ok) throw new Error('Failed to fetch filter config');
      const data = await response.json();
      setConfig(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const updateFilters = async (updates: FilterUpdate) => {
    try {
      setUpdating(true);
      const response = await fetch(`${API_URL}/api/filters/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });

      if (!response.ok) throw new Error('Failed to update filters');

      const result = await response.json();
      if (result.success) {
        // Refresh config after successful update
        await fetchConfig();
      }
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Update failed');
      throw err;
    } finally {
      setUpdating(false);
    }
  };

  const toggleFilter = async (filterKey: string) => {
    if (!config) return;

    const currentState = config.filters[filterKey]?.enabled ?? true;
    await updateFilters({ [filterKey]: !currentState });
  };

  useEffect(() => {
    fetchConfig();
    // Refresh every 30 seconds
    const interval = setInterval(fetchConfig, 30000);
    return () => clearInterval(interval);
  }, []);

  return {
    config,
    loading,
    error,
    updating,
    toggleFilter,
    updateFilters,
    refresh: fetchConfig,
  };
}
