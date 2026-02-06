"use client";

import { useState, useEffect, useCallback } from 'react';
import type { TradingStatus } from '@/types/trading';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function useTradingData() {
  const [data, setData] = useState<TradingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastFetch, setLastFetch] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/status`, {
        cache: 'no-store',
      });

      if (!res.ok) {
        throw new Error(`HTTP error: ${res.status}`);
      }

      const json = await res.json();
      setData(json);
      setError(null);
      setLastFetch(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchData();

    // Poll every second
    const interval = setInterval(fetchData, 1000);

    return () => clearInterval(interval);
  }, [fetchData]);

  const dataAge = lastFetch
    ? (Date.now() - lastFetch.getTime()) / 1000
    : 999;

  return { data, loading, error, dataAge, refetch: fetchData };
}
