"use client";

import { useState, useEffect, useCallback } from "react";
import type { Signal, SignalStats } from "@/types/signals";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SignalFilters {
  page: number;
  limit: number;
  type: string;
  executed: string;
  startDate: string;
  endDate: string;
}

export function useSignals(filters: SignalFilters) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  const fetchSignals = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(filters.page),
        limit: String(filters.limit),
        type: filters.type,
        executed: filters.executed,
      });
      if (filters.startDate) params.set("start_date", filters.startDate);
      if (filters.endDate) params.set("end_date", filters.endDate);

      const res = await fetch(`${API_URL}/api/signals?${params}`);
      const json = await res.json();
      setSignals(json.signals || []);
      setTotal(json.total || 0);
    } catch {
      setSignals([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [filters.page, filters.limit, filters.type, filters.executed, filters.startDate, filters.endDate]);

  useEffect(() => {
    fetchSignals();
  }, [fetchSignals]);

  return { signals, total, loading, refetch: fetchSignals };
}

export function useSignalStats(hours: number = 24) {
  const [stats, setStats] = useState<SignalStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/signals/stats?hours=${hours}`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, [hours]);

  return { stats, loading };
}
