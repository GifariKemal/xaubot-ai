"use client";

import { useState, useEffect, useCallback } from "react";
import type { Trade, TradeStats, EquityCurvePoint } from "@/types/trades";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TradeFilters {
  page: number;
  limit: number;
  direction: string;
  startDate: string;
  endDate: string;
}

export function useTrades(filters: TradeFilters) {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  const fetchTrades = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(filters.page),
        limit: String(filters.limit),
        direction: filters.direction,
      });
      if (filters.startDate) params.set("start_date", filters.startDate);
      if (filters.endDate) params.set("end_date", filters.endDate);

      const res = await fetch(`${API_URL}/api/trades?${params}`);
      const json = await res.json();
      setTrades(json.trades || []);
      setTotal(json.total || 0);
    } catch {
      setTrades([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [filters.page, filters.limit, filters.direction, filters.startDate, filters.endDate]);

  useEffect(() => {
    fetchTrades();
  }, [fetchTrades]);

  return { trades, total, loading, refetch: fetchTrades };
}

export function useTradeStats(startDate: string, endDate: string) {
  const [stats, setStats] = useState<TradeStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams();
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);

    fetch(`${API_URL}/api/trades/stats?${params}`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, [startDate, endDate]);

  return { stats, loading };
}

export function useEquityCurve(startDate: string, endDate: string) {
  const [points, setPoints] = useState<EquityCurvePoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams();
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);

    fetch(`${API_URL}/api/trades/equity-curve?${params}`)
      .then((r) => r.json())
      .then((data) => setPoints(data.points || []))
      .catch(() => setPoints([]))
      .finally(() => setLoading(false));
  }, [startDate, endDate]);

  return { points, loading };
}
