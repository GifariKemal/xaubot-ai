"use client";

import { useState, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface FeatureImportance {
  name: string;
  importance: number;
}

export interface ModelMetrics {
  featureImportance: FeatureImportance[];
  trainAuc: number;
  testAuc: number;
  sampleCount: number;
  updatedAt: string | null;
}

export interface TrainingRun {
  id: number;
  started_at: string;
  completed_at: string;
  train_auc: number;
  test_auc: number;
  sample_count: number;
  features_used: number;
  trigger_reason: string;
}

export interface RegimeDistribution {
  regime: string;
  count: number;
}

export function useModelMetrics() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/model/metrics`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setMetrics(null))
      .finally(() => setLoading(false));
  }, []);

  return { metrics, loading };
}

export function useTrainingHistory() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/model/training-history`)
      .then((r) => r.json())
      .then((data) => setRuns(data.runs || []))
      .catch(() => setRuns([]))
      .finally(() => setLoading(false));
  }, []);

  return { runs, loading };
}

export function useRegimeDistribution() {
  const [distribution, setDistribution] = useState<RegimeDistribution[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/api/model/regime-distribution`)
      .then((r) => r.json())
      .then((data) => setDistribution(data.distribution || []))
      .catch(() => setDistribution([]))
      .finally(() => setLoading(false));
  }, []);

  return { distribution, loading };
}
