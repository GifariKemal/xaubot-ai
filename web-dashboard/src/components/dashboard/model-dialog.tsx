"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Brain, BarChart3, Clock, Layers } from "lucide-react";
import {
  useModelMetrics,
  useTrainingHistory,
  useRegimeDistribution,
} from "@/hooks/use-model-insights";
import { formatUSD } from "@/lib/utils";

function FeatureImportanceBars() {
  const { metrics, loading } = useModelMetrics();

  if (loading || !metrics?.featureImportance?.length) {
    return <p className="text-sm text-muted-foreground">No feature data available</p>;
  }

  const features = metrics.featureImportance.slice(0, 15);
  const max = Math.max(...features.map((f) => f.importance));

  return (
    <div className="space-y-1.5">
      {features.map((f, i) => (
        <div key={f.name} className="flex items-center gap-2 text-xs">
          <span className="w-32 text-muted-foreground truncate text-right">{f.name}</span>
          <div className="flex-1 h-3 bg-surface-light rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bar-purple bar-animate-in"
              style={{ width: `${(f.importance / max) * 100}%`, animationDelay: `${i * 30}ms` }}
            />
          </div>
          <span className="w-10 text-right font-number">{(f.importance * 100).toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

function RegimePie() {
  const { distribution, loading } = useRegimeDistribution();
  const colors: Record<string, string> = {
    trending: "bg-apple-green",
    ranging: "bg-apple-blue",
    volatile: "bg-apple-red",
  };

  if (loading || distribution.length === 0) {
    return <p className="text-sm text-muted-foreground">No regime data</p>;
  }

  const total = distribution.reduce((s, d) => s + d.count, 0);

  return (
    <div className="space-y-2">
      {distribution.map((d) => (
        <div key={d.regime} className="flex items-center gap-2 text-sm">
          <span className={`w-3 h-3 rounded-full ${colors[d.regime] ?? "bg-muted"}`} />
          <span className="capitalize">{d.regime}</span>
          <span className="ml-auto font-number text-muted-foreground">
            {total > 0 ? `${((d.count / total) * 100).toFixed(1)}%` : "0%"}
          </span>
          <span className="font-number w-10 text-right">{d.count}</span>
        </div>
      ))}
    </div>
  );
}

function TrainingHistory() {
  const { runs, loading } = useTrainingHistory();

  if (loading || runs.length === 0) {
    return <p className="text-sm text-muted-foreground">No training history</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            <th className="text-left px-2 py-1.5 font-medium">Time</th>
            <th className="text-right px-2 py-1.5 font-medium">Train AUC</th>
            <th className="text-right px-2 py-1.5 font-medium">Test AUC</th>
            <th className="text-right px-2 py-1.5 font-medium">Samples</th>
            <th className="text-left px-2 py-1.5 font-medium">Trigger</th>
          </tr>
        </thead>
        <tbody>
          {runs.slice(0, 10).map((r) => (
            <tr key={r.id} className="border-b border-border/50 row-hover">
              <td className="px-2 py-1.5 font-number">
                {r.started_at ? new Date(r.started_at).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "—"}
              </td>
              <td className="px-2 py-1.5 text-right font-number">{r.train_auc?.toFixed(3) ?? "—"}</td>
              <td className="px-2 py-1.5 text-right font-number">{r.test_auc?.toFixed(3) ?? "—"}</td>
              <td className="px-2 py-1.5 text-right font-number">{r.sample_count?.toLocaleString() ?? "—"}</td>
              <td className="px-2 py-1.5 text-muted-foreground">{r.trigger_reason ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelDialog({ children }: { children: React.ReactNode }) {
  const { metrics } = useModelMetrics();

  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh]">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 shadow-lg">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <div>
              <DialogTitle className="text-xl">Model Insights</DialogTitle>
              <DialogDescription>
                XGBoost model performance & feature analysis
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6 pt-4 overflow-y-auto">
          {/* Quick stats */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { label: "Train AUC", value: metrics?.trainAuc?.toFixed(3) ?? "—", color: "text-apple-blue" },
              { label: "Test AUC", value: metrics?.testAuc?.toFixed(3) ?? "—", color: "text-apple-green" },
              { label: "Samples", value: metrics?.sampleCount?.toLocaleString() ?? "—", color: "text-apple-purple" },
              { label: "Updated", value: metrics?.updatedAt ? new Date(metrics.updatedAt).toLocaleDateString() : "—", color: "text-muted-foreground" },
            ].map((s) => (
              <div key={s.label} className="p-2.5 rounded-lg bg-surface-light border border-border">
                <p className="text-[10px] text-muted-foreground">{s.label}</p>
                <p className={`text-sm font-bold font-number ${s.color}`}>{s.value}</p>
              </div>
            ))}
          </div>

          {/* Feature importance */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
              <BarChart3 className="h-4 w-4 text-apple-purple" />
              Feature Importance (Top 15)
            </h3>
            <FeatureImportanceBars />
          </div>

          {/* Regime distribution */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
              <Layers className="h-4 w-4 text-apple-blue" />
              Regime Distribution (7d)
            </h3>
            <RegimePie />
          </div>

          {/* Training history */}
          <div>
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
              <Clock className="h-4 w-4 text-apple-cyan" />
              Training History
            </h3>
            <TrainingHistory />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
