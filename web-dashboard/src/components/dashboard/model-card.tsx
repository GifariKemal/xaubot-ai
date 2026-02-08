"use client";

import { Brain } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useModelMetrics } from "@/hooks/use-model-insights";
import { ModelDialog } from "./model-dialog";

export function ModelCard() {
  const { metrics, loading } = useModelMetrics();

  const auc = metrics?.testAuc ?? 0;
  const topFeature = metrics?.featureImportance?.[0]?.name ?? "—";
  const aucColor = auc >= 0.8 ? "text-success" : auc >= 0.7 ? "text-apple-blue" : auc >= 0.6 ? "text-warning" : "text-danger";
  const aucBadge = auc >= 0.8 ? "success" : auc >= 0.7 ? "info" : auc >= 0.6 ? "warning" : "danger";

  return (
    <ModelDialog>
      <Card className="glass glass-purple h-full cursor-pointer">
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="flex items-center gap-1.5 text-sm">
            <Brain className="h-3.5 w-3.5 text-apple-purple" />
            Model
          </CardTitle>
          <Badge variant={aucBadge as "success" | "info" | "warning" | "danger"} className="text-xs">
            AUC {loading ? "..." : auc.toFixed(2)}
          </Badge>
        </CardHeader>
        <CardContent>
          <div className="space-y-1">
            <p className={`text-xl font-bold font-number ${aucColor}`}>
              {loading ? "—" : `${(auc * 100).toFixed(1)}%`}
            </p>
            <p className="text-xs text-muted-foreground truncate">
              Top: {topFeature}
            </p>
            <p className="text-[10px] text-muted-foreground">
              {metrics?.sampleCount ? `${metrics.sampleCount.toLocaleString()} samples` : "Click for details"}
            </p>
          </div>
        </CardContent>
      </Card>
    </ModelDialog>
  );
}
