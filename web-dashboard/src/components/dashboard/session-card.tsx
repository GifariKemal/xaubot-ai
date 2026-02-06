"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock, Sparkles } from "lucide-react";

interface SessionCardProps {
  session: string;
  isGoldenTime: boolean;
  canTrade: boolean;
}

export function SessionCard({ session, isGoldenTime, canTrade }: SessionCardProps) {
  return (
    <Card className="bg-card/50 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Clock className="h-4 w-4" />
          SESSION
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-center">
          <span className="text-lg font-bold text-amber-500">{session}</span>
        </div>

        <div className={`rounded-md p-2 text-center ${isGoldenTime ? 'bg-green-500/20' : 'bg-muted'}`}>
          <div className="flex items-center justify-center gap-2">
            <Sparkles className={`h-4 w-4 ${isGoldenTime ? 'text-yellow-400' : 'text-muted-foreground'}`} />
            <span className={`text-sm font-semibold ${isGoldenTime ? 'text-green-400' : 'text-muted-foreground'}`}>
              GOLDEN: {isGoldenTime ? 'YES' : 'NO'}
            </span>
          </div>
        </div>

        <div className="flex justify-center">
          <Badge variant={canTrade ? "default" : "destructive"}>
            {canTrade ? 'CAN TRADE' : 'NO TRADE'}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
