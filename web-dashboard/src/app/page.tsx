"use client";

import { useTradingData } from "@/hooks/use-trading-data";
import { useStaggerEntry } from "@/hooks/use-stagger-entry";
import {
  Header,
  PriceCard,
  AccountCard,
  SessionCard,
  RiskCard,
  SignalCard,
  RegimeCard,
  PositionsCard,
  LogCard,
  PriceChart,
  EquityChart,
  BotStatusCard,
  EntryFilterCard,
  PerformanceCard,
  ModelCard,
} from "@/components/dashboard";
import { Skeleton } from "@/components/ui/skeleton";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

function LoadingSkeleton() {
  return (
    <div className="flex-1 min-h-0 flex flex-col gap-1.5 p-1.5">
      <div className="flex-[1] min-h-0 grid grid-cols-4 gap-1.5">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="rounded-lg" />
        ))}
      </div>
      <div className="flex-[1] min-h-0 grid grid-cols-4 gap-1.5">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="rounded-lg" />
        ))}
      </div>
      <div className="flex-[1.6] min-h-0 grid grid-cols-5 gap-1.5">
        <Skeleton className="col-span-3 rounded-lg" />
        <Skeleton className="col-span-2 rounded-lg" />
      </div>
      <div className="flex-[1.2] min-h-0 grid grid-cols-4 gap-1.5">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="rounded-lg" />
        ))}
      </div>
    </div>
  );
}

function ErrorDisplay({ message }: { message: string }) {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="w-14 h-14 rounded-full bg-danger-bg mx-auto flex items-center justify-center">
          <span className="text-danger text-2xl font-bold">!</span>
        </div>
        <p className="text-danger text-lg font-semibold">Connection Error</p>
        <p className="text-muted-foreground">{message}</p>
        <p className="text-muted-foreground/60 text-sm">
          Make sure the API server is running on port 8000
        </p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data, loading, error, dataAge } = useTradingData();
  const visible = useStaggerEntry(17, 40);

  const now = new Date();
  const wibTime = now.toLocaleTimeString("en-US", {
    timeZone: "Asia/Jakarta",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  if (loading && !data) {
    return (
      <div className="h-screen flex flex-col overflow-hidden bg-background">
        <Header connected={false} lastUpdate={wibTime} dataAge={999} />
        <LoadingSkeleton />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="h-screen flex flex-col overflow-hidden bg-background">
        <Header connected={false} lastUpdate={wibTime} dataAge={999} />
        <ErrorDisplay message={error} />
      </div>
    );
  }

  if (!data) return null;

  return (
    <TooltipProvider delayDuration={200}>
      <div className="h-screen flex flex-col overflow-hidden bg-background">
        <Header
          connected={data.connected}
          lastUpdate={wibTime}
          dataAge={dataAge}
        />

        {/* Main grid — fills remaining height, NO scroll */}
        <main className="flex-1 min-h-0 flex flex-col gap-1.5 p-1.5">

          {/* Row 1: Market Overview */}
          <div className="flex-[1] min-h-0 grid grid-cols-4 gap-1.5">
            <div className={cn("stagger-enter h-full", visible[0] && "visible")}>
              <PriceCard
                price={data.price}
                spread={data.spread}
                priceChange={data.priceChange}
                priceHistory={data.priceHistory}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[1] && "visible")}>
              <AccountCard
                balance={data.balance}
                equity={data.equity}
                profit={data.profit}
                equityHistory={data.equityHistory}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[2] && "visible")}>
              <SessionCard
                session={data.session}
                isGoldenTime={data.isGoldenTime}
                canTrade={data.canTrade}
                sessionMultiplier={data.sessionMultiplier}
                timeFilter={data.timeFilter}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[3] && "visible")}>
              <RiskCard
                dailyLoss={data.dailyLoss}
                dailyProfit={data.dailyProfit}
                consecutiveLosses={data.consecutiveLosses}
                riskPercent={data.riskPercent}
                riskMode={data.riskMode}
              />
            </div>
          </div>

          {/* Row 2: AI Signals */}
          <div className="flex-[1] min-h-0 grid grid-cols-5 gap-1.5">
            <div className={cn("stagger-enter h-full", visible[4] && "visible")}>
              <SignalCard
                title="SMC Signal"
                icon="smc"
                signal={data.smc.signal}
                confidence={data.smc.confidence}
                detail={`${data.smc.reason || ""}${data.h1Bias ? ` | H1: ${data.h1Bias}` : ""}`}
                updatedAt={data.smc.updatedAt}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[5] && "visible")}>
              <SignalCard
                title="ML Prediction"
                icon="ml"
                signal={data.ml.signal}
                confidence={data.ml.confidence}
                buyProb={data.ml.buyProb}
                sellProb={data.ml.sellProb}
                updatedAt={data.ml.updatedAt}
                threshold={data.dynamicThreshold}
                marketQuality={data.marketQuality}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[6] && "visible")}>
              <RegimeCard
                name={data.regime.name}
                volatility={data.regime.volatility}
                confidence={data.regime.confidence}
                updatedAt={data.regime.updatedAt}
                h1Bias={data.h1Bias}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[7] && "visible")}>
              <PerformanceCard
                marketScore={data.marketScore}
                marketQuality={data.marketQuality}
                dynamicThreshold={data.dynamicThreshold}
                performance={data.performance}
                riskMode={data.riskMode}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[8] && "visible")}>
              <ModelCard />
            </div>
          </div>

          {/* Row 3: Charts */}
          <div className="flex-[1.6] min-h-0 grid grid-cols-5 gap-1.5">
            <div className={cn("stagger-enter col-span-3 min-h-0", visible[9] && "visible")}>
              <PriceChart data={data.priceHistory} />
            </div>
            <div className={cn("stagger-enter col-span-2 min-h-0", visible[10] && "visible")}>
              <EquityChart
                equityData={data.equityHistory}
                balanceData={data.balanceHistory}
              />
            </div>
          </div>

          {/* Row 4: Trading + Log */}
          <div className="flex-[1.2] min-h-0 grid grid-cols-4 gap-1.5">
            <div className={cn("stagger-enter h-full", visible[11] && "visible")}>
              <EntryFilterCard filters={data.entryFilters || []} />
            </div>
            <div className={cn("stagger-enter h-full", visible[12] && "visible")}>
              <PositionsCard
                positions={data.positions}
                positionDetails={data.positionDetails}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[13] && "visible")}>
              <BotStatusCard
                riskMode={data.riskMode}
                cooldown={data.cooldown}
                autoTrainer={data.autoTrainer}
                performance={data.performance}
                marketClose={data.marketClose}
                settings={data.settings}
              />
            </div>
            <div className={cn("stagger-enter h-full", visible[14] && "visible")}>
              <LogCard logs={data.logs} />
            </div>
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}
