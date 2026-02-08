"use client";

import { useTradingData } from "@/hooks/use-trading-data";
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
  BotStatusCard,
  EntryFilterCard,
} from "@/components/dashboard";
import { Skeleton } from "@/components/ui/skeleton";

function LoadingSkeleton() {
  return (
    <div className="flex-1 min-h-0 flex flex-col gap-1.5 p-1.5">
      <div className="flex gap-1.5">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={`r1-${i}`} className="flex-1 h-[80px] rounded-lg" />
        ))}
      </div>
      <div className="flex gap-1.5">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={`r2-${i}`} className="flex-1 h-[90px] rounded-lg" />
        ))}
      </div>
      <div className="flex-1 min-h-0 flex gap-1.5">
        <Skeleton className="flex-[3] rounded-lg" />
        <Skeleton className="flex-1 rounded-lg" />
      </div>
    </div>
  );
}

function ErrorDisplay({ message }: { message: string }) {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center space-y-3">
        <div className="w-12 h-12 rounded-full bg-danger-bg mx-auto flex items-center justify-center">
          <span className="text-danger text-xl">!</span>
        </div>
        <p className="text-danger text-base font-semibold">Connection Error</p>
        <p className="text-muted-foreground text-sm">{message}</p>
        <p className="text-muted-foreground/60 text-xs">
          Make sure the API server is running on port 8000
        </p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data, loading, error, dataAge } = useTradingData();

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
      <div className="fixed inset-0 overflow-hidden flex flex-col bg-background">
        <Header connected={false} lastUpdate={wibTime} dataAge={999} />
        <LoadingSkeleton />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="fixed inset-0 overflow-hidden flex flex-col bg-background">
        <Header connected={false} lastUpdate={wibTime} dataAge={999} />
        <ErrorDisplay message={error} />
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="fixed inset-0 overflow-hidden flex flex-col bg-background max-w-full">
      <Header
        connected={data.connected}
        lastUpdate={wibTime}
        dataAge={dataAge}
      />

      <main className="flex-1 min-h-0 flex flex-col gap-1.5 p-1.5 overflow-hidden">
        {/* ── Row 1: Status ── */}
        <div
          className="grid gap-1.5 overflow-hidden"
          style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}
        >
          <div className="min-w-0 overflow-hidden">
            <PriceCard
              price={data.price}
              spread={data.spread}
              priceChange={data.priceChange}
              priceHistory={data.priceHistory}
            />
          </div>
          <div className="min-w-0 overflow-hidden">
            <AccountCard
              balance={data.balance}
              equity={data.equity}
              profit={data.profit}
              equityHistory={data.equityHistory}
            />
          </div>
          <div className="min-w-0 overflow-hidden">
            <SessionCard
              session={data.session}
              isGoldenTime={data.isGoldenTime}
              canTrade={data.canTrade}
              sessionMultiplier={data.sessionMultiplier}
              timeFilter={data.timeFilter}
            />
          </div>
          <div className="min-w-0 overflow-hidden">
            <RiskCard
              dailyLoss={data.dailyLoss}
              dailyProfit={data.dailyProfit}
              consecutiveLosses={data.consecutiveLosses}
              riskPercent={data.riskPercent}
              riskMode={data.riskMode}
            />
          </div>
        </div>

        {/* ── Row 2: Signals + Bot Status ── */}
        <div
          className="grid gap-1.5 overflow-hidden"
          style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}
        >
          <div className="min-w-0 overflow-hidden">
            <SignalCard
              title="SMC Signal"
              icon="smc"
              signal={data.smc.signal}
              confidence={data.smc.confidence}
              detail={`${data.smc.reason || ""}${data.h1Bias ? ` | H1: ${data.h1Bias}` : ""}`}
              updatedAt={data.smc.updatedAt}
            />
          </div>
          <div className="min-w-0 overflow-hidden">
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
          <div className="min-w-0 overflow-hidden">
            <RegimeCard
              name={data.regime.name}
              volatility={data.regime.volatility}
              confidence={data.regime.confidence}
              updatedAt={data.regime.updatedAt}
              h1Bias={data.h1Bias}
            />
          </div>
          <div className="min-w-0 overflow-hidden">
            <BotStatusCard
              riskMode={data.riskMode}
              cooldown={data.cooldown}
              autoTrainer={data.autoTrainer}
              performance={data.performance}
              marketClose={data.marketClose}
            />
          </div>
        </div>

        {/* ── Row 3: Chart + Sidebar (fills remaining) ── */}
        <div
          className="flex-1 min-h-0 grid gap-1.5 overflow-hidden"
          style={{ gridTemplateColumns: '3fr 1fr' }}
        >
          <div className="min-w-0 min-h-0 overflow-hidden">
            <PriceChart data={data.priceHistory} />
          </div>
          <div className="min-w-0 min-h-0 overflow-hidden flex flex-col gap-1.5">
            <div className="min-h-0" style={{ flex: '0 0 auto', maxHeight: '40%' }}>
              <EntryFilterCard filters={data.entryFilters || []} />
            </div>
            <div className="flex-1 min-h-0">
              <PositionsCard
                positions={data.positions}
                positionDetails={data.positionDetails}
              />
            </div>
            <div className="flex-1 min-h-0">
              <LogCard logs={data.logs} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
