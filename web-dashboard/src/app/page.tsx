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
  EquityChart,
} from "@/components/dashboard";
import { Skeleton } from "@/components/ui/skeleton";

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      {[...Array(8)].map((_, i) => (
        <Skeleton key={i} className="h-[150px] rounded-xl" />
      ))}
    </div>
  );
}

function ErrorDisplay({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="text-center">
        <p className="text-destructive text-lg font-semibold">Connection Error</p>
        <p className="text-muted-foreground">{message}</p>
        <p className="text-sm text-muted-foreground mt-2">
          Make sure the API server is running on port 8000
        </p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data, loading, error, dataAge } = useTradingData();

  // Format current time for header
  const now = new Date();
  const wibTime = now.toLocaleTimeString('en-US', {
    timeZone: 'Asia/Jakarta',
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  if (loading && !data) {
    return (
      <div className="min-h-screen bg-background">
        <Header connected={false} lastUpdate={wibTime} dataAge={999} />
        <LoadingSkeleton />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="min-h-screen bg-background">
        <Header connected={false} lastUpdate={wibTime} dataAge={999} />
        <ErrorDisplay message={error} />
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="min-h-screen bg-background">
      <Header
        connected={data.connected}
        lastUpdate={wibTime}
        dataAge={dataAge}
      />

      <main className="container py-4">
        <div className="grid grid-cols-2 gap-4">
          {/* Row 1: Price Chart (full width) */}
          <PriceChart data={data.priceHistory} />

          {/* Row 2: Price & Account */}
          <PriceCard
            price={data.price}
            spread={data.spread}
            priceChange={data.priceChange}
          />
          <AccountCard
            balance={data.balance}
            equity={data.equity}
            profit={data.profit}
          />

          {/* Row 3: Session & Risk */}
          <SessionCard
            session={data.session}
            isGoldenTime={data.isGoldenTime}
            canTrade={data.canTrade}
          />
          <RiskCard
            dailyLoss={data.dailyLoss}
            dailyProfit={data.dailyProfit}
            consecutiveLosses={data.consecutiveLosses}
            riskPercent={data.riskPercent}
          />

          {/* Row 4: SMC & ML */}
          <SignalCard
            title="SMC SIGNAL"
            icon="smc"
            signal={data.smc.signal}
            confidence={data.smc.confidence}
            detail={data.smc.reason}
          />
          <SignalCard
            title="ML PREDICTION"
            icon="ml"
            signal={data.ml.signal}
            confidence={data.ml.confidence}
            buyProb={data.ml.buyProb}
            sellProb={data.ml.sellProb}
          />

          {/* Row 5: Regime & Positions */}
          <RegimeCard
            name={data.regime.name}
            volatility={data.regime.volatility}
            confidence={data.regime.confidence}
          />
          <PositionsCard positions={data.positions} />

          {/* Row 6: Equity Chart (full width) */}
          <EquityChart
            equityData={data.equityHistory}
            balanceData={data.balanceHistory}
          />

          {/* Row 7: Log (full width) */}
          <LogCard logs={data.logs} />
        </div>
      </main>

      {/* Footer Status */}
      <footer className="fixed bottom-0 w-full border-t bg-background/95 backdrop-blur py-2">
        <div className="container flex justify-between text-xs text-muted-foreground">
          <span>Last update: {data.timestamp}</span>
          <span>AI Trading Bot Monitor v1.0</span>
        </div>
      </footer>
    </div>
  );
}
