"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Bot, TrendingUp, TrendingDown, ShieldAlert, Clock, Zap,
  AlertTriangle, Coffee, CheckCircle2, XCircle, Minus,
  Activity, Target, BarChart3, Eye,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { TradingStatus } from "@/types/trading";

type InsightType = "info" | "success" | "warning" | "danger";

interface Insight {
  icon: React.ReactNode;
  text: string;
  type: InsightType;
}

const typeStyles: Record<InsightType, string> = {
  info: "text-blue-400/90",
  success: "text-apple-green",
  warning: "text-orange-400",
  danger: "text-apple-red",
};

const sessionNames: Record<string, string> = {
  sydney: "Sydney", tokyo: "Tokyo", london: "London",
  new_york: "New York", off_hours: "Off Hours",
};

function generateInsights(data: TradingStatus): Insight[] {
  const insights: Insight[] = [];
  const hasPositions = data.positions && data.positions.length > 0;
  const sessionName = sessionNames[data.session?.toLowerCase()] || data.session;
  const regimeName = data.regime?.name || "unknown";
  const regimeConf = data.regime?.confidence ? (data.regime.confidence * 100).toFixed(0) : "?";
  const smcSignal = data.smc?.signal || "NONE";
  const smcConf = data.smc?.confidence ? (data.smc.confidence * 100).toFixed(0) : "0";
  const mlSignal = data.ml?.signal || "HOLD";
  const mlConf = data.ml?.confidence ? (data.ml.confidence * 100).toFixed(0) : "0";
  const h1Bias = data.h1Bias || "N/A";
  const dynThreshold = data.dynamicThreshold ? (data.dynamicThreshold * 100).toFixed(0) : "60";
  const spread = data.spread?.toFixed(1) || "?";

  // ═══════════════════════════════════════════
  // 1. STATUS MARKET
  // ═══════════════════════════════════════════
  if (data.marketClose && !data.marketClose.marketOpen) {
    insights.push({
      icon: <Coffee className="h-3.5 w-3.5" />,
      text: "Market sedang tutup. Bot dalam mode standby — tidak ada analisis atau eksekusi. Menunggu market buka kembali.",
      type: "info",
    });
    return insights;
  }

  if (data.marketClose?.nearWeekend) {
    const hrs = data.marketClose.hoursToWeekendClose;
    insights.push({
      icon: <AlertTriangle className="h-3.5 w-3.5" />,
      text: `Peringatan: Weekend close dalam ${hrs.toFixed(1)} jam (Sabtu 05:00 WIB). Bot akan menolak entry baru dan memastikan semua posisi ditutup sebelum market close.`,
      type: "warning",
    });
  }

  // ═══════════════════════════════════════════
  // 2. RANGKUMAN SITUASI (top-level summary)
  // ═══════════════════════════════════════════
  if (hasPositions) {
    const totalProfit = data.positions.reduce((sum, p) => sum + p.profit, 0);
    const profitStr = totalProfit >= 0 ? `+$${totalProfit.toFixed(2)}` : `-$${Math.abs(totalProfit).toFixed(2)}`;
    insights.push({
      icon: <Activity className="h-3.5 w-3.5" />,
      text: `Sedang dalam posisi (${data.positions.length} trade aktif, total P/L: ${profitStr}). Bot memantau exit conditions setiap 5 detik.`,
      type: totalProfit >= 0 ? "success" : "warning",
    });
  } else {
    insights.push({
      icon: <Eye className="h-3.5 w-3.5" />,
      text: `Tidak ada posisi terbuka. Bot menganalisis market setiap candle M15 untuk mencari peluang entry.`,
      type: "info",
    });
  }

  // ═══════════════════════════════════════════
  // 3. SESI & WAKTU
  // ═══════════════════════════════════════════
  if (data.isGoldenTime) {
    insights.push({
      icon: <Zap className="h-3.5 w-3.5" />,
      text: `Sesi ${sessionName} — GOLDEN TIME! Overlap London-NY menghasilkan volatilitas tertinggi. Lot size dinaikkan ${data.sessionMultiplier || 1}x. Peluang terbaik untuk entry.`,
      type: "success",
    });
  } else if (!data.canTrade) {
    const nextSession = data.session?.toLowerCase() === "off_hours" ? "Sydney (04:00 WIB)"
      : data.session?.toLowerCase() === "sydney" ? "London (14:00 WIB)"
      : "sesi berikutnya";
    insights.push({
      icon: <Clock className="h-3.5 w-3.5" />,
      text: `Sesi ${sessionName} — di luar jam trading aktif. Volume rendah, spread bisa melebar. Menunggu ${nextSession}.`,
      type: "info",
    });
  } else {
    const multiplierText = data.sessionMultiplier && data.sessionMultiplier > 1
      ? ` Lot multiplier: ${data.sessionMultiplier}x.`
      : data.sessionMultiplier && data.sessionMultiplier < 1
      ? ` SAFE MODE: lot dikurangi ${data.sessionMultiplier}x.`
      : "";
    insights.push({
      icon: <Clock className="h-3.5 w-3.5" />,
      text: `Sesi ${sessionName} aktif — market terbuka untuk trading.${multiplierText}`,
      type: "info",
    });
  }

  if (data.timeFilter?.isBlocked) {
    insights.push({
      icon: <AlertTriangle className="h-3.5 w-3.5" />,
      text: `Jam ${data.timeFilter.wibHour}:00 WIB diblokir oleh time filter (jam-jam dengan win rate rendah secara historis). Entry ditunda.`,
      type: "warning",
    });
  }

  // ═══════════════════════════════════════════
  // 4. KONDISI MARKET (regime + spread + H1)
  // ═══════════════════════════════════════════
  const regimeDetail = regimeName.includes("trending")
    ? `Market trending (HMM confidence ${regimeConf}%) — kondisi ideal. Harga bergerak terarah, sinyal lebih reliable.`
    : regimeName.includes("high_volatility") || regimeName.includes("volatile")
    ? `Volatilitas tinggi (HMM ${regimeConf}%) — HATI-HATI! Pergerakan harga liar, SL bisa cepat tersentuh. Bot menaikkan spread tolerance.`
    : regimeName.includes("low_volatility") || regimeName.includes("ranging")
    ? `Volatilitas rendah / ranging (HMM ${regimeConf}%) — market diam, peluang kecil. Bot menunggu breakout atau perubahan regime.`
    : `Regime: ${regimeName} (HMM ${regimeConf}%).`;

  insights.push({
    icon: regimeName.includes("trending") ? <TrendingUp className="h-3.5 w-3.5" />
      : regimeName.includes("volatile") || regimeName.includes("high") ? <AlertTriangle className="h-3.5 w-3.5" />
      : <BarChart3 className="h-3.5 w-3.5" />,
    text: regimeDetail,
    type: regimeName.includes("trending") ? "success"
      : regimeName.includes("volatile") || regimeName.includes("high") ? "warning"
      : "info",
  });

  // H1 Bias detail
  if (h1Bias && h1Bias !== "N/A") {
    const h1Text = h1Bias === "BULLISH"
      ? `H1 Bias: BULLISH — harga di atas EMA20 H1, tren naik jangka menengah. Entry BUY diizinkan.`
      : h1Bias === "BEARISH"
      ? `H1 Bias: BEARISH — harga di bawah EMA20 H1, tren turun jangka menengah. Entry SELL diizinkan.`
      : `H1 Bias: NEUTRAL — harga dekat EMA20 H1, tidak ada tren jelas. Entry di-hold sampai arah terbentuk.`;
    insights.push({
      icon: h1Bias === "BULLISH" ? <TrendingUp className="h-3.5 w-3.5" />
        : h1Bias === "BEARISH" ? <TrendingDown className="h-3.5 w-3.5" />
        : <Minus className="h-3.5 w-3.5" />,
      text: h1Text,
      type: h1Bias === "NEUTRAL" ? "warning" : "info",
    });
  }

  // Spread
  insights.push({
    icon: <Activity className="h-3.5 w-3.5" />,
    text: `Spread saat ini: ${spread} pips. ${parseFloat(spread) > 40 ? "Cukup lebar — entry mungkin ditunda." : parseFloat(spread) > 25 ? "Normal." : "Ketat — kondisi bagus."}`,
    type: parseFloat(spread) > 40 ? "warning" : "info",
  });

  // ═══════════════════════════════════════════
  // 5. ANALISIS SINYAL
  // ═══════════════════════════════════════════
  if (smcSignal !== "NONE" && smcSignal !== "HOLD") {
    const smcReason = data.smc?.reason || "";
    const mlAgrees = mlSignal === smcSignal;
    const mlAboveThreshold = data.ml?.confidence && data.ml.confidence * 100 >= parseFloat(dynThreshold);

    if (mlAgrees && mlAboveThreshold) {
      insights.push({
        icon: <Zap className="h-3.5 w-3.5" />,
        text: `SINYAL KUAT: SMC ${smcSignal} (${smcConf}%) + ML ${mlSignal} (${mlConf}%) — keduanya sepakat dan di atas threshold ${dynThreshold}%.${smcReason ? ` SMC: ${smcReason}.` : ""} Tinggal filter lain terpenuhi untuk entry.`,
        type: "success",
      });
    } else if (mlAgrees && !mlAboveThreshold) {
      insights.push({
        icon: <Target className="h-3.5 w-3.5" />,
        text: `SMC ${smcSignal} (${smcConf}%) & ML setuju ${mlSignal}, tapi confidence ML (${mlConf}%) masih di bawah threshold (${dynThreshold}%). Perlu lebih yakin.`,
        type: "warning",
      });
    } else {
      insights.push({
        icon: <Minus className="h-3.5 w-3.5" />,
        text: `Sinyal konflik — SMC: ${smcSignal} (${smcConf}%) vs ML: ${mlSignal} (${mlConf}%). Bot menunggu kedua model sinkron sebelum entry.${smcReason ? ` SMC: ${smcReason}.` : ""}`,
        type: "info",
      });
    }
  } else {
    insights.push({
      icon: <Minus className="h-3.5 w-3.5" />,
      text: `Belum ada sinyal — SMC: ${smcSignal}, ML: ${mlSignal} (${mlConf}%). Menunggu setup terbentuk di candle M15 berikutnya.`,
      type: "info",
    });
  }

  // ═══════════════════════════════════════════
  // 6. POSISI TERBUKA (detail)
  // ═══════════════════════════════════════════
  if (hasPositions) {
    for (const pos of data.positions) {
      const detail = data.positionDetails?.find((d) => d.ticket === pos.ticket);
      const profitStr = pos.profit >= 0 ? `+$${pos.profit.toFixed(2)}` : `-$${Math.abs(pos.profit).toFixed(2)}`;
      const dir = pos.type;

      const ageMinutes = detail?.tradeHours ? detail.tradeHours * 60 : 0;
      const ageText = ageMinutes > 0
        ? ageMinutes < 60 ? `${ageMinutes.toFixed(0)}m` : `${(ageMinutes / 60).toFixed(1)}j`
        : "";
      const momentumVal = detail?.momentum ?? 0;
      const tpProb = detail?.tpProbability ?? 0;
      const peakProfit = detail?.peakProfit ?? 0;
      const drawdown = detail?.drawdownFromPeak ?? 0;
      const reversalWarns = detail?.reversalWarnings ?? 0;

      let analysis = "";
      if (pos.profit >= 20) {
        analysis = `Profit sangat baik! Trailing SL aktif mengunci keuntungan. Peak: $${peakProfit.toFixed(2)}, drawdown dari peak: $${drawdown.toFixed(2)}. TP probability: ${tpProb.toFixed(0)}%.`;
      } else if (pos.profit >= 10) {
        analysis = `Profit bagus — momentum ${momentumVal > 0 ? "positif" : "melemah"} (${momentumVal.toFixed(0)}). Peak profit: $${peakProfit.toFixed(2)}. ${tpProb > 50 ? "Peluang capai TP masih tinggi." : "Mulai pantau untuk ambil profit."}`;
      } else if (pos.profit >= 0) {
        analysis = `Masih floating ${profitStr} (${ageText}). Momentum: ${momentumVal.toFixed(0)}. ${ageMinutes < 15 ? "Masih dalam grace period 15 menit — biarkan trade berkembang." : "Memantau arah selanjutnya."}`;
      } else if (ageMinutes < 15) {
        analysis = `Loss ${profitStr} tapi masih GRACE PERIOD (${ageText}/15m). Early cut TIDAK aktif — memberi waktu 1 candle M15 untuk develop. Hard SL tetap jadi safety net.`;
      } else {
        analysis = `Loss ${profitStr} (${ageText}), momentum: ${momentumVal.toFixed(0)}. ${momentumVal < -50 ? "Momentum lemah — early cut bisa trigger kapan saja!" : "Momentum belum terlalu buruk, masih ada harapan recovery."}${reversalWarns > 0 ? ` Reversal warning: ${reversalWarns}x.` : ""}`;
      }

      insights.push({
        icon: pos.profit >= 5 ? <TrendingUp className="h-3.5 w-3.5" />
          : pos.profit >= 0 ? <Clock className="h-3.5 w-3.5" />
          : pos.profit > -15 ? <AlertTriangle className="h-3.5 w-3.5" />
          : <ShieldAlert className="h-3.5 w-3.5" />,
        text: `📊 #${pos.ticket} ${dir} @ ${pos.priceOpen.toFixed(2)} — ${analysis}`,
        type: pos.profit >= 5 ? "success" : pos.profit >= 0 ? "info" : pos.profit > -15 ? "warning" : "danger",
      });
    }
  }

  // ═══════════════════════════════════════════
  // 7. KENAPA TIDAK ENTRY (detail per filter)
  // ═══════════════════════════════════════════
  if (!hasPositions) {
    const filters = data.entryFilters || [];
    const blockers = filters.filter((f) => !f.passed && !f.detail?.includes("[DISABLED]"));
    const disabledFilters = filters.filter((f) => f.detail?.includes("[DISABLED]"));
    const passedCount = filters.filter((f) => f.passed).length;

    if (blockers.length > 0) {
      insights.push({
        icon: <XCircle className="h-3.5 w-3.5" />,
        text: `Entry diblokir oleh ${blockers.length} filter (${passedCount}/${filters.length} lolos). Filter pertama yang gagal: "${blockers[0].name}" — ${blockers[0].detail || "tidak memenuhi syarat"}. Bot tidak akan entry sampai SEMUA filter hijau.`,
        type: "warning",
      });
    } else if (filters.length > 0 && blockers.length === 0) {
      insights.push({
        icon: <CheckCircle2 className="h-3.5 w-3.5" />,
        text: `Semua ${filters.length} filter terpenuhi! Bot siap entry begitu ada sinyal valid dari SMC + ML.`,
        type: "success",
      });
    }

    if (disabledFilters.length > 0) {
      const names = disabledFilters.map((f) => f.name).join(", ");
      insights.push({
        icon: <AlertTriangle className="h-3.5 w-3.5" />,
        text: `${disabledFilters.length} filter dinonaktifkan manual: ${names}. Filter ini di-bypass (auto-pass). Aktifkan kembali di panel Filters jika diperlukan.`,
        type: "warning",
      });
    }
  }

  // ═══════════════════════════════════════════
  // 8. RISK & MODAL
  // ═══════════════════════════════════════════
  const netPnl = (data.dailyProfit || 0) - (data.dailyLoss || 0);
  if (data.dailyLoss > 0 || data.dailyProfit > 0) {
    const pnlStr = netPnl >= 0 ? `+$${netPnl.toFixed(2)}` : `-$${Math.abs(netPnl).toFixed(2)}`;
    const remaining = data.riskMode?.remainingDailyRisk;
    const riskDetail = remaining !== undefined ? ` Sisa risk harian: $${remaining.toFixed(0)}.` : "";

    if (netPnl < 0 && remaining !== undefined && remaining < 50) {
      insights.push({
        icon: <ShieldAlert className="h-3.5 w-3.5" />,
        text: `⚠️ P/L hari ini: ${pnlStr} (loss $${data.dailyLoss.toFixed(2)}, profit $${data.dailyProfit.toFixed(2)}).${riskDetail} Mendekati batas — bot sangat konservatif.`,
        type: "danger",
      });
    } else {
      insights.push({
        icon: netPnl >= 0 ? <TrendingUp className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />,
        text: `P/L hari ini: ${pnlStr} (loss $${data.dailyLoss.toFixed(2)}, profit $${data.dailyProfit.toFixed(2)}).${riskDetail}`,
        type: netPnl >= 0 ? "success" : "warning",
      });
    }
  }

  if (data.riskMode) {
    const mode = data.riskMode.mode?.toUpperCase() || "NORMAL";
    if (mode !== "NORMAL") {
      insights.push({
        icon: <ShieldAlert className="h-3.5 w-3.5" />,
        text: `Risk mode: ${mode} — ${data.riskMode.reason}. Lot: ${data.riskMode.recommendedLot} (max ${data.riskMode.maxAllowedLot}).`,
        type: mode === "RECOVERY" || mode === "PROTECTIVE" ? "danger" : "warning",
      });
    }
  }

  // ═══════════════════════════════════════════
  // 9. PERFORMA BOT
  // ═══════════════════════════════════════════
  if (data.performance) {
    const p = data.performance;
    const uptimeText = p.uptimeHours < 1
      ? `${(p.uptimeHours * 60).toFixed(0)} menit`
      : `${p.uptimeHours.toFixed(1)} jam`;
    insights.push({
      icon: <Bot className="h-3.5 w-3.5" />,
      text: `Bot aktif ${uptimeText}, loop ke-${p.loopCount}. Avg execution: ${p.avgExecutionMs.toFixed(0)}ms. Session trades: ${p.totalSessionTrades} (P/L: ${p.totalSessionProfit >= 0 ? "+" : ""}$${p.totalSessionProfit.toFixed(2)}).`,
      type: "info",
    });
  }

  return insights;
}

interface AssistantCardProps {
  data: TradingStatus;
}

export function AssistantCard({ data }: AssistantCardProps) {
  const insights = generateInsights(data);
  const [wibTime, setWibTime] = useState("");

  useEffect(() => {
    const update = () => {
      setWibTime(
        new Date().toLocaleString("id-ID", {
          timeZone: "Asia/Jakarta",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="glass h-full overflow-hidden flex flex-col accent-top-blue">
      <CardHeader>
        <CardTitle className="text-sm font-medium flex items-center gap-1.5 uppercase tracking-wider text-blue-400">
          <Bot className="h-4 w-4" />
          Asisten Bot
          <span className="ml-auto text-[10px] font-normal normal-case tracking-normal text-muted-foreground/60 font-mono">
            {wibTime} WIB
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-auto pr-2">
        <div className="space-y-2">
          {insights.map((insight, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2 text-[11px] leading-[1.5]"
            >
              <span className={cn("mt-0.5 shrink-0", typeStyles[insight.type])}>
                {insight.icon}
              </span>
              <span className="text-foreground/80">{insight.text}</span>
            </div>
          ))}
        </div>

        {/* Last analysis timestamp */}
        <div className="mt-3 pt-2 border-t border-white/5 text-[10px] text-muted-foreground/40 flex items-center gap-1">
          <Clock className="h-2.5 w-2.5" />
          Analisis terakhir: {data.timestamp
            ? new Date(data.timestamp).toLocaleString("id-ID", {
                timeZone: "Asia/Jakarta",
                day: "2-digit",
                month: "short",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false,
              })
            : wibTime
          } WIB
        </div>
      </CardContent>
    </Card>
  );
}
