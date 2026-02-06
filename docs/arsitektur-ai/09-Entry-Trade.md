# Entry Trade — Proses Masuk Posisi

> **File utama:** `main_live.py`
> **File pendukung:** `src/smc_polars.py`, `src/ml_model.py`, `src/smart_risk_manager.py`, `src/session_filter.py`

---

## Apa Itu Entry Trade?

Entry Trade adalah keseluruhan proses dari **mendeteksi peluang** hingga **mengirim order ke broker**. Bot menggunakan **10+ filter** yang harus SEMUA lolos sebelum satu trade dieksekusi.

**Analogi:** Entry Trade seperti **proses boarding pesawat** — harus punya tiket (signal), passport valid (confirmation), lulus security check (risk), tepat waktu (session), dan gate terbuka (position limit).

---

## Checklist Entry (Semua Harus PASS)

```
 1. [SESSION]     Session filter izinkan trading?
 2. [RISK MODE]   Trading mode bukan STOPPED?
 3. [SMC SIGNAL]  Ada signal dari SMC Analyzer?
 4. [ML CONFIRM]  XGBoost confidence >= 50%?
 5. [ML AGREE]    ML tidak strongly disagree (>65% berlawanan)?
 6. [QUALITY]     Market quality bukan AVOID/CRISIS?
 7. [CONFIRM]     Signal konsisten 2 bar berturut?
 8. [PULLBACK]    Bukan sedang pullback/retrace?
 9. [COOLDOWN]    Sudah 5 menit sejak trade terakhir?
10. [POS LIMIT]   Posisi terbuka < 2?
11. [LOT SIZE]    Lot > 0 setelah semua adjustment?

SEMUA PASS -> Execute Trade
SATU GAGAL -> Skip, tunggu loop berikutnya
```

---

## Step-by-Step Flow

### Step 1: Session Filter

```python
# main_live.py Lines 472-483
session_ok, session_reason, session_multiplier = self.session_filter.can_trade()

if not session_ok:
    return  # Skip — bukan waktu trading

# Simpan multiplier untuk lot sizing nanti
self._current_session_multiplier = session_multiplier
```

**Bisa block:** Weekend, Friday >23:00, danger zone (00:00-06:00), low volatility session.

---

### Step 2: Risk Mode Check

```python
# main_live.py Lines 537-542
risk_rec = self.smart_risk.get_trading_recommendation()

if not risk_rec["can_trade"]:
    return  # STOPPED mode — daily/total limit tercapai
```

**Bisa block:** Mode STOPPED (daily loss >= $250, total loss >= $500).

---

### Step 3: SMC Signal Generation

```python
# main_live.py Lines 498-499
smc_signal = self.smc.generate_signal(df)

if smc_signal is None:
    return  # Tidak ada setup SMC yang valid
```

**SMC membutuhkan:**
- Market structure (bullish/bearish) ATAU BOS/CHoCH
- DAN (FVG ATAU Order Block)
- Minimum 2:1 risk/reward

**Output:** Entry price, SL, TP, confidence (55-85%), reason.

---

### Step 4: ML Confidence Check

```python
# main_live.py Lines 419-425
ml_prediction = self.ml_model.predict(df, feature_cols)

# Lines 664-669
if ml_prediction.confidence < 0.50:
    return  # ML terlalu tidak yakin
```

---

### Step 5: ML Agreement Check

```python
# main_live.py Lines 676-684
# Jika SMC bilang BUY tapi ML bilang SELL dengan confidence > 65%:
if smc_signal.signal_type == "BUY":
    if ml_prediction.signal == "SELL" and ml_prediction.confidence > 0.65:
        return  # ML strongly disagrees — VETO

if smc_signal.signal_type == "SELL":
    if ml_prediction.signal == "BUY" and ml_prediction.confidence > 0.65:
        return  # ML strongly disagrees — VETO
```

---

### Step 6: Dynamic Market Quality

```python
# main_live.py Lines 618-657
# Analisis kualitas pasar berdasarkan:
# - Session (London/NY = tinggi, Sydney = rendah)
# - Regime (low vol = bagus, crisis = block)
# - Volatility (medium = ideal)
# - Trend strength
# - SMC confluence
# - ML signal alignment

quality_score = analyze_market_quality(...)
# EXCELLENT (80+), GOOD (60+), MODERATE (40+), POOR (20+), AVOID (<20), CRISIS

if quality == "AVOID" or quality == "CRISIS":
    return  # Pasar tidak layak untuk trading
```

---

### Step 7: Signal Confirmation (2 Bar Berturut)

```python
# main_live.py Lines 686-709
signal_key = f"{smc_signal.signal_type}_{smc_signal.entry_price:.0f}"

if signal_key in self._signal_persistence:
    self._signal_persistence[signal_key] += 1
else:
    self._signal_persistence[signal_key] = 1

if self._signal_persistence[signal_key] < 2:
    return  # Belum dikonfirmasi — tunggu 1 loop lagi

# Signal sudah muncul 2x berturut -> CONFIRMED
```

**Tujuan:** Mencegah whipsaw — signal yang hanya muncul 1 detik kemungkinan noise.

---

### Step 8: Pullback Filter

```python
# main_live.py Lines 742-871
can_enter, pullback_reason = self._check_pullback_filter(df, signal.signal_type)

if not can_enter:
    return  # Sedang pullback, tunggu momentum selaras
```

**Untuk signal BUY, block jika:**
- Harga turun > $2 dalam 3 candle terakhir
- MACD bearish + harga turun
- Harga jauh di bawah EMA9 + terus turun

**Untuk signal SELL, block jika:**
- Harga naik > $2 dalam 3 candle terakhir
- MACD bullish + harga naik
- Harga jauh di atas EMA9 + terus naik

**Komponen yang dicek:**

```
1. Short-term Momentum (3 candle terakhir)
   -> Arah pergerakan harga terkini

2. MACD Histogram
   -> Rising = bullish momentum
   -> Falling = bearish momentum

3. Harga vs EMA9
   -> Di atas = bullish bias
   -> Di bawah = bearish bias

4. RSI Extreme
   -> RSI > 80 = overbought (block BUY)
   -> RSI < 20 = oversold (block SELL)
```

---

### Step 9: Trade Cooldown

```python
# main_live.py Lines 520-524
trade_cooldown = 300  # 5 menit

if last_trade_time:
    elapsed = (now - last_trade_time).total_seconds()
    if elapsed < trade_cooldown:
        return  # Tunggu cooldown selesai
```

**Tujuan:** Mencegah overtrading — minimal 5 menit antar trade.

---

### Step 10: Position Limit

```python
# main_live.py Lines 588-592
can_open, limit_reason = self.smart_risk.can_open_position()

if not can_open:
    return  # Sudah 2 posisi terbuka (max)
```

---

### Step 11: Lot Size Calculation

```python
# main_live.py Lines 544-560
safe_lot = self.smart_risk.calculate_lot_size(
    entry_price=signal.entry_price,
    confidence=signal.confidence,        # SMC confidence
    regime=regime_name,                  # HMM regime
    ml_confidence=ml_prediction.confidence,  # ML confidence
)

# Apply session multiplier
safe_lot = max(0.01, safe_lot * session_multiplier)

if safe_lot <= 0:
    return  # Lot 0 = tidak boleh trade
```

---

## Eksekusi Order

Setelah semua 11 filter lolos:

```python
# main_live.py Lines 985-1008
# Step A: Ambil harga real-time
tick = mt5.get_tick(symbol)
current_price = tick.ask if BUY else tick.bid

# Step B: Validasi broker SL (min 10 pips)
broker_sl = signal.stop_loss
if jarak_terlalu_dekat:
    broker_sl = paksa_lebih_lebar

# Step C: Kirim order
result = mt5.send_order(
    symbol="XAUUSD",
    order_type="BUY" / "SELL",
    volume=0.01 - 0.02,          # Lot dari risk calculation
    sl=broker_sl,                # ATR-based SL (v3)
    tp=signal.take_profit,       # SMC TP (ATR-capped)
    magic=123456,                # ID bot
    comment="AI Safe v3",
)

# Step D: Fallback jika broker reject SL
if gagal dan error 10016:
    result = mt5.send_order(sl=0, ...)  # Tanpa broker SL

# Step E: Register posisi untuk monitoring
if result.success:
    smart_risk.register_position(
        ticket=result.order_id,
        entry_price=signal.entry_price,
        lot_size=position.lot_size,
        direction=signal.signal_type,
    )
```

---

## Post-Entry

```python
# Step F: Log trade detail
trade_logger.log_trade_open(
    signal, ml_prediction, regime, market_quality, ...
)

# Step G: Kirim notifikasi Telegram
await telegram.send_trade_open(trade_info)

# Step H: Update cooldown timer
last_trade_time = now
```

---

## Diagram Flow Lengkap

```
Loop setiap 1 detik
    |
    v
Fetch 200 bar M15 -> Feature Eng -> SMC -> HMM -> XGBoost
    |
    v
[1] Session OK? ----NO----> Skip
    |YES
[2] Risk OK? -------NO----> Skip (STOPPED)
    |YES
[3] SMC Signal? ----NO----> Skip (tidak ada setup)
    |YES
[4] ML >= 50%? -----NO----> Skip (terlalu uncertain)
    |YES
[5] ML Agree? ------NO----> Skip (ML veto)
    |YES
[6] Quality OK? ----NO----> Skip (AVOID/CRISIS)
    |YES
[7] Confirmed 2x? --NO----> Skip (tunggu konfirmasi)
    |YES
[8] No Pullback? ---NO----> Skip (retrace)
    |YES
[9] Cooldown OK? ---NO----> Skip (< 5 menit)
    |YES
[10] Pos < 2? ------NO----> Skip (full)
    |YES
[11] Lot > 0? ------NO----> Skip
    |YES
    v
EXECUTE TRADE -> Register -> Log -> Telegram
```

---

## Statistik Filter

Dalam kondisi normal, dari ratusan loop per jam:
- **~95%** diblokir oleh "tidak ada SMC signal" (pasar sideways)
- **~3%** diblokir oleh ML disagreement atau low confidence
- **~1%** diblokir oleh pullback filter atau session
- **<1%** lolos semua filter dan menghasilkan trade

**Rata-rata:** 3-8 trade per hari (sangat selektif).
