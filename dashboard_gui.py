"""
Trading Bot Dashboard - Live Monitoring GUI
============================================
Real-time view of SMC, ML, Market conditions, and system status.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import bot components
try:
    from src.mt5_connector import MT5Connector
    from src.smc_polars import SMCAnalyzer, SMCSignal
    from src.ml_model import TradingModel, PredictionResult
    from src.regime_detector import MarketRegimeDetector, RegimeState, MarketRegime
    from src.session_filter import SessionFilter
    from src.feature_eng import FeatureEngineer
    from src.smart_risk_manager import SmartRiskManager
    from src.config import TradingConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project directory")
    sys.exit(1)


class TradingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot Dashboard - Live Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')

        # Initialize components
        self.mt5 = None
        self.smc = None
        self.ml = None
        self.hmm = None
        self.analyzer = None
        self.feature_eng = None
        self.session = None
        self.config = TradingConfig()

        # State
        self.running = False
        self.last_update = None

        # Setup UI
        self.setup_styles()
        self.create_widgets()

        # Start connection
        self.connect_systems()

    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('Dashboard.TFrame', background='#1a1a2e')
        style.configure('Card.TFrame', background='#16213e')
        style.configure('Header.TLabel',
                       background='#1a1a2e',
                       foreground='#e94560',
                       font=('Segoe UI', 16, 'bold'))
        style.configure('CardTitle.TLabel',
                       background='#16213e',
                       foreground='#00d9ff',
                       font=('Segoe UI', 11, 'bold'))
        style.configure('Value.TLabel',
                       background='#16213e',
                       foreground='#ffffff',
                       font=('Consolas', 12))
        style.configure('ValueBig.TLabel',
                       background='#16213e',
                       foreground='#00ff88',
                       font=('Consolas', 18, 'bold'))
        style.configure('Buy.TLabel',
                       background='#16213e',
                       foreground='#00ff88',
                       font=('Consolas', 14, 'bold'))
        style.configure('Sell.TLabel',
                       background='#16213e',
                       foreground='#ff4757',
                       font=('Consolas', 14, 'bold'))
        style.configure('Hold.TLabel',
                       background='#16213e',
                       foreground='#ffa502',
                       font=('Consolas', 14, 'bold'))

    def create_widgets(self):
        """Create all dashboard widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dashboard.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header = ttk.Label(main_frame, text="🤖 TRADING BOT DASHBOARD", style='Header.TLabel')
        header.pack(pady=(0, 10))

        # Top row - Price and Account
        top_frame = ttk.Frame(main_frame, style='Dashboard.TFrame')
        top_frame.pack(fill=tk.X, pady=5)

        self.create_price_card(top_frame)
        self.create_account_card(top_frame)
        self.create_session_card(top_frame)

        # Middle row - Signals
        mid_frame = ttk.Frame(main_frame, style='Dashboard.TFrame')
        mid_frame.pack(fill=tk.X, pady=5)

        self.create_smc_card(mid_frame)
        self.create_ml_card(mid_frame)
        self.create_regime_card(mid_frame)

        # Third row - Risk and Positions
        third_frame = ttk.Frame(main_frame, style='Dashboard.TFrame')
        third_frame.pack(fill=tk.X, pady=5)

        self.create_risk_card(third_frame)
        self.create_positions_card(third_frame)

        # Bottom - Settings and Log
        bottom_frame = ttk.Frame(main_frame, style='Dashboard.TFrame')
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.create_settings_card(bottom_frame)
        self.create_log_card(bottom_frame)

        # Status bar
        self.status_var = tk.StringVar(value="Connecting...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              background='#0f3460', foreground='#ffffff',
                              font=('Segoe UI', 9))
        status_bar.pack(fill=tk.X, pady=(5, 0))

    def create_card(self, parent, title, width=280):
        """Create a styled card frame"""
        card = tk.Frame(parent, bg='#16213e', bd=1, relief=tk.RAISED)
        card.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        title_label = ttk.Label(card, text=title, style='CardTitle.TLabel')
        title_label.pack(pady=(10, 5), padx=10, anchor='w')

        content = tk.Frame(card, bg='#16213e')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        return content

    def create_price_card(self, parent):
        """Price information card"""
        content = self.create_card(parent, "📊 PRICE")

        self.price_var = tk.StringVar(value="---.--")
        price_label = tk.Label(content, textvariable=self.price_var,
                              bg='#16213e', fg='#00ff88',
                              font=('Consolas', 24, 'bold'))
        price_label.pack(pady=5)

        self.spread_var = tk.StringVar(value="Spread: -- pips")
        spread_label = tk.Label(content, textvariable=self.spread_var,
                               bg='#16213e', fg='#888888',
                               font=('Consolas', 10))
        spread_label.pack()

        self.time_var = tk.StringVar(value="--:--:--")
        time_label = tk.Label(content, textvariable=self.time_var,
                             bg='#16213e', fg='#aaaaaa',
                             font=('Consolas', 10))
        time_label.pack()

    def create_account_card(self, parent):
        """Account information card"""
        content = self.create_card(parent, "💰 ACCOUNT")

        self.balance_var = tk.StringVar(value="Balance: $---.--")
        balance_label = tk.Label(content, textvariable=self.balance_var,
                                bg='#16213e', fg='#ffffff',
                                font=('Consolas', 12))
        balance_label.pack(anchor='w', pady=2)

        self.equity_var = tk.StringVar(value="Equity: $---.--")
        equity_label = tk.Label(content, textvariable=self.equity_var,
                               bg='#16213e', fg='#ffffff',
                               font=('Consolas', 12))
        equity_label.pack(anchor='w', pady=2)

        self.profit_var = tk.StringVar(value="P/L: $0.00")
        profit_label = tk.Label(content, textvariable=self.profit_var,
                               bg='#16213e', fg='#00ff88',
                               font=('Consolas', 12, 'bold'))
        profit_label.pack(anchor='w', pady=2)

    def create_session_card(self, parent):
        """Session information card"""
        content = self.create_card(parent, "🕐 SESSION")

        self.session_var = tk.StringVar(value="Loading...")
        session_label = tk.Label(content, textvariable=self.session_var,
                                bg='#16213e', fg='#ffa502',
                                font=('Consolas', 11, 'bold'))
        session_label.pack(anchor='w', pady=2)

        self.golden_var = tk.StringVar(value="Golden Time: --")
        golden_label = tk.Label(content, textvariable=self.golden_var,
                               bg='#16213e', fg='#ffcc00',
                               font=('Consolas', 11))
        golden_label.pack(anchor='w', pady=2)

        self.can_trade_var = tk.StringVar(value="Can Trade: --")
        can_trade_label = tk.Label(content, textvariable=self.can_trade_var,
                                  bg='#16213e', fg='#00ff88',
                                  font=('Consolas', 11))
        can_trade_label.pack(anchor='w', pady=2)

    def create_smc_card(self, parent):
        """SMC Signal card"""
        content = self.create_card(parent, "📈 SMC SIGNAL")

        self.smc_signal_var = tk.StringVar(value="---")
        self.smc_signal_label = tk.Label(content, textvariable=self.smc_signal_var,
                                         bg='#16213e', fg='#888888',
                                         font=('Consolas', 20, 'bold'))
        self.smc_signal_label.pack(pady=5)

        self.smc_conf_var = tk.StringVar(value="Confidence: --%")
        smc_conf_label = tk.Label(content, textvariable=self.smc_conf_var,
                                 bg='#16213e', fg='#aaaaaa',
                                 font=('Consolas', 10))
        smc_conf_label.pack()

        self.smc_reason_var = tk.StringVar(value="")
        smc_reason_label = tk.Label(content, textvariable=self.smc_reason_var,
                                   bg='#16213e', fg='#666666',
                                   font=('Consolas', 9), wraplength=250)
        smc_reason_label.pack(pady=5)

    def create_ml_card(self, parent):
        """ML Prediction card"""
        content = self.create_card(parent, "🤖 ML PREDICTION")

        self.ml_signal_var = tk.StringVar(value="---")
        self.ml_signal_label = tk.Label(content, textvariable=self.ml_signal_var,
                                        bg='#16213e', fg='#888888',
                                        font=('Consolas', 20, 'bold'))
        self.ml_signal_label.pack(pady=5)

        self.ml_conf_var = tk.StringVar(value="Confidence: --%")
        ml_conf_label = tk.Label(content, textvariable=self.ml_conf_var,
                                bg='#16213e', fg='#aaaaaa',
                                font=('Consolas', 10))
        ml_conf_label.pack()

        self.ml_prob_var = tk.StringVar(value="Buy: --% | Sell: --%")
        ml_prob_label = tk.Label(content, textvariable=self.ml_prob_var,
                                bg='#16213e', fg='#666666',
                                font=('Consolas', 9))
        ml_prob_label.pack(pady=5)

    def create_regime_card(self, parent):
        """Market Regime card"""
        content = self.create_card(parent, "🌊 MARKET REGIME")

        self.regime_var = tk.StringVar(value="---")
        self.regime_label = tk.Label(content, textvariable=self.regime_var,
                                     bg='#16213e', fg='#888888',
                                     font=('Consolas', 14, 'bold'))
        self.regime_label.pack(pady=5)

        self.volatility_var = tk.StringVar(value="Volatility: --")
        vol_label = tk.Label(content, textvariable=self.volatility_var,
                            bg='#16213e', fg='#aaaaaa',
                            font=('Consolas', 10))
        vol_label.pack()

        self.atr_var = tk.StringVar(value="ATR: --")
        atr_label = tk.Label(content, textvariable=self.atr_var,
                            bg='#16213e', fg='#666666',
                            font=('Consolas', 9))
        atr_label.pack(pady=5)

    def create_risk_card(self, parent):
        """Risk Management card"""
        content = self.create_card(parent, "⚠️ RISK STATUS")

        self.daily_loss_var = tk.StringVar(value="Daily Loss: $0.00")
        daily_loss_label = tk.Label(content, textvariable=self.daily_loss_var,
                                   bg='#16213e', fg='#ff4757',
                                   font=('Consolas', 11))
        daily_loss_label.pack(anchor='w', pady=2)

        self.daily_profit_var = tk.StringVar(value="Daily Profit: $0.00")
        daily_profit_label = tk.Label(content, textvariable=self.daily_profit_var,
                                     bg='#16213e', fg='#00ff88',
                                     font=('Consolas', 11))
        daily_profit_label.pack(anchor='w', pady=2)

        self.total_loss_var = tk.StringVar(value="Total Loss: $0.00")
        total_loss_label = tk.Label(content, textvariable=self.total_loss_var,
                                   bg='#16213e', fg='#ffa502',
                                   font=('Consolas', 11))
        total_loss_label.pack(anchor='w', pady=2)

        self.consec_loss_var = tk.StringVar(value="Consecutive Losses: 0")
        consec_label = tk.Label(content, textvariable=self.consec_loss_var,
                               bg='#16213e', fg='#aaaaaa',
                               font=('Consolas', 10))
        consec_label.pack(anchor='w', pady=2)

    def create_positions_card(self, parent):
        """Open Positions card"""
        card = tk.Frame(parent, bg='#16213e', bd=1, relief=tk.RAISED)
        card.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        title_label = ttk.Label(card, text="📋 OPEN POSITIONS", style='CardTitle.TLabel')
        title_label.pack(pady=(10, 5), padx=10, anchor='w')

        # Positions listbox
        self.positions_text = tk.Text(card, height=5, bg='#0f3460', fg='#ffffff',
                                      font=('Consolas', 9), bd=0)
        self.positions_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.positions_text.insert('1.0', "No open positions")
        self.positions_text.config(state=tk.DISABLED)

    def create_settings_card(self, parent):
        """Settings display card"""
        card = tk.Frame(parent, bg='#16213e', bd=1, relief=tk.RAISED, width=350)
        card.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH)
        card.pack_propagate(False)

        title_label = ttk.Label(card, text="⚙️ SETTINGS", style='CardTitle.TLabel')
        title_label.pack(pady=(10, 5), padx=10, anchor='w')

        content = tk.Frame(card, bg='#16213e')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        settings_info = [
            ("Symbol:", self.config.symbol),
            ("Capital:", f"${self.config.capital:,.2f}"),
            ("Max Daily Loss:", f"{self.config.risk.max_daily_loss}%"),
            ("Risk Per Trade:", f"{self.config.risk.risk_per_trade}%"),
            ("Min Lot:", f"{self.config.risk.min_lot_size}"),
            ("Max Lot:", f"{self.config.risk.max_lot_size}"),
            ("Timeframe:", self.config.execution_timeframe),
            ("Golden Time:", "19:00-23:00 WIB"),
        ]

        for label, value in settings_info:
            row = tk.Frame(content, bg='#16213e')
            row.pack(fill=tk.X, pady=1)

            lbl = tk.Label(row, text=label, bg='#16213e', fg='#888888',
                          font=('Consolas', 9), width=15, anchor='w')
            lbl.pack(side=tk.LEFT)

            val = tk.Label(row, text=value, bg='#16213e', fg='#ffffff',
                          font=('Consolas', 9), anchor='w')
            val.pack(side=tk.LEFT)

    def create_log_card(self, parent):
        """Activity log card"""
        card = tk.Frame(parent, bg='#16213e', bd=1, relief=tk.RAISED)
        card.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        title_label = ttk.Label(card, text="📝 ACTIVITY LOG", style='CardTitle.TLabel')
        title_label.pack(pady=(10, 5), padx=10, anchor='w')

        self.log_text = scrolledtext.ScrolledText(card, height=8, bg='#0f3460', fg='#00ff88',
                                                  font=('Consolas', 9), bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def log(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def connect_systems(self):
        """Connect to MT5 and initialize components"""
        self.log("Connecting to MT5...")

        try:
            self.mt5 = MT5Connector(
                login=self.config.mt5_login,
                password=self.config.mt5_password,
                server=self.config.mt5_server,
                path=self.config.mt5_path,
            )
            if not self.mt5.connect():
                self.log("ERROR: Failed to connect to MT5!")
                return

            self.log("MT5 connected successfully!")

            # Initialize components
            self.smc = SMCAnalyzer()
            self.ml = TradingModel(model_path="models/xgboost_model")
            self.ml.load()  # Load saved model
            self.hmm = MarketRegimeDetector(model_path="models/hmm_regime")
            self.hmm.load()  # Load saved regime model
            self.session = SessionFilter()
            self.feature_eng = FeatureEngineer()

            self.log("All components initialized")
            if self.ml.fitted:
                self.log("ML Model loaded successfully")
            else:
                self.log("WARNING: ML Model not fitted")
            self.running = True

            # Start update thread
            update_thread = threading.Thread(target=self.update_loop, daemon=True)
            update_thread.start()

        except Exception as e:
            self.log(f"ERROR: {e}")

    def update_loop(self):
        """Main update loop running in background thread"""
        while self.running:
            try:
                self.update_data()
                time.sleep(1)  # Update every second
            except Exception as e:
                self.log(f"Update error: {e}")
                time.sleep(5)

    def update_data(self):
        """Fetch and update all data"""
        if not self.mt5:
            return

        # Get current time
        wib = ZoneInfo("Asia/Jakarta")
        now = datetime.now(wib)
        self.time_var.set(now.strftime("%H:%M:%S WIB"))

        # Check golden time
        is_golden = 19 <= now.hour <= 23
        self.golden_var.set(f"Golden Time: {'YES 🌟' if is_golden else 'NO'}")

        # Get price data
        try:
            tick = self.mt5.get_tick(self.config.symbol)
            if tick:
                price = (tick.bid + tick.ask) / 2
                spread = (tick.ask - tick.bid) * 100  # in pips
                self.price_var.set(f"{price:.2f}")
                self.spread_var.set(f"Spread: {spread:.1f} pips")
        except:
            pass

        # Get account info
        try:
            balance = self.mt5.account_balance
            equity = self.mt5.account_equity
            profit = equity - balance
            self.balance_var.set(f"Balance: ${balance:,.2f}")
            self.equity_var.set(f"Equity: ${equity:,.2f}")
            self.profit_var.set(f"P/L: ${profit:+,.2f}")
        except:
            pass

        # Get session info
        try:
            session_info = self.session.get_status_report()
            if session_info:
                current_session = session_info.get('current_session', 'Unknown')
                self.session_var.set(current_session)
                can_trade, reason, mult = self.session.can_trade()
                self.can_trade_var.set(f"Can Trade: {'YES ✓' if can_trade else 'NO ✗'}")
        except:
            pass

        # Get market data for analysis
        try:
            df = self.mt5.get_market_data(self.config.symbol, self.config.execution_timeframe, 500)
            if df is not None and len(df) > 100:
                self.update_signals(df)
        except Exception as e:
            self.log(f"Data error: {e}")

        # Get positions
        try:
            positions = self.mt5.get_open_positions(self.config.symbol)
            self.update_positions(positions)
        except:
            pass

        # Update risk state
        self.update_risk_state()

        # Update status
        self.last_update = datetime.now()
        self.status_var.set(f"Last update: {self.last_update.strftime('%H:%M:%S')} | Running...")

    def update_signals(self, df):
        """Update SMC and ML signals"""
        # Build complete feature DataFrame (same as main_live.py)
        # 1. Technical features
        df = self.feature_eng.calculate_all(df, include_ml_features=True)

        # 2. SMC features
        df = self.smc.calculate_all(df)

        # 3. Regime detection
        try:
            df = self.hmm.predict(df)
            regime = self.hmm.get_current_state(df)
            if regime:
                regime_name = regime.regime.value.replace('_', ' ').title()
                self.regime_var.set(regime_name)
                self.volatility_var.set(f"Volatility: {regime.volatility:.2f}")
                self.atr_var.set(f"Conf: {regime.confidence:.0%}")

                # Update color based on regime
                if "HIGH" in regime.regime.value:
                    self.regime_label.config(fg='#ff4757')
                elif "LOW" in regime.regime.value:
                    self.regime_label.config(fg='#00ff88')
                else:
                    self.regime_label.config(fg='#ffa502')
        except Exception as e:
            pass

        # 4. SMC Signal
        try:
            smc_signal = self.smc.generate_signal(df)
            if smc_signal:
                signal_type = smc_signal.signal_type
                self.smc_signal_var.set(signal_type)
                self.smc_conf_var.set(f"Confidence: {smc_signal.confidence:.0%}")
                self.smc_reason_var.set(smc_signal.reason[:50] + "..." if len(smc_signal.reason) > 50 else smc_signal.reason)

                # Update color
                if signal_type == "BUY":
                    self.smc_signal_label.config(fg='#00ff88')
                elif signal_type == "SELL":
                    self.smc_signal_label.config(fg='#ff4757')
            else:
                self.smc_signal_var.set("NO SIGNAL")
                self.smc_signal_label.config(fg='#888888')
                self.smc_conf_var.set("Confidence: --%")
                self.smc_reason_var.set("")
        except:
            pass

        # 5. ML Prediction (with all features now available)
        try:
            if self.ml.fitted:
                # Get available features that match model's expected features
                available_features = [f for f in self.ml.feature_names if f in df.columns]
                ml_pred = self.ml.predict(df, available_features)
                if ml_pred:
                    signal = ml_pred.signal
                    self.ml_signal_var.set(signal)
                    self.ml_conf_var.set(f"Confidence: {ml_pred.confidence:.0%}")
                    # probability is for BUY, so sell_prob = 1 - probability
                    buy_prob = ml_pred.probability
                    sell_prob = 1.0 - buy_prob
                    self.ml_prob_var.set(f"Buy: {buy_prob:.0%} | Sell: {sell_prob:.0%}")

                    # Update color
                    if signal == "BUY":
                        self.ml_signal_label.config(fg='#00ff88')
                    elif signal == "SELL":
                        self.ml_signal_label.config(fg='#ff4757')
                    else:
                        self.ml_signal_label.config(fg='#ffa502')
        except:
            pass

    def update_positions(self, positions):
        """Update positions display (positions is a Polars DataFrame)"""
        self.positions_text.config(state=tk.NORMAL)
        self.positions_text.delete('1.0', tk.END)

        if positions is None or positions.is_empty():
            self.positions_text.insert('1.0', "No open positions")
        else:
            for row in positions.iter_rows(named=True):
                ticket = row.get('ticket', 'N/A')
                pos_type = "BUY" if row.get('type', 0) == 0 else "SELL"
                volume = row.get('volume', 0)
                profit = row.get('profit', 0)
                price_open = row.get('price_open', 0)

                line = f"#{ticket} | {pos_type} {volume} @ {price_open:.2f} | P/L: ${profit:+.2f}\n"
                self.positions_text.insert(tk.END, line)

        self.positions_text.config(state=tk.DISABLED)

    def update_risk_state(self):
        """Update risk state from file"""
        try:
            risk_file = Path("data/risk_state.txt")
            if risk_file.exists():
                content = risk_file.read_text()
                lines = content.strip().split('\n')

                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        if key == 'daily_loss':
                            self.daily_loss_var.set(f"Daily Loss: ${float(value):,.2f}")
                        elif key == 'daily_profit':
                            self.daily_profit_var.set(f"Daily Profit: ${float(value):,.2f}")
                        elif key == 'total_loss':
                            self.total_loss_var.set(f"Total Loss: ${float(value):,.2f}")
                        elif key == 'consecutive_losses':
                            self.consec_loss_var.set(f"Consecutive Losses: {value}")
        except:
            pass

    def on_closing(self):
        """Handle window close"""
        self.running = False
        if self.mt5:
            self.mt5.disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TradingDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
