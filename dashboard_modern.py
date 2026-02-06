"""
Trading Bot Dashboard - Modern UI with CustomTkinter
=====================================================
Beautiful responsive dashboard with dark/light mode toggle.
"""

import customtkinter as ctk
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import bot components
try:
    from src.mt5_connector import MT5Connector
    from src.smc_polars import SMCAnalyzer
    from src.ml_model import TradingModel
    from src.regime_detector import MarketRegimeDetector
    from src.session_filter import SessionFilter
    from src.feature_eng import FeatureEngineer
    from src.config import TradingConfig
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set default appearance
ctk.set_default_color_theme("blue")

# Theme colors
THEMES = {
    "dark": {
        "bg": "#0d0d1a",
        "card": "#1a1a2e",
        "card_inner": "#0d0d1a",
        "text": "#ffffff",
        "text_secondary": "#888888",
        "text_muted": "#666666",
        "accent": "#00d4ff",
        "green": "#00ff88",
        "red": "#ff4757",
        "orange": "#ffa500",
        "highlight": "#2d4a2d",
        "highlight_off": "#2d2d44",
    },
    "light": {
        "bg": "#f0f2f5",
        "card": "#ffffff",
        "card_inner": "#f8f9fa",
        "text": "#1a1a2e",
        "text_secondary": "#555555",
        "text_muted": "#888888",
        "accent": "#0066cc",
        "green": "#00aa55",
        "red": "#dd3344",
        "orange": "#ee8800",
        "highlight": "#d4edda",
        "highlight_off": "#e9ecef",
    }
}


class ModernCard(ctk.CTkFrame):
    """Reusable card component with title - theme aware"""

    def __init__(self, master, title, theme="dark", **kwargs):
        self.theme = theme
        colors = THEMES[theme]
        super().__init__(master, corner_radius=12, fg_color=colors["card"], **kwargs)

        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=colors["accent"]
        )
        self.title_label.pack(anchor="w", padx=12, pady=(10, 6))

        # Content frame
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def update_theme(self, theme):
        """Update card colors for theme"""
        self.theme = theme
        colors = THEMES[theme]
        self.configure(fg_color=colors["card"])
        self.title_label.configure(text_color=colors["accent"])


class TradingDashboard(ctk.CTk):
    """Modern Trading Dashboard with CustomTkinter - Responsive & Theme Toggle"""

    def __init__(self):
        super().__init__()

        # Theme state
        self.current_theme = "dark"
        ctk.set_appearance_mode("dark")

        # Window setup - responsive minimum size for split screen
        self.title("AI Trading Bot")
        self.geometry("700x800")
        self.minsize(600, 700)
        self.configure(fg_color=THEMES[self.current_theme]["bg"])

        # Store all themed widgets for updates
        self.themed_widgets = []
        self.cards = []

        # Initialize components
        self.mt5 = None
        self.smc = None
        self.ml = None
        self.hmm = None
        self.session = None
        self.feature_eng = None
        self.config = TradingConfig()

        # State
        self.running = False
        self.last_update = None

        # Create UI
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()

        # Start connection
        self.after(100, self.connect_systems)

    def toggle_theme(self):
        """Toggle between dark and light mode"""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        ctk.set_appearance_mode(self.current_theme)
        self.apply_theme()

    def apply_theme(self):
        """Apply current theme to all widgets"""
        colors = THEMES[self.current_theme]

        # Update main window
        self.configure(fg_color=colors["bg"])

        # Update theme button
        if self.current_theme == "dark":
            self.theme_btn.configure(text="☀️ Light")
        else:
            self.theme_btn.configure(text="🌙 Dark")

        # Update all cards
        for card in self.cards:
            card.update_theme(self.current_theme)

        # Update header
        self.title_label.configure(text_color=colors["text"])
        self.subtitle_label.configure(text_color=colors["text_muted"])
        self.time_label.configure(text_color=colors["text_secondary"])

        # Update status bar
        self.status_frame.configure(fg_color=colors["card"])
        self.status_label.configure(text_color=colors["text_secondary"])
        self.update_label.configure(text_color=colors["text_muted"])

        # Update text boxes
        self.log_text.configure(fg_color=colors["card_inner"], text_color=colors["green"])
        self.positions_text.configure(fg_color=colors["card_inner"], text_color=colors["text"])

        # Update settings labels
        if hasattr(self, 'settings_labels'):
            for lbl, val_lbl in self.settings_labels:
                lbl.configure(text_color=colors["text_secondary"])
                val_lbl.configure(text_color=colors["text"])

        # Update golden time frame
        is_golden = hasattr(self, 'golden_frame')
        if is_golden:
            # Reapply golden time based on current state
            current_text = self.golden_label.cget("text")
            if "YES" in current_text:
                self.golden_frame.configure(fg_color=colors["highlight"])
                self.golden_label.configure(text_color=colors["green"])
            else:
                self.golden_frame.configure(fg_color=colors["highlight_off"])
                self.golden_label.configure(text_color=colors["text_secondary"])

    def create_header(self):
        """Create header section"""
        colors = THEMES[self.current_theme]

        header_frame = ctk.CTkFrame(self, fg_color="transparent", height=50)
        header_frame.pack(fill="x", padx=15, pady=(10, 5))
        header_frame.pack_propagate(False)

        # Logo/Title
        self.title_label = ctk.CTkLabel(
            header_frame,
            text="AI TRADING BOT",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=colors["text"]
        )
        self.title_label.pack(side="left")

        self.subtitle_label = ctk.CTkLabel(
            header_frame,
            text="  Live",
            font=ctk.CTkFont(size=12),
            text_color=colors["text_muted"]
        )
        self.subtitle_label.pack(side="left", pady=(4, 0))

        # Theme toggle button
        self.theme_btn = ctk.CTkButton(
            header_frame,
            text="☀️ Light",
            width=80,
            height=28,
            corner_radius=14,
            font=ctk.CTkFont(size=11),
            command=self.toggle_theme
        )
        self.theme_btn.pack(side="right", padx=5)

        # Connection status
        self.connection_label = ctk.CTkLabel(
            header_frame,
            text="● Connecting...",
            font=ctk.CTkFont(size=12),
            text_color=colors["orange"]
        )
        self.connection_label.pack(side="right", padx=10)

        # Time display
        self.time_label = ctk.CTkLabel(
            header_frame,
            text="--:--:-- WIB",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=colors["text_secondary"]
        )
        self.time_label.pack(side="right", padx=10)

    def create_main_layout(self):
        """Create main dashboard layout - responsive 2-column for split screen"""
        # Scrollable main container for small windows
        self.main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_scroll.pack(fill="both", expand=True, padx=10, pady=5)

        # Configure 2-column grid (responsive for split screen)
        self.main_scroll.grid_columnconfigure(0, weight=1, minsize=280)
        self.main_scroll.grid_columnconfigure(1, weight=1, minsize=280)

        # Row 0: Price & Account
        self.create_price_card(self.main_scroll, 0, 0)
        self.create_account_card(self.main_scroll, 0, 1)

        # Row 1: Session & Risk
        self.create_session_card(self.main_scroll, 1, 0)
        self.create_risk_card(self.main_scroll, 1, 1)

        # Row 2: SMC & ML
        self.create_smc_card(self.main_scroll, 2, 0)
        self.create_ml_card(self.main_scroll, 2, 1)

        # Row 3: Regime & Positions
        self.create_regime_card(self.main_scroll, 3, 0)
        self.create_positions_card(self.main_scroll, 3, 1)

        # Row 4: Settings (full width)
        self.create_settings_card(self.main_scroll, 4, 0)

        # Row 5: Log (full width)
        self.create_log_card(self.main_scroll, 5, 0)

    def create_price_card(self, parent, row, col):
        """Price information card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "PRICE", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Price value
        self.price_label = ctk.CTkLabel(
            card.content,
            text="-----.--",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=colors["green"]
        )
        self.price_label.pack(pady=(5, 0))

        # Symbol
        self.symbol_label = ctk.CTkLabel(
            card.content,
            text="XAUUSD",
            font=ctk.CTkFont(size=11),
            text_color=colors["text_muted"]
        )
        self.symbol_label.pack()

        # Spread
        self.spread_label = ctk.CTkLabel(
            card.content,
            text="Spread: -- pips",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.spread_label.pack(pady=(3, 0))

    def create_account_card(self, parent, row, col):
        """Account information card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "ACCOUNT", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Balance
        balance_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        balance_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(balance_frame, text="Balance:", font=ctk.CTkFont(size=11), text_color=colors["text_secondary"]).pack(side="left")
        self.balance_label = ctk.CTkLabel(balance_frame, text="$-----.--", font=ctk.CTkFont(size=11, weight="bold"), text_color=colors["text"])
        self.balance_label.pack(side="right")

        # Equity
        equity_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        equity_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(equity_frame, text="Equity:", font=ctk.CTkFont(size=11), text_color=colors["text_secondary"]).pack(side="left")
        self.equity_label = ctk.CTkLabel(equity_frame, text="$-----.--", font=ctk.CTkFont(size=11, weight="bold"), text_color=colors["text"])
        self.equity_label.pack(side="right")

        # P/L
        pl_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        pl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(pl_frame, text="P/L:", font=ctk.CTkFont(size=11), text_color=colors["text_secondary"]).pack(side="left")
        self.pl_label = ctk.CTkLabel(pl_frame, text="$0.00", font=ctk.CTkFont(size=12, weight="bold"), text_color=colors["green"])
        self.pl_label.pack(side="right")

    def create_session_card(self, parent, row, col):
        """Session information card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "SESSION", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Current session
        self.session_label = ctk.CTkLabel(
            card.content,
            text="Loading...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=colors["orange"]
        )
        self.session_label.pack(pady=(3, 0))

        # Golden time indicator
        self.golden_frame = ctk.CTkFrame(card.content, fg_color=colors["highlight_off"], corner_radius=6)
        self.golden_frame.pack(pady=6, padx=3, fill="x")
        self.golden_label = ctk.CTkLabel(
            self.golden_frame,
            text="GOLDEN: --",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=colors["text_secondary"]
        )
        self.golden_label.pack(pady=4)

        # Can trade
        self.can_trade_label = ctk.CTkLabel(
            card.content,
            text="Can Trade: --",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.can_trade_label.pack()

    def create_risk_card(self, parent, row, col):
        """Risk status card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "RISK STATUS", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Daily loss
        dl_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        dl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(dl_frame, text="Daily Loss:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.daily_loss_label = ctk.CTkLabel(dl_frame, text="$0.00", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["red"])
        self.daily_loss_label.pack(side="right")

        # Daily profit
        dp_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        dp_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(dp_frame, text="Daily Profit:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.daily_profit_label = ctk.CTkLabel(dp_frame, text="$0.00", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["green"])
        self.daily_profit_label.pack(side="right")

        # Consecutive losses
        cl_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        cl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(cl_frame, text="Consec. Losses:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.consec_loss_label = ctk.CTkLabel(cl_frame, text="0", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["text"])
        self.consec_loss_label.pack(side="right")

    def create_smc_card(self, parent, row, col):
        """SMC Signal card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "SMC SIGNAL", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Signal
        self.smc_signal_label = ctk.CTkLabel(
            card.content,
            text="---",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=colors["text_muted"]
        )
        self.smc_signal_label.pack(pady=(5, 3))

        # Confidence bar
        self.smc_confidence_bar = ctk.CTkProgressBar(card.content, height=6, corner_radius=3)
        self.smc_confidence_bar.pack(pady=4, fill="x", padx=10)
        self.smc_confidence_bar.set(0)

        # Confidence text
        self.smc_conf_label = ctk.CTkLabel(
            card.content,
            text="Confidence: --%",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.smc_conf_label.pack()

        # Reason
        self.smc_reason_label = ctk.CTkLabel(
            card.content,
            text="Waiting...",
            font=ctk.CTkFont(size=9),
            text_color=colors["text_muted"],
            wraplength=250
        )
        self.smc_reason_label.pack(pady=(3, 0))

    def create_ml_card(self, parent, row, col):
        """ML Prediction card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "ML PREDICTION", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Signal
        self.ml_signal_label = ctk.CTkLabel(
            card.content,
            text="---",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=colors["text_muted"]
        )
        self.ml_signal_label.pack(pady=(5, 3))

        # Confidence bar
        self.ml_confidence_bar = ctk.CTkProgressBar(card.content, height=6, corner_radius=3)
        self.ml_confidence_bar.pack(pady=4, fill="x", padx=10)
        self.ml_confidence_bar.set(0)

        # Confidence text
        self.ml_conf_label = ctk.CTkLabel(
            card.content,
            text="Confidence: --%",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.ml_conf_label.pack()

        # Probabilities
        self.ml_prob_label = ctk.CTkLabel(
            card.content,
            text="Buy: --% | Sell: --%",
            font=ctk.CTkFont(size=9),
            text_color=colors["text_muted"]
        )
        self.ml_prob_label.pack(pady=(3, 0))

    def create_regime_card(self, parent, row, col):
        """Market Regime card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "MARKET REGIME", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Regime
        self.regime_label = ctk.CTkLabel(
            card.content,
            text="---",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=colors["text_muted"]
        )
        self.regime_label.pack(pady=(8, 5))

        # Volatility
        vol_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        vol_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(vol_frame, text="Volatility:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.volatility_label = ctk.CTkLabel(vol_frame, text="--", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["text"])
        self.volatility_label.pack(side="right")

        # Confidence
        conf_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        conf_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(conf_frame, text="Confidence:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.regime_conf_label = ctk.CTkLabel(conf_frame, text="--%", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["text"])
        self.regime_conf_label.pack(side="right")

    def create_positions_card(self, parent, row, col):
        """Open Positions card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "OPEN POSITIONS", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Positions text
        self.positions_text = ctk.CTkTextbox(
            card.content,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=colors["card_inner"],
            text_color=colors["text"],
            height=90,
            corner_radius=6
        )
        self.positions_text.pack(fill="both", expand=True)
        self.positions_text.insert("1.0", "No open positions")
        self.positions_text.configure(state="disabled")

    def create_settings_card(self, parent, row, col):
        """Settings display card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "SETTINGS", theme=self.current_theme)
        card.grid(row=row, column=col, columnspan=2, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Settings grid - use 2 columns for compact layout
        settings_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        settings_frame.pack(fill="both", expand=True)
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(1, weight=1)

        settings = [
            ("Symbol", self.config.symbol),
            ("Capital", f"${self.config.capital:,.2f}"),
            ("Max Daily Loss", f"{self.config.risk.max_daily_loss}%"),
            ("Risk Per Trade", f"{self.config.risk.risk_per_trade}%"),
            ("Min Lot", f"{self.config.risk.min_lot_size}"),
            ("Max Lot", f"{self.config.risk.max_lot_size}"),
            ("Timeframe", self.config.execution_timeframe),
            ("Golden Time", "19:00-23:00 WIB"),
        ]

        self.settings_labels = []  # Store for theme updates

        for i, (label, value) in enumerate(settings):
            row_idx = i // 2
            col_idx = i % 2

            row_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
            row_frame.grid(row=row_idx, column=col_idx, sticky="w", padx=5, pady=1)

            lbl = ctk.CTkLabel(
                row_frame,
                text=f"{label}:",
                font=ctk.CTkFont(size=10),
                text_color=colors["text_secondary"],
                anchor="w"
            )
            lbl.pack(side="left")

            val_lbl = ctk.CTkLabel(
                row_frame,
                text=f" {value}",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=colors["text"],
                anchor="w"
            )
            val_lbl.pack(side="left")

            self.settings_labels.append((lbl, val_lbl))

    def create_log_card(self, parent, row, col):
        """Activity log card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "ACTIVITY LOG", theme=self.current_theme)
        card.grid(row=row, column=col, columnspan=2, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        self.log_text = ctk.CTkTextbox(
            card.content,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=colors["card_inner"],
            text_color=colors["green"],
            height=100,
            corner_radius=6
        )
        self.log_text.pack(fill="both", expand=True)

    def create_status_bar(self):
        """Create status bar"""
        colors = THEMES[self.current_theme]

        self.status_frame = ctk.CTkFrame(self, fg_color=colors["card"], height=28, corner_radius=0)
        self.status_frame.pack(fill="x", side="bottom")
        self.status_frame.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Initializing...",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.status_label.pack(side="left", padx=12, pady=4)

        self.update_label = ctk.CTkLabel(
            self.status_frame,
            text="Last update: --:--:--",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_muted"]
        )
        self.update_label.pack(side="right", padx=12, pady=4)

    def log(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def connect_systems(self):
        """Connect to MT5 and initialize components"""
        colors = THEMES[self.current_theme]
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
                self.connection_label.configure(text="● Disconnected", text_color=colors["red"])
                return

            self.log("MT5 connected successfully!")
            self.connection_label.configure(text="● Connected", text_color=colors["green"])

            # Initialize components
            self.smc = SMCAnalyzer()
            self.ml = TradingModel(model_path="models/xgboost_model")
            self.ml.load()
            self.hmm = MarketRegimeDetector(model_path="models/hmm_regime")
            self.hmm.load()
            self.session = SessionFilter()
            self.feature_eng = FeatureEngineer()

            self.log("All components initialized")
            if self.ml.fitted:
                self.log(f"ML Model loaded ({len(self.ml.feature_names)} features)")

            self.running = True

            # Start update thread
            update_thread = threading.Thread(target=self.update_loop, daemon=True)
            update_thread.start()

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.connection_label.configure(text="● Error", text_color=colors["red"])

    def update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                self.after(0, self.update_data)
                time.sleep(1)
            except Exception as e:
                self.after(0, lambda: self.log(f"Update error: {e}"))
                time.sleep(5)

    def update_data(self):
        """Fetch and update all data"""
        if not self.mt5:
            return

        # Update time
        wib = ZoneInfo("Asia/Jakarta")
        now = datetime.now(wib)
        self.time_label.configure(text=now.strftime("%H:%M:%S WIB"))

        # Check golden time (use theme colors)
        colors = THEMES[self.current_theme]
        is_golden = 19 <= now.hour < 23
        if is_golden:
            self.golden_frame.configure(fg_color=colors["highlight"])
            self.golden_label.configure(text="GOLDEN TIME: YES", text_color=colors["green"])
        else:
            self.golden_frame.configure(fg_color=colors["highlight_off"])
            self.golden_label.configure(text="GOLDEN TIME: NO", text_color=colors["text_secondary"])

        # Update price
        try:
            tick = self.mt5.get_tick(self.config.symbol)
            if tick:
                price = (tick.bid + tick.ask) / 2
                spread = (tick.ask - tick.bid) * 100
                self.price_label.configure(text=f"{price:.2f}")
                self.spread_label.configure(text=f"Spread: {spread:.1f} pips")
        except:
            pass

        # Update account
        try:
            balance = self.mt5.account_balance
            equity = self.mt5.account_equity
            profit = equity - balance
            self.balance_label.configure(text=f"${balance:,.2f}")
            self.equity_label.configure(text=f"${equity:,.2f}")

            if profit >= 0:
                self.pl_label.configure(text=f"+${profit:.2f}", text_color=colors["green"])
            else:
                self.pl_label.configure(text=f"-${abs(profit):.2f}", text_color=colors["red"])
        except:
            pass

        # Update session
        try:
            session_info = self.session.get_status_report()
            if session_info:
                self.session_label.configure(text=session_info.get('current_session', 'Unknown'))
                can_trade, reason, _ = self.session.can_trade()
                if can_trade:
                    self.can_trade_label.configure(text="Can Trade: YES", text_color=colors["green"])
                else:
                    self.can_trade_label.configure(text="Can Trade: NO", text_color=colors["red"])
        except:
            pass

        # Get market data and update signals
        try:
            df = self.mt5.get_market_data(self.config.symbol, self.config.execution_timeframe, 500)
            if df is not None and len(df) > 100:
                self.update_signals(df)
        except:
            pass

        # Update positions
        try:
            positions = self.mt5.get_open_positions(self.config.symbol)
            self.update_positions(positions)
        except:
            pass

        # Update risk state
        self.update_risk_state()

        # Update status
        self.last_update = datetime.now()
        self.update_label.configure(text=f"Last update: {self.last_update.strftime('%H:%M:%S')}")
        self.status_label.configure(text="Running...")

    def update_signals(self, df):
        """Update SMC, ML, and Regime signals"""
        colors = THEMES[self.current_theme]

        # Build complete features
        df = self.feature_eng.calculate_all(df, include_ml_features=True)
        df = self.smc.calculate_all(df)

        # Regime
        try:
            df = self.hmm.predict(df)
            regime = self.hmm.get_current_state(df)
            if regime:
                regime_name = regime.regime.value.replace('_', ' ').title()
                self.regime_label.configure(text=regime_name)
                self.volatility_label.configure(text=f"{regime.volatility:.2f}")
                self.regime_conf_label.configure(text=f"{regime.confidence:.0%}")

                if "HIGH" in regime.regime.value:
                    self.regime_label.configure(text_color=colors["red"])
                elif "LOW" in regime.regime.value:
                    self.regime_label.configure(text_color=colors["green"])
                else:
                    self.regime_label.configure(text_color=colors["orange"])
        except:
            pass

        # SMC Signal
        try:
            smc_signal = self.smc.generate_signal(df)
            if smc_signal:
                self.smc_signal_label.configure(text=smc_signal.signal_type)
                self.smc_confidence_bar.set(smc_signal.confidence)
                self.smc_conf_label.configure(text=f"Confidence: {smc_signal.confidence:.0%}")
                self.smc_reason_label.configure(text=smc_signal.reason[:60] + "..." if len(smc_signal.reason) > 60 else smc_signal.reason)

                if smc_signal.signal_type == "BUY":
                    self.smc_signal_label.configure(text_color=colors["green"])
                    self.smc_confidence_bar.configure(progress_color=colors["green"])
                else:
                    self.smc_signal_label.configure(text_color=colors["red"])
                    self.smc_confidence_bar.configure(progress_color=colors["red"])
            else:
                self.smc_signal_label.configure(text="NO SIGNAL", text_color=colors["text_muted"])
                self.smc_confidence_bar.set(0)
                self.smc_conf_label.configure(text="Confidence: --%")
                self.smc_reason_label.configure(text="Waiting for setup...")
        except:
            pass

        # ML Prediction
        try:
            if self.ml.fitted:
                available_features = [f for f in self.ml.feature_names if f in df.columns]
                ml_pred = self.ml.predict(df, available_features)
                if ml_pred:
                    self.ml_signal_label.configure(text=ml_pred.signal)
                    self.ml_confidence_bar.set(ml_pred.confidence)
                    self.ml_conf_label.configure(text=f"Confidence: {ml_pred.confidence:.0%}")

                    buy_prob = ml_pred.probability
                    sell_prob = 1.0 - buy_prob
                    self.ml_prob_label.configure(text=f"Buy: {buy_prob:.0%} | Sell: {sell_prob:.0%}")

                    if ml_pred.signal == "BUY":
                        self.ml_signal_label.configure(text_color=colors["green"])
                        self.ml_confidence_bar.configure(progress_color=colors["green"])
                    elif ml_pred.signal == "SELL":
                        self.ml_signal_label.configure(text_color=colors["red"])
                        self.ml_confidence_bar.configure(progress_color=colors["red"])
                    else:
                        self.ml_signal_label.configure(text_color=colors["orange"])
                        self.ml_confidence_bar.configure(progress_color=colors["orange"])
        except:
            pass

    def update_positions(self, positions):
        """Update positions display"""
        self.positions_text.configure(state="normal")
        self.positions_text.delete("1.0", "end")

        if positions is None or positions.is_empty():
            self.positions_text.insert("1.0", "No open positions")
        else:
            for row in positions.iter_rows(named=True):
                ticket = row.get('ticket', 'N/A')
                pos_type = "BUY" if row.get('type', 0) == 0 else "SELL"
                volume = row.get('volume', 0)
                profit = row.get('profit', 0)
                price_open = row.get('price_open', 0)

                line = f"#{ticket} | {pos_type} {volume} @ {price_open:.2f} | P/L: ${profit:+.2f}\n"
                self.positions_text.insert("end", line)

        self.positions_text.configure(state="disabled")

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
                            self.daily_loss_label.configure(text=f"${float(value):,.2f}")
                        elif key == 'daily_profit':
                            self.daily_profit_label.configure(text=f"${float(value):,.2f}")
                        elif key == 'consecutive_losses':
                            self.consec_loss_label.configure(text=value)
        except:
            pass

    def on_closing(self):
        """Handle window close"""
        self.running = False
        if self.mt5:
            self.mt5.disconnect()
        self.destroy()


def main():
    app = TradingDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
