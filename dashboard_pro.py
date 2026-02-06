"""
Trading Bot Dashboard PRO - MONITORING ONLY
============================================
Pure monitoring dashboard (no control) with:
- Real-time Price & Equity charts
- Visual alarms for critical conditions
- Data freshness/heartbeat indicator
- Detailed AI reasoning in logs
- Stale data visual warning
"""

import customtkinter as ctk
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import deque
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Matplotlib for charts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('TkAgg')

# Import bot components
try:
    from src.mt5_connector import MT5Connector
    from src.smc_polars import SMCAnalyzer
    from src.ml_model import TradingModel
    from src.regime_detector import MarketRegimeDetector
    from src.session_filter import SessionFilter
    from src.feature_eng import FeatureEngineer
    from src.config import TradingConfig
    from loguru import logger
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
        "yellow": "#ffd93d",
        "highlight": "#2d4a2d",
        "highlight_off": "#2d2d44",
        "chart_bg": "#0d0d1a",
        "chart_line": "#00d4ff",
        "chart_equity": "#00ff88",
        "danger": "#ff2222",
        "stale": "#444444",  # Gray for stale data
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
        "yellow": "#cc9900",
        "highlight": "#d4edda",
        "highlight_off": "#e9ecef",
        "chart_bg": "#f8f9fa",
        "chart_line": "#0066cc",
        "chart_equity": "#00aa55",
        "danger": "#cc0000",
        "stale": "#cccccc",
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

    def set_alarm(self, is_alarm=False):
        """Set alarm state (red background)"""
        colors = THEMES[self.theme]
        if is_alarm:
            self.configure(fg_color=colors["danger"])
        else:
            self.configure(fg_color=colors["card"])


class TradingDashboardPro(ctk.CTk):
    """Production Monitoring Dashboard - NO CONTROL, PURE MONITORING"""

    def __init__(self):
        super().__init__()

        # Theme state
        self.current_theme = "dark"
        ctk.set_appearance_mode("dark")

        # Window setup
        self.title("AI Trading Bot - Monitor")
        self.geometry("750x900")
        self.minsize(650, 750)
        self.configure(fg_color=THEMES[self.current_theme]["bg"])

        # Store all themed widgets
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
        self.last_data_time = None  # For stale data detection
        self.data_stale = False
        self.consecutive_errors = 0

        # Data history for charts
        self.price_history = deque(maxlen=120)  # 2 hours of data
        self.equity_history = deque(maxlen=120)
        self.balance_history = deque(maxlen=120)
        self.time_history = deque(maxlen=120)

        # Risk alarm state
        self.risk_alarm_active = False

        # Create UI
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()

        # Start connection
        self.after(100, self.connect_systems)

        # Heartbeat checker
        self.after(5000, self.check_data_freshness)

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
            self.theme_btn.configure(text="Light")
        else:
            self.theme_btn.configure(text="Dark")

        # Update all cards
        for card in self.cards:
            card.update_theme(self.current_theme)

        # Update header
        self.title_label.configure(text_color=colors["text"])
        self.subtitle_label.configure(text_color=colors["accent"])
        self.time_label.configure(text_color=colors["text_secondary"])

        # Update charts
        self.update_price_chart()
        self.update_equity_chart()

        # Update text boxes
        if hasattr(self, 'log_text'):
            self.log_text.configure(fg_color=colors["card_inner"], text_color=colors["green"])

    def check_data_freshness(self):
        """Check if data is stale (>5 seconds old) - HEARTBEAT"""
        colors = THEMES[self.current_theme]

        if self.last_data_time:
            age = (datetime.now() - self.last_data_time).total_seconds()

            if age > 5:
                # DATA IS STALE - Visual warning
                self.data_stale = True
                self.heartbeat_label.configure(
                    text=f"DATA STALE ({age:.0f}s)",
                    text_color=colors["red"]
                )
                # Gray out the main window slightly
                self.status_frame.configure(fg_color=colors["stale"])
                self.status_label.configure(text="WARNING: Data tidak terupdate!", text_color=colors["red"])
            else:
                self.data_stale = False
                self.heartbeat_label.configure(
                    text=f"LIVE ({age:.1f}s)",
                    text_color=colors["green"]
                )
                if self.consecutive_errors == 0:
                    self.status_frame.configure(fg_color=colors["card"])

        # Schedule next check
        self.after(1000, self.check_data_freshness)

    def create_header(self):
        """Create header section with heartbeat indicator"""
        colors = THEMES[self.current_theme]

        header_frame = ctk.CTkFrame(self, fg_color="transparent", height=45)
        header_frame.pack(fill="x", padx=15, pady=(8, 3))
        header_frame.pack_propagate(False)

        # Logo/Title
        self.title_label = ctk.CTkLabel(
            header_frame,
            text="AI TRADING BOT",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=colors["text"]
        )
        self.title_label.pack(side="left")

        self.subtitle_label = ctk.CTkLabel(
            header_frame,
            text="  MONITOR",
            font=ctk.CTkFont(size=11),
            text_color=colors["accent"]
        )
        self.subtitle_label.pack(side="left", pady=(3, 0))

        # Theme toggle
        self.theme_btn = ctk.CTkButton(
            header_frame,
            text="Light",
            width=55,
            height=24,
            corner_radius=12,
            font=ctk.CTkFont(size=10),
            command=self.toggle_theme
        )
        self.theme_btn.pack(side="right", padx=3)

        # Heartbeat indicator (DATA FRESHNESS)
        self.heartbeat_label = ctk.CTkLabel(
            header_frame,
            text="CONNECTING...",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=colors["orange"]
        )
        self.heartbeat_label.pack(side="right", padx=10)

        # Connection status
        self.connection_label = ctk.CTkLabel(
            header_frame,
            text="Connecting...",
            font=ctk.CTkFont(size=10),
            text_color=colors["orange"]
        )
        self.connection_label.pack(side="right", padx=8)

        # Time display
        self.time_label = ctk.CTkLabel(
            header_frame,
            text="--:--:-- WIB",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=colors["text_secondary"]
        )
        self.time_label.pack(side="right", padx=8)

    def create_main_layout(self):
        """Create main dashboard layout"""
        # Scrollable main container
        self.main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.main_scroll.pack(fill="both", expand=True, padx=10, pady=3)

        # Configure 2-column grid
        self.main_scroll.grid_columnconfigure(0, weight=1, minsize=300)
        self.main_scroll.grid_columnconfigure(1, weight=1, minsize=300)

        # Row 0: Price Chart (full width)
        self.create_price_chart_card(self.main_scroll, 0, 0)

        # Row 1: Price & Account
        self.create_price_card(self.main_scroll, 1, 0)
        self.create_account_card(self.main_scroll, 1, 1)

        # Row 2: Session & Risk (with ALARM capability)
        self.create_session_card(self.main_scroll, 2, 0)
        self.create_risk_card(self.main_scroll, 2, 1)

        # Row 3: SMC & ML
        self.create_smc_card(self.main_scroll, 3, 0)
        self.create_ml_card(self.main_scroll, 3, 1)

        # Row 4: Regime & Positions
        self.create_regime_card(self.main_scroll, 4, 0)
        self.create_positions_card(self.main_scroll, 4, 1)

        # Row 5: Equity Chart (replaced Settings) - full width
        self.create_equity_chart_card(self.main_scroll, 5, 0)

        # Row 6: Log (full width, larger)
        self.create_log_card(self.main_scroll, 6, 0)

    def create_price_chart_card(self, parent, row, col):
        """Mini price chart - sparkline style"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "PRICE CHART (2H)", theme=self.current_theme)
        card.grid(row=row, column=col, columnspan=2, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Create matplotlib figure
        self.price_fig = Figure(figsize=(6, 1.3), dpi=100, facecolor=colors["chart_bg"])
        self.price_ax = self.price_fig.add_subplot(111)
        self.price_ax.set_facecolor(colors["chart_bg"])

        # Style
        self.price_ax.tick_params(colors=colors["text_muted"], labelsize=7)
        for spine in ['top', 'right']:
            self.price_ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.price_ax.spines[spine].set_color(colors["text_muted"])

        self.price_fig.tight_layout(pad=0.3)

        # Embed in tkinter
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, master=card.content)
        self.price_canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_equity_chart_card(self, parent, row, col):
        """Equity/Balance chart - REPLACED SETTINGS"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "EQUITY vs BALANCE (2H)", theme=self.current_theme)
        card.grid(row=row, column=col, columnspan=2, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Create matplotlib figure
        self.equity_fig = Figure(figsize=(6, 1.3), dpi=100, facecolor=colors["chart_bg"])
        self.equity_ax = self.equity_fig.add_subplot(111)
        self.equity_ax.set_facecolor(colors["chart_bg"])

        # Style
        self.equity_ax.tick_params(colors=colors["text_muted"], labelsize=7)
        for spine in ['top', 'right']:
            self.equity_ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.equity_ax.spines[spine].set_color(colors["text_muted"])

        self.equity_fig.tight_layout(pad=0.3)

        # Embed
        self.equity_canvas = FigureCanvasTkAgg(self.equity_fig, master=card.content)
        self.equity_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_price_chart(self):
        """Redraw the price chart"""
        if len(self.price_history) < 2:
            return

        colors = THEMES[self.current_theme]
        self.price_ax.clear()

        prices = list(self.price_history)
        self.price_ax.plot(prices, color=colors["chart_line"], linewidth=1.5)
        self.price_ax.fill_between(range(len(prices)), prices, alpha=0.15, color=colors["chart_line"])

        # Style
        self.price_ax.set_facecolor(colors["chart_bg"])
        self.price_ax.tick_params(colors=colors["text_muted"], labelsize=7)
        for spine in ['top', 'right']:
            self.price_ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.price_ax.spines[spine].set_color(colors["text_muted"])

        self.price_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
        self.price_fig.tight_layout(pad=0.3)
        self.price_canvas.draw()

    def update_equity_chart(self):
        """Redraw equity/balance chart"""
        if len(self.equity_history) < 2:
            return

        colors = THEMES[self.current_theme]
        self.equity_ax.clear()

        equity = list(self.equity_history)
        balance = list(self.balance_history)

        # Plot both lines
        self.equity_ax.plot(equity, color=colors["chart_equity"], linewidth=1.5, label='Equity')
        self.equity_ax.plot(balance, color=colors["text_muted"], linewidth=1, linestyle='--', label='Balance')

        # Fill between (profit/loss visual)
        self.equity_ax.fill_between(
            range(len(equity)), balance, equity,
            where=[e >= b for e, b in zip(equity, balance)],
            alpha=0.2, color=colors["green"]
        )
        self.equity_ax.fill_between(
            range(len(equity)), balance, equity,
            where=[e < b for e, b in zip(equity, balance)],
            alpha=0.2, color=colors["red"]
        )

        # Style
        self.equity_ax.set_facecolor(colors["chart_bg"])
        self.equity_ax.tick_params(colors=colors["text_muted"], labelsize=7)
        for spine in ['top', 'right']:
            self.equity_ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.equity_ax.spines[spine].set_color(colors["text_muted"])

        self.equity_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        self.equity_fig.tight_layout(pad=0.3)
        self.equity_canvas.draw()

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
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=colors["green"]
        )
        self.price_label.pack(pady=(3, 0))

        # Change indicator
        self.price_change_label = ctk.CTkLabel(
            card.content,
            text="-- (--)",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_muted"]
        )
        self.price_change_label.pack()

        # Spread
        self.spread_label = ctk.CTkLabel(
            card.content,
            text="Spread: -- pips",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.spread_label.pack(pady=(2, 0))

    def create_account_card(self, parent, row, col):
        """Account information card"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "ACCOUNT", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Balance
        balance_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        balance_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(balance_frame, text="Balance:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.balance_label = ctk.CTkLabel(balance_frame, text="$-----.--", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["text"])
        self.balance_label.pack(side="right")

        # Equity
        equity_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        equity_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(equity_frame, text="Equity:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.equity_label = ctk.CTkLabel(equity_frame, text="$-----.--", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["text"])
        self.equity_label.pack(side="right")

        # P/L
        pl_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        pl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(pl_frame, text="P/L:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.pl_label = ctk.CTkLabel(pl_frame, text="$0.00", font=ctk.CTkFont(size=11, weight="bold"), text_color=colors["green"])
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
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=colors["orange"]
        )
        self.session_label.pack(pady=(2, 0))

        # Golden time
        self.golden_frame = ctk.CTkFrame(card.content, fg_color=colors["highlight_off"], corner_radius=6)
        self.golden_frame.pack(pady=5, padx=3, fill="x")
        self.golden_label = ctk.CTkLabel(
            self.golden_frame,
            text="GOLDEN: --",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=colors["text_secondary"]
        )
        self.golden_label.pack(pady=3)

        # Can trade
        self.can_trade_label = ctk.CTkLabel(
            card.content,
            text="Can Trade: --",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.can_trade_label.pack()

    def create_risk_card(self, parent, row, col):
        """Risk status card - WITH ALARM CAPABILITY"""
        colors = THEMES[self.current_theme]
        self.risk_card = ModernCard(parent, "RISK STATUS", theme=self.current_theme)
        self.risk_card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(self.risk_card)

        # Daily loss
        dl_frame = ctk.CTkFrame(self.risk_card.content, fg_color="transparent")
        dl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(dl_frame, text="Daily Loss:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.daily_loss_label = ctk.CTkLabel(dl_frame, text="$0.00", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["red"])
        self.daily_loss_label.pack(side="right")

        # Daily profit
        dp_frame = ctk.CTkFrame(self.risk_card.content, fg_color="transparent")
        dp_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(dp_frame, text="Daily Profit:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.daily_profit_label = ctk.CTkLabel(dp_frame, text="$0.00", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["green"])
        self.daily_profit_label.pack(side="right")

        # Consecutive losses
        cl_frame = ctk.CTkFrame(self.risk_card.content, fg_color="transparent")
        cl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(cl_frame, text="Consec. Losses:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.consec_loss_label = ctk.CTkLabel(cl_frame, text="0", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["text"])
        self.consec_loss_label.pack(side="right")

        # Risk % indicator
        risk_pct_frame = ctk.CTkFrame(self.risk_card.content, fg_color="transparent")
        risk_pct_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(risk_pct_frame, text="Risk Used:", font=ctk.CTkFont(size=10), text_color=colors["text_secondary"]).pack(side="left")
        self.risk_pct_label = ctk.CTkLabel(risk_pct_frame, text="0%", font=ctk.CTkFont(size=10, weight="bold"), text_color=colors["green"])
        self.risk_pct_label.pack(side="right")

    def create_smc_card(self, parent, row, col):
        """SMC Signal card with reasoning"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "SMC SIGNAL", theme=self.current_theme)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        # Signal
        self.smc_signal_label = ctk.CTkLabel(
            card.content,
            text="---",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=colors["text_muted"]
        )
        self.smc_signal_label.pack(pady=(3, 2))

        # Confidence bar
        self.smc_confidence_bar = ctk.CTkProgressBar(card.content, height=6, corner_radius=3)
        self.smc_confidence_bar.pack(pady=3, fill="x", padx=10)
        self.smc_confidence_bar.set(0)

        # Reason (AI REASONING)
        self.smc_reason_label = ctk.CTkLabel(
            card.content,
            text="Waiting...",
            font=ctk.CTkFont(size=9),
            text_color=colors["text_muted"],
            wraplength=250
        )
        self.smc_reason_label.pack(pady=(2, 0))

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
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=colors["text_muted"]
        )
        self.ml_signal_label.pack(pady=(3, 2))

        # Confidence bar
        self.ml_confidence_bar = ctk.CTkProgressBar(card.content, height=6, corner_radius=3)
        self.ml_confidence_bar.pack(pady=3, fill="x", padx=10)
        self.ml_confidence_bar.set(0)

        # Probabilities
        self.ml_prob_label = ctk.CTkLabel(
            card.content,
            text="Buy: --% | Sell: --%",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_secondary"]
        )
        self.ml_prob_label.pack()

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
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=colors["text_muted"]
        )
        self.regime_label.pack(pady=(5, 3))

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

        # Positions frame
        self.positions_frame = ctk.CTkScrollableFrame(
            card.content,
            fg_color=colors["card_inner"],
            height=70,
            corner_radius=6
        )
        self.positions_frame.pack(fill="both", expand=True)

        # Placeholder
        self.no_positions_label = ctk.CTkLabel(
            self.positions_frame,
            text="No open positions",
            font=ctk.CTkFont(size=10),
            text_color=colors["text_muted"]
        )
        self.no_positions_label.pack(pady=10)

    def create_log_card(self, parent, row, col):
        """Activity log card - LARGER for better monitoring"""
        colors = THEMES[self.current_theme]
        card = ModernCard(parent, "AI ACTIVITY LOG", theme=self.current_theme)
        card.grid(row=row, column=col, columnspan=2, padx=4, pady=4, sticky="nsew")
        self.cards.append(card)

        self.log_text = ctk.CTkTextbox(
            card.content,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=colors["card_inner"],
            text_color=colors["green"],
            height=120,  # Larger log area
            corner_radius=6
        )
        self.log_text.pack(fill="both", expand=True)

    def create_status_bar(self):
        """Create status bar with data freshness indicator"""
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

    def log(self, message, level="info"):
        """Add message to activity log with better formatting"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            colors = THEMES[self.current_theme]

            # Color based on level
            if level == "error":
                prefix = "[ERROR]"
            elif level == "warn":
                prefix = "[WARN]"
            elif level == "trade":
                prefix = "[TRADE]"
            else:
                prefix = "[INFO]"

            self.log_text.insert("end", f"[{timestamp}] {prefix} {message}\n")
            self.log_text.see("end")

            # Also log to file
            logger.info(f"Dashboard: {message}")
        except Exception as e:
            print(f"Log error: {e}")

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
                self.log("Failed to connect to MT5!", "error")
                self.connection_label.configure(text="Disconnected", text_color=colors["red"])
                return

            self.log("MT5 connected successfully!")
            self.connection_label.configure(text="Connected", text_color=colors["green"])

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
            self.log(f"Connection error: {e}", "error")
            logger.error(f"Dashboard connect error: {e}")
            self.connection_label.configure(text="Error", text_color=colors["red"])

    def update_loop(self):
        """Main update loop with proper error tracking"""
        while self.running:
            try:
                self.after(0, self.update_data)
                self.consecutive_errors = 0
                time.sleep(1)
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Dashboard update error: {e}")
                self.after(0, lambda: self.log(f"Update error: {e}", "error"))
                time.sleep(3)

    def update_data(self):
        """Fetch and update all data"""
        if not self.mt5:
            return

        colors = THEMES[self.current_theme]

        # Mark data as fresh
        self.last_data_time = datetime.now()

        # Update time
        wib = ZoneInfo("Asia/Jakarta")
        now = datetime.now(wib)
        self.time_label.configure(text=now.strftime("%H:%M:%S WIB"))

        # Check golden time
        is_golden = 19 <= now.hour < 23
        if is_golden:
            self.golden_frame.configure(fg_color=colors["highlight"])
            self.golden_label.configure(text="GOLDEN: YES", text_color=colors["green"])
        else:
            self.golden_frame.configure(fg_color=colors["highlight_off"])
            self.golden_label.configure(text="GOLDEN: NO", text_color=colors["text_secondary"])

        # Update price
        try:
            tick = self.mt5.get_tick(self.config.symbol)
            if tick:
                price = (tick.bid + tick.ask) / 2
                spread = (tick.ask - tick.bid) * 100
                self.price_label.configure(text=f"{price:.2f}")
                self.spread_label.configure(text=f"Spread: {spread:.1f} pips")

                # Update price history
                self.price_history.append(price)

                # Calculate change
                if len(self.price_history) > 1:
                    prev_price = list(self.price_history)[-2]
                    change = price - prev_price
                    if change >= 0:
                        self.price_change_label.configure(text=f"+{change:.2f}", text_color=colors["green"])
                    else:
                        self.price_change_label.configure(text=f"{change:.2f}", text_color=colors["red"])

                # Update chart every 10 ticks
                if len(self.price_history) % 10 == 0:
                    self.update_price_chart()

        except Exception as e:
            logger.debug(f"Price update error: {e}")

        # Update account & equity chart
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

            # Update equity/balance history
            self.equity_history.append(equity)
            self.balance_history.append(balance)

            # Update equity chart every 10 updates
            if len(self.equity_history) % 10 == 0:
                self.update_equity_chart()

        except Exception as e:
            logger.debug(f"Account update error: {e}")

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
        except Exception as e:
            logger.debug(f"Session update error: {e}")

        # Get market data and update signals
        try:
            df = self.mt5.get_market_data(self.config.symbol, self.config.execution_timeframe, 500)
            if df is not None and len(df) > 100:
                self.update_signals(df)
        except Exception as e:
            logger.debug(f"Signal update error: {e}")

        # Update positions
        try:
            positions = self.mt5.get_open_positions(self.config.symbol)
            self.update_positions(positions)
        except Exception as e:
            logger.debug(f"Position update error: {e}")

        # Update risk state with ALARM check
        self.update_risk_state()

        # Update status
        self.last_update = datetime.now()
        self.update_label.configure(text=f"Last update: {self.last_update.strftime('%H:%M:%S')}")

        if not self.data_stale:
            self.status_label.configure(text="Monitoring...")

    def update_signals(self, df):
        """Update SMC, ML, and Regime signals with REASONING"""
        colors = THEMES[self.current_theme]

        try:
            df = self.feature_eng.calculate_all(df, include_ml_features=True)
            df = self.smc.calculate_all(df)
        except Exception as e:
            logger.debug(f"Feature calculation error: {e}")
            return

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
        except Exception as e:
            logger.debug(f"Regime update error: {e}")

        # SMC Signal with REASONING
        try:
            smc_signal = self.smc.generate_signal(df)
            if smc_signal:
                self.smc_signal_label.configure(text=smc_signal.signal_type)
                self.smc_confidence_bar.set(smc_signal.confidence)

                # Show AI REASONING
                reason = smc_signal.reason if smc_signal.reason else "Signal generated"
                self.smc_reason_label.configure(text=reason[:80] + "..." if len(reason) > 80 else reason)

                if smc_signal.signal_type == "BUY":
                    self.smc_signal_label.configure(text_color=colors["green"])
                    self.smc_confidence_bar.configure(progress_color=colors["green"])
                else:
                    self.smc_signal_label.configure(text_color=colors["red"])
                    self.smc_confidence_bar.configure(progress_color=colors["red"])
            else:
                self.smc_signal_label.configure(text="NO SIGNAL", text_color=colors["text_muted"])
                self.smc_confidence_bar.set(0)
                self.smc_reason_label.configure(text="Waiting for SMC setup...")
        except Exception as e:
            logger.debug(f"SMC update error: {e}")

        # ML Prediction
        try:
            if self.ml.fitted:
                available_features = [f for f in self.ml.feature_names if f in df.columns]
                ml_pred = self.ml.predict(df, available_features)
                if ml_pred:
                    self.ml_signal_label.configure(text=ml_pred.signal)
                    self.ml_confidence_bar.set(ml_pred.confidence)

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
        except Exception as e:
            logger.debug(f"ML update error: {e}")

    def update_positions(self, positions):
        """Update positions display"""
        colors = THEMES[self.current_theme]

        # Clear existing
        for widget in self.positions_frame.winfo_children():
            widget.destroy()

        if positions is None or positions.is_empty():
            label = ctk.CTkLabel(
                self.positions_frame,
                text="No open positions",
                font=ctk.CTkFont(size=10),
                text_color=colors["text_muted"]
            )
            label.pack(pady=10)
            return

        for row in positions.iter_rows(named=True):
            ticket = row.get('ticket', 'N/A')
            pos_type = "BUY" if row.get('type', 0) == 0 else "SELL"
            volume = row.get('volume', 0)
            profit = row.get('profit', 0)
            price_open = row.get('price_open', 0)

            # Position row
            pos_frame = ctk.CTkFrame(self.positions_frame, fg_color="transparent")
            pos_frame.pack(fill="x", pady=1, padx=3)

            type_color = colors["green"] if pos_type == "BUY" else colors["red"]
            profit_color = colors["green"] if profit >= 0 else colors["red"]

            ctk.CTkLabel(
                pos_frame,
                text=f"{pos_type} {volume} @ {price_open:.2f}",
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color=type_color
            ).pack(side="left")

            ctk.CTkLabel(
                pos_frame,
                text=f"${profit:+.2f}",
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color=profit_color
            ).pack(side="right")

    def update_risk_state(self):
        """Update risk state with ALARM for high risk"""
        colors = THEMES[self.current_theme]

        try:
            risk_file = Path("data/risk_state.txt")
            if risk_file.exists():
                content = risk_file.read_text()
                lines = content.strip().split('\n')

                daily_loss = 0.0
                daily_profit = 0.0
                consec_losses = 0

                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        if key == 'daily_loss':
                            daily_loss = float(value)
                            self.daily_loss_label.configure(text=f"${daily_loss:,.2f}")
                        elif key == 'daily_profit':
                            daily_profit = float(value)
                            self.daily_profit_label.configure(text=f"${daily_profit:,.2f}")
                        elif key == 'consecutive_losses':
                            consec_losses = int(value)
                            self.consec_loss_label.configure(text=str(consec_losses))

                # Calculate risk percentage (based on $250 max daily loss = 5% of $5000)
                max_daily_loss = self.config.capital * (self.config.risk.max_daily_loss / 100)
                risk_pct = (daily_loss / max_daily_loss * 100) if max_daily_loss > 0 else 0

                self.risk_pct_label.configure(text=f"{risk_pct:.0f}%")

                # ALARM: If risk > 80%
                if risk_pct >= 80:
                    self.risk_card.set_alarm(True)
                    self.risk_pct_label.configure(text_color=colors["danger"])
                    if not self.risk_alarm_active:
                        self.log(f"RISK ALARM: Daily loss at {risk_pct:.0f}%!", "warn")
                        self.risk_alarm_active = True
                elif risk_pct >= 50:
                    self.risk_card.set_alarm(False)
                    self.risk_pct_label.configure(text_color=colors["orange"])
                    self.risk_alarm_active = False
                else:
                    self.risk_card.set_alarm(False)
                    self.risk_pct_label.configure(text_color=colors["green"])
                    self.risk_alarm_active = False

        except Exception as e:
            logger.debug(f"Risk state update error: {e}")

    def on_closing(self):
        """Handle window close"""
        self.running = False
        if self.mt5:
            self.mt5.disconnect()
        self.destroy()


def main():
    app = TradingDashboardPro()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
