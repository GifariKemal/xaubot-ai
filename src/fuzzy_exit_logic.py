"""
Fuzzy Logic Controller for Exit Confidence Aggregation
=======================================================
Combines multiple weak exit signals into a single confidence score (0.0-1.0).

Current problem: 8 isolated exit checks return True/False, missing weak correlations.
Fuzzy solution: Aggregate velocity, acceleration, profit_retention, RSI, time, etc.
into probabilistic exit decision.

Input variables (6):
- velocity: $/second (-0.5 to +0.5)
- acceleration: $/s² (-0.01 to +0.01)
- profit_retention: current_profit / peak_profit (0.0-1.2)
- rsi: RSI indicator (0-100)
- time_in_trade: Minutes since entry (0-60+)
- profit_level: profit / tp_target (0.0-2.0)

Output:
- exit_confidence: 0.0-1.0 (exit if > 0.70, warning if > 0.50)

Rule base: 30+ fuzzy rules derived from v6 exit logic.

Author: AI Assistant (Phase 3 - Advanced Exit Strategies)
"""

import numpy as np
from typing import Optional

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    _SKFUZZY_AVAILABLE = True
except ImportError:
    _SKFUZZY_AVAILABLE = False


class FuzzyExitController:
    """
    Fuzzy logic system for exit confidence calculation.

    Aggregates 6 input variables into exit confidence score.
    """

    def __init__(self):
        """Initialize fuzzy control system with rules."""
        if not _SKFUZZY_AVAILABLE:
            raise ImportError(
                "scikit-fuzzy not installed. Install with: pip install scikit-fuzzy"
            )

        # === INPUT VARIABLES ===
        self.velocity = ctrl.Antecedent(np.linspace(-0.5, 0.5, 101), 'velocity')
        self.acceleration = ctrl.Antecedent(np.linspace(-0.01, 0.01, 101), 'accel')
        self.profit_retention = ctrl.Antecedent(np.linspace(0, 1.2, 121), 'retention')
        self.rsi = ctrl.Antecedent(np.linspace(0, 100, 101), 'rsi')
        self.time_in_trade = ctrl.Antecedent(np.linspace(0, 60, 61), 'time')
        self.profit_level = ctrl.Antecedent(np.linspace(0, 2.0, 101), 'profit_lvl')

        # === OUTPUT VARIABLE ===
        self.exit_confidence = ctrl.Consequent(np.linspace(0, 1, 101), 'exit_conf')

        # === MEMBERSHIP FUNCTIONS ===
        self._define_membership_functions()

        # === FUZZY RULES ===
        self.rules = self._create_rule_base()

        # Create control system
        self.exit_ctrl = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.exit_ctrl)

    def _define_membership_functions(self):
        """Define membership functions for all variables."""

        # VELOCITY ($/second)
        self.velocity['crashing'] = fuzz.trapmf(self.velocity.universe, [-0.5, -0.5, -0.15, -0.08])
        self.velocity['declining'] = fuzz.trimf(self.velocity.universe, [-0.15, -0.05, 0])
        self.velocity['stalling'] = fuzz.trimf(self.velocity.universe, [-0.03, 0, 0.03])
        self.velocity['growing'] = fuzz.trimf(self.velocity.universe, [0, 0.05, 0.15])
        self.velocity['accelerating'] = fuzz.trapmf(self.velocity.universe, [0.08, 0.15, 0.5, 0.5])

        # ACCELERATION ($/s²)
        self.acceleration['strong_negative'] = fuzz.trapmf(self.acceleration.universe, [-0.01, -0.01, -0.005, -0.002])
        self.acceleration['negative'] = fuzz.trimf(self.acceleration.universe, [-0.005, -0.001, 0])
        self.acceleration['neutral'] = fuzz.trimf(self.acceleration.universe, [-0.001, 0, 0.001])
        self.acceleration['positive'] = fuzz.trimf(self.acceleration.universe, [0, 0.001, 0.005])
        self.acceleration['strong_positive'] = fuzz.trapmf(self.acceleration.universe, [0.002, 0.005, 0.01, 0.01])

        # PROFIT RETENTION (current / peak)
        self.profit_retention['collapsed'] = fuzz.trapmf(self.profit_retention.universe, [0, 0, 0.3, 0.5])
        self.profit_retention['low'] = fuzz.trimf(self.profit_retention.universe, [0.3, 0.5, 0.7])
        self.profit_retention['medium'] = fuzz.trimf(self.profit_retention.universe, [0.6, 0.8, 0.95])
        self.profit_retention['high'] = fuzz.trimf(self.profit_retention.universe, [0.9, 1.0, 1.1])
        self.profit_retention['peak'] = fuzz.trapmf(self.profit_retention.universe, [1.05, 1.15, 1.2, 1.2])

        # RSI (0-100)
        self.rsi['oversold'] = fuzz.trapmf(self.rsi.universe, [0, 0, 20, 30])
        self.rsi['low'] = fuzz.trimf(self.rsi.universe, [20, 35, 45])
        self.rsi['neutral'] = fuzz.trimf(self.rsi.universe, [40, 50, 60])
        self.rsi['high'] = fuzz.trimf(self.rsi.universe, [55, 65, 80])
        self.rsi['overbought'] = fuzz.trapmf(self.rsi.universe, [70, 80, 100, 100])

        # TIME IN TRADE (minutes)
        self.time_in_trade['very_short'] = fuzz.trapmf(self.time_in_trade.universe, [0, 0, 3, 5])
        self.time_in_trade['short'] = fuzz.trimf(self.time_in_trade.universe, [3, 7, 12])
        self.time_in_trade['medium'] = fuzz.trimf(self.time_in_trade.universe, [10, 15, 25])
        self.time_in_trade['long'] = fuzz.trimf(self.time_in_trade.universe, [20, 35, 50])
        self.time_in_trade['very_long'] = fuzz.trapmf(self.time_in_trade.universe, [45, 55, 60, 60])

        # PROFIT LEVEL (profit / tp_target)
        self.profit_level['none'] = fuzz.trapmf(self.profit_level.universe, [0, 0, 0.1, 0.2])
        self.profit_level['small'] = fuzz.trimf(self.profit_level.universe, [0.1, 0.3, 0.5])
        self.profit_level['medium'] = fuzz.trimf(self.profit_level.universe, [0.4, 0.6, 0.8])
        self.profit_level['high'] = fuzz.trimf(self.profit_level.universe, [0.7, 0.9, 1.1])
        self.profit_level['exceeded'] = fuzz.trapmf(self.profit_level.universe, [1.0, 1.2, 2.0, 2.0])

        # EXIT CONFIDENCE (0-1)
        self.exit_confidence['very_low'] = fuzz.trimf(self.exit_confidence.universe, [0, 0, 0.25])
        self.exit_confidence['low'] = fuzz.trimf(self.exit_confidence.universe, [0.1, 0.3, 0.5])
        self.exit_confidence['medium'] = fuzz.trimf(self.exit_confidence.universe, [0.4, 0.6, 0.75])
        self.exit_confidence['high'] = fuzz.trimf(self.exit_confidence.universe, [0.65, 0.8, 0.95])
        self.exit_confidence['very_high'] = fuzz.trapmf(self.exit_confidence.universe, [0.85, 0.95, 1.0, 1.0])

    def _create_rule_base(self):
        """Create 30+ fuzzy rules for exit decisions."""
        rules = []

        # === VELOCITY-BASED RULES (highest priority) ===
        # Rule 1: Crashing velocity = immediate exit
        rules.append(ctrl.Rule(
            self.velocity['crashing'],
            self.exit_confidence['very_high']
        ))

        # Rule 2: Declining velocity + negative acceleration = high exit
        rules.append(ctrl.Rule(
            self.velocity['declining'] & self.acceleration['negative'],
            self.exit_confidence['high']
        ))

        # Rule 3: Declining velocity + collapsed retention = very high exit
        rules.append(ctrl.Rule(
            self.velocity['declining'] & self.profit_retention['collapsed'],
            self.exit_confidence['very_high']
        ))

        # Rule 4: Stalling velocity + low retention = medium exit
        rules.append(ctrl.Rule(
            self.velocity['stalling'] & self.profit_retention['low'],
            self.exit_confidence['medium']
        ))

        # Rule 5: Stalling velocity + long time = high exit
        rules.append(ctrl.Rule(
            self.velocity['stalling'] & self.time_in_trade['long'],
            self.exit_confidence['high']
        ))

        # === ACCELERATION-BASED RULES ===
        # Rule 6: Strong negative accel + medium profit = high exit
        rules.append(ctrl.Rule(
            self.acceleration['strong_negative'] & self.profit_level['medium'],
            self.exit_confidence['high']
        ))

        # Rule 7: Negative accel + declining velocity = high exit
        rules.append(ctrl.Rule(
            self.acceleration['negative'] & self.velocity['declining'],
            self.exit_confidence['high']
        ))

        # === PROFIT RETENTION RULES ===
        # Rule 8: Collapsed retention (regardless of velocity) = very high exit
        rules.append(ctrl.Rule(
            self.profit_retention['collapsed'],
            self.exit_confidence['very_high']
        ))

        # Rule 9: Low retention + stalling velocity = high exit
        rules.append(ctrl.Rule(
            self.profit_retention['low'] & self.velocity['stalling'],
            self.exit_confidence['high']
        ))

        # Rule 10: Low retention + medium time = medium exit
        rules.append(ctrl.Rule(
            self.profit_retention['low'] & self.time_in_trade['medium'],
            self.exit_confidence['medium']
        ))

        # === RSI REVERSAL RULES (position-dependent) ===
        # Rule 11: Oversold RSI + high profit (SELL position exiting at support) = high exit
        rules.append(ctrl.Rule(
            self.rsi['oversold'] & self.profit_retention['high'],
            self.exit_confidence['high']
        ))

        # Rule 12: Overbought RSI + high profit (BUY position exiting at resistance) = high exit
        rules.append(ctrl.Rule(
            self.rsi['overbought'] & self.profit_retention['high'],
            self.exit_confidence['high']
        ))

        # Rule 13: Oversold RSI + low retention (SELL position, price bouncing) = medium exit
        rules.append(ctrl.Rule(
            self.rsi['oversold'] & self.profit_retention['low'],
            self.exit_confidence['medium']
        ))

        # === TIME-BASED RULES ===
        # Rule 14: Very long time + stalling velocity = high exit (trade exhausted)
        rules.append(ctrl.Rule(
            self.time_in_trade['very_long'] & self.velocity['stalling'],
            self.exit_confidence['high']
        ))

        # Rule 15: Long time + low retention = high exit
        rules.append(ctrl.Rule(
            self.time_in_trade['long'] & self.profit_retention['low'],
            self.exit_confidence['high']
        ))

        # Rule 16: Medium time + collapsed retention = very high exit
        rules.append(ctrl.Rule(
            self.time_in_trade['medium'] & self.profit_retention['collapsed'],
            self.exit_confidence['very_high']
        ))

        # === PROFIT LEVEL RULES ===
        # Rule 17: Exceeded profit + declining velocity = high exit (take profit)
        rules.append(ctrl.Rule(
            self.profit_level['exceeded'] & self.velocity['declining'],
            self.exit_confidence['high']
        ))

        # Rule 18: High profit + stalling velocity = medium exit
        rules.append(ctrl.Rule(
            self.profit_level['high'] & self.velocity['stalling'],
            self.exit_confidence['medium']
        ))

        # Rule 19: High profit + strong negative accel = high exit
        rules.append(ctrl.Rule(
            self.profit_level['high'] & self.acceleration['strong_negative'],
            self.exit_confidence['high']
        ))

        # === POSITIVE SCENARIOS (low exit confidence) ===
        # Rule 20: Growing velocity + high retention = very low exit (hold)
        rules.append(ctrl.Rule(
            self.velocity['growing'] & self.profit_retention['high'],
            self.exit_confidence['very_low']
        ))

        # Rule 21: Accelerating velocity + positive accel = very low exit (strong trend)
        rules.append(ctrl.Rule(
            self.velocity['accelerating'] & self.acceleration['positive'],
            self.exit_confidence['very_low']
        ))

        # Rule 22: Peak retention + growing velocity = very low exit (at new high)
        rules.append(ctrl.Rule(
            self.profit_retention['peak'] & self.velocity['growing'],
            self.exit_confidence['very_low']
        ))

        # === COMBINATION RULES (weak signals together) ===
        # Rule 23: Stalling + neutral accel + medium retention + long time = medium exit
        rules.append(ctrl.Rule(
            self.velocity['stalling'] & self.acceleration['neutral'] &
            self.profit_retention['medium'] & self.time_in_trade['long'],
            self.exit_confidence['medium']
        ))

        # Rule 24: Declining + negative accel + low retention = very high exit (triple threat)
        rules.append(ctrl.Rule(
            self.velocity['declining'] & self.acceleration['negative'] &
            self.profit_retention['low'],
            self.exit_confidence['very_high']
        ))

        # Rule 25: Small profit + very long time + stalling = high exit (cut losses)
        rules.append(ctrl.Rule(
            self.profit_level['small'] & self.time_in_trade['very_long'] &
            self.velocity['stalling'],
            self.exit_confidence['high']
        ))

        # === EARLY EXIT RULES (prevent holding too long) ===
        # Rule 26: Medium profit + declining + long time = high exit
        rules.append(ctrl.Rule(
            self.profit_level['medium'] & self.velocity['declining'] &
            self.time_in_trade['long'],
            self.exit_confidence['high']
        ))

        # Rule 27: High profit + low retention + declining = very high exit (protect gains)
        rules.append(ctrl.Rule(
            self.profit_level['high'] & self.profit_retention['low'] &
            self.velocity['declining'],
            self.exit_confidence['very_high']
        ))

        # === DEFENSIVE RULES (prevent premature exit) ===
        # Rule 28: Short time + growing velocity = very low exit (give time to develop)
        rules.append(ctrl.Rule(
            self.time_in_trade['short'] & self.velocity['growing'],
            self.exit_confidence['very_low']
        ))

        # Rule 29: Very short time + high retention = very low exit (just started)
        rules.append(ctrl.Rule(
            self.time_in_trade['very_short'] & self.profit_retention['high'],
            self.exit_confidence['very_low']
        ))

        # Rule 30: Medium profit + accelerating velocity = low exit (let it run)
        rules.append(ctrl.Rule(
            self.profit_level['medium'] & self.velocity['accelerating'],
            self.exit_confidence['low']
        ))

        return rules

    def evaluate(
        self,
        velocity: float,
        acceleration: float,
        profit_retention: float,
        rsi: float,
        time_in_trade: float,
        profit_level: float,
    ) -> float:
        """
        Evaluate exit confidence for current trade state.

        Args:
            velocity: Profit velocity ($/second)
            acceleration: Profit acceleration ($/s²)
            profit_retention: current_profit / peak_profit
            rsi: RSI indicator (0-100)
            time_in_trade: Minutes since entry
            profit_level: profit / tp_target

        Returns:
            Exit confidence (0.0-1.0)
            > 0.75: High confidence, exit now
            0.50-0.75: Medium confidence, warning
            < 0.50: Low confidence, hold
        """
        # Clamp inputs to universe ranges
        velocity = np.clip(velocity, -0.5, 0.5)
        acceleration = np.clip(acceleration, -0.01, 0.01)
        profit_retention = np.clip(profit_retention, 0, 1.2)
        rsi = np.clip(rsi, 0, 100)
        time_in_trade = np.clip(time_in_trade, 0, 60)
        profit_level = np.clip(profit_level, 0, 2.0)

        # Set inputs
        self.simulation.input['velocity'] = velocity
        self.simulation.input['accel'] = acceleration
        self.simulation.input['retention'] = profit_retention
        self.simulation.input['rsi'] = rsi
        self.simulation.input['time'] = time_in_trade
        self.simulation.input['profit_lvl'] = profit_level

        # Compute output
        try:
            self.simulation.compute()
            return float(self.simulation.output['exit_conf'])
        except Exception as e:
            # Fallback: if fuzzy system fails, return conservative confidence
            # (likely due to no rules firing)
            return 0.3

    def visualize(self, variable_name: str):
        """
        Visualize membership functions for a variable.

        Args:
            variable_name: 'velocity', 'accel', 'retention', 'rsi', 'time', 'profit_lvl', 'exit_conf'
        """
        import matplotlib.pyplot as plt

        var_map = {
            'velocity': self.velocity,
            'accel': self.acceleration,
            'retention': self.profit_retention,
            'rsi': self.rsi,
            'time': self.time_in_trade,
            'profit_lvl': self.profit_level,
            'exit_conf': self.exit_confidence,
        }

        if variable_name not in var_map:
            raise ValueError(f"Unknown variable: {variable_name}")

        var = var_map[variable_name]
        var.view()
        plt.show()
