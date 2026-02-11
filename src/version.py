"""
XAUBot AI - Centralized Version Management
==========================================

Semantic Versioning (SemVer): MAJOR.MINOR.PATCH

MAJOR: Incompatible API changes, breaking changes
MINOR: New features, backward compatible
PATCH: Bug fixes, backward compatible

Author: AI Assistant
"""

import os
from pathlib import Path
from typing import Dict, Tuple
from loguru import logger


class VersionManager:
    """
    Centralized version management for XAUBot AI.

    Auto-detects version based on enabled features and components.
    Reads base version from VERSION file, calculates effective version.
    """

    def __init__(self):
        self.base_version = self._read_version_file()
        self.features = self._detect_features()
        self.effective_version = self._calculate_version()

    def _read_version_file(self) -> Tuple[int, int, int]:
        """Read version from VERSION file."""
        version_file = Path(__file__).parent.parent / "VERSION"
        try:
            with open(version_file, 'r') as f:
                version_str = f.read().strip()
                parts = version_str.split('.')
                if len(parts) != 3:
                    raise ValueError(f"Invalid version format: {version_str}")
                return tuple(int(p) for p in parts)
        except Exception as e:
            logger.warning(f"Could not read VERSION file: {e}, using default 0.0.0")
            return (0, 0, 0)

    def _detect_features(self) -> Dict[str, bool]:
        """
        Auto-detect enabled features from environment and imports.
        """
        features = {}

        # Core features (always enabled)
        features['mt5_integration'] = True
        features['smc_analysis'] = True
        features['ml_prediction'] = True
        features['hmm_regime'] = True

        # Advanced exit features (from environment flags)
        features['kalman_filter'] = os.environ.get("KALMAN_ENABLED", "1") == "1"
        features['advanced_exits'] = os.environ.get("ADVANCED_EXITS_ENABLED", "1") == "1"
        features['predictive_intelligence'] = os.environ.get("PREDICTIVE_ENABLED", "1") == "1"

        # Component detection
        try:
            # Fuzzy Logic
            from src.fuzzy_exit_logic import FuzzyExitController
            features['fuzzy_logic'] = True
        except ImportError:
            features['fuzzy_logic'] = False

        try:
            # Kelly Criterion
            from src.kelly_position_scaler import KellyPositionScaler
            features['kelly_criterion'] = True
        except ImportError:
            features['kelly_criterion'] = False

        try:
            # Trajectory Predictor
            from src.trajectory_predictor import TrajectoryPredictor
            features['trajectory_predictor'] = True
        except ImportError:
            features['trajectory_predictor'] = False

        try:
            # Momentum Persistence
            from src.momentum_persistence import MomentumPersistence
            features['momentum_persistence'] = True
        except ImportError:
            features['momentum_persistence'] = False

        try:
            # Recovery Detector
            from src.recovery_detector import RecoveryDetector
            features['recovery_detector'] = True
        except ImportError:
            features['recovery_detector'] = False

        return features

    def _calculate_version(self) -> Tuple[int, int, int]:
        """
        Calculate effective version based on base + features.

        Version increments:
        - Kalman Filter: +0.1.0 (MINOR)
        - Fuzzy Logic: +0.1.0 (MINOR)
        - Kelly Criterion: +0.1.0 (MINOR)
        - Predictive Intelligence (all 3): +0.3.0 (MINOR)
        - Each predictor separately: +0.1.0 (MINOR)
        """
        major, minor, patch = self.base_version

        # MINOR increments for features
        if self.features.get('kalman_filter', False):
            minor += 1  # v0.1.0

        if self.features.get('fuzzy_logic', False):
            minor += 1  # v0.2.0

        if self.features.get('kelly_criterion', False):
            minor += 1  # v0.3.0

        # Predictive Intelligence components
        predictive_count = sum([
            self.features.get('trajectory_predictor', False),
            self.features.get('momentum_persistence', False),
            self.features.get('recovery_detector', False)
        ])

        if predictive_count > 0:
            minor += predictive_count  # Each predictor = +0.1.0

        return (major, minor, patch)

    def get_version_string(self) -> str:
        """Get version as string (MAJOR.MINOR.PATCH)."""
        major, minor, patch = self.effective_version
        return f"{major}.{minor}.{patch}"

    def get_detailed_version(self) -> str:
        """
        Get detailed version with feature breakdown.

        Example: "v0.6.0 (Kalman + Fuzzy + Kelly + Predictive)"
        """
        major, minor, patch = self.effective_version
        version_str = f"v{major}.{minor}.{patch}"

        # Build feature list
        feature_list = []

        if self.features.get('kalman_filter', False):
            feature_list.append("Kalman")

        if self.features.get('fuzzy_logic', False):
            feature_list.append("Fuzzy")

        if self.features.get('kelly_criterion', False):
            feature_list.append("Kelly")

        # Check if all 3 predictive components enabled
        predictive_all = all([
            self.features.get('trajectory_predictor', False),
            self.features.get('momentum_persistence', False),
            self.features.get('recovery_detector', False)
        ])

        if predictive_all:
            feature_list.append("Predictive")
        else:
            # Add individual predictive components
            if self.features.get('trajectory_predictor', False):
                feature_list.append("Trajectory")
            if self.features.get('momentum_persistence', False):
                feature_list.append("Momentum")
            if self.features.get('recovery_detector', False):
                feature_list.append("Recovery")

        if feature_list:
            features_str = " + ".join(feature_list)
            return f"{version_str} ({features_str})"
        else:
            return f"{version_str} (Core)"

    def get_exit_strategy_version(self) -> str:
        """Get exit strategy version label."""
        if self.features.get('predictive_intelligence', False):
            return "Exit v6.3 Predictive Intelligence"
        elif self.features.get('advanced_exits', False):
            return "Exit v6.2 Advanced"
        elif self.features.get('kalman_filter', False):
            return "Exit v6.0 Kalman"
        else:
            return "Exit v5.0 Dynamic"

    def print_version_info(self):
        """Print comprehensive version information."""
        logger.info("=" * 60)
        logger.info(f"XAUBot AI {self.get_detailed_version()}")
        logger.info(f"Exit Strategy: {self.get_exit_strategy_version()}")
        logger.info("=" * 60)
        logger.info("Enabled Features:")

        for feature, enabled in sorted(self.features.items()):
            status = "✓" if enabled else "✗"
            feature_name = feature.replace('_', ' ').title()
            logger.info(f"  [{status}] {feature_name}")

        logger.info("=" * 60)

    def get_component_versions(self) -> Dict[str, str]:
        """Get version info for each component."""
        return {
            "core": self.get_version_string(),
            "exit_strategy": self.get_exit_strategy_version(),
            "detailed": self.get_detailed_version(),
            "base": f"{self.base_version[0]}.{self.base_version[1]}.{self.base_version[2]}",
            "effective": self.get_version_string()
        }


# Global version instance
__version_manager__ = VersionManager()

# Expose convenient module-level attributes
__version__ = __version_manager__.get_version_string()
__version_detailed__ = __version_manager__.get_detailed_version()
__exit_strategy__ = __version_manager__.get_exit_strategy_version()


def get_version() -> str:
    """Get version string."""
    return __version__


def get_detailed_version() -> str:
    """Get detailed version with features."""
    return __version_detailed__


def print_version_info():
    """Print version information."""
    __version_manager__.print_version_info()


if __name__ == "__main__":
    # Test version detection
    print_version_info()
    print(f"\nVersion: {get_version()}")
    print(f"Detailed: {get_detailed_version()}")
    print(f"\nComponents: {__version_manager__.get_component_versions()}")
