"""
Strategy Factory

Creates trading strategies from configuration.
"""

import logging
import sys
from typing import Dict, List, Type

import backtrader as bt

# Import decision-engine components
import os
decision_engine_path = os.environ.get("DECISION_ENGINE_PATH", "/app")
if decision_engine_path not in sys.path:
    sys.path.insert(0, decision_engine_path)

from decision_engine.rules.base import Rule
from decision_engine.rules.registry import RULE_REGISTRY, RuleRegistry

from .rule_based_strategy import DecisionEngineStrategy

logger = logging.getLogger(__name__)


# Available rule configurations
RULE_CONFIGS: Dict[str, Dict] = {
    "buy_dip_in_uptrend": {
        "description": "Buy when RSI dips in uptrend (your primary rule)",
        "default_params": {"rsi_threshold": 40.0},
    },
    "strong_buy_signal": {
        "description": "Buy when RSI dips in strong triple-SMA uptrend",
        "default_params": {"rsi_threshold": 35.0},
    },
    "rsi_oversold": {
        "description": "Buy when RSI is oversold",
        "default_params": {"threshold": 30.0},
    },
    "rsi_overbought": {
        "description": "Sell when RSI is overbought",
        "default_params": {"threshold": 70.0},
    },
    "macd_bullish_crossover": {
        "description": "Buy on MACD bullish crossover",
        "default_params": {},
    },
    "macd_bearish_crossover": {
        "description": "Sell on MACD bearish crossover",
        "default_params": {},
    },
    "weekly_uptrend": {
        "description": "Weekly uptrend (SMA_20 > SMA_50)",
        "default_params": {},
    },
    "monthly_uptrend": {
        "description": "Monthly uptrend (SMA_50 > SMA_200)",
        "default_params": {},
    },
    "golden_cross": {
        "description": "Full trend alignment (SMA_20 > SMA_50 > SMA_200)",
        "default_params": {},
    },
    # Enhanced rules
    "enhanced_buy_dip": {
        "description": "Enhanced dip buying with volume confirmation and stricter filters",
        "default_params": {"rsi_oversold": 35.0, "min_trend_spread": 1.5},
    },
    "momentum_reversal": {
        "description": "Buy RSI recovery + MACD bullish crossover",
        "default_params": {},
    },
    "trend_continuation": {
        "description": "Buy pullbacks to SMA_20 in strong uptrend",
        "default_params": {"pullback_tolerance_pct": 2.0},
    },
    "average_down": {
        "description": "Scale into position on deeper dips (use with --allow-scale-in)",
        "default_params": {"rsi_extreme": 30.0, "max_scale_ins": 2},
    },
    # Mining Stock Rules
    "commodity_breakout": {
        "description": "Buy miners on breakout above SMA_20 (commodity leverage play)",
        "default_params": {"breakout_threshold_pct": 2.0},
    },
    "miner_metal_ratio": {
        "description": "Buy miners when oversold at support (mean reversion)",
        "default_params": {"rsi_oversold": 35.0},
    },
    "dollar_weakness": {
        "description": "Buy miners in strong uptrend (USD weakness indicator)",
        "default_params": {"min_trend_spread": 2.0},
    },
    "seasonality": {
        "description": "Adjust signals based on seasonal patterns (strong Jan-Feb, Aug-Sep)",
        "default_params": {},
    },
    "volume_breakout": {
        "description": "Buy miners on 1.5x+ volume breakouts (high conviction)",
        "default_params": {"min_volume_ratio": 1.5},
    },
}


def list_available_rules() -> Dict[str, str]:
    """Get list of available rules with descriptions."""
    return {name: config["description"] for name, config in RULE_CONFIGS.items()}


POSITIVE_NUMERIC_PARAMS = {
    "rsi_threshold", "rsi_oversold", "rsi_extreme", "threshold",
    "min_trend_spread", "pullback_tolerance_pct", "breakout_threshold_pct",
    "min_volume_ratio", "max_scale_ins",
}

CONFIDENCE_PARAMS = {"min_confidence", "confidence"}


def _validate_rule_params(name: str, rule_params: Dict) -> bool:
    for key, value in rule_params.items():
        if value is None:
            logger.error(f"Rule {name}: parameter '{key}' is None")
            return False
        if key in POSITIVE_NUMERIC_PARAMS:
            if not isinstance(value, (int, float)) or value <= 0:
                logger.error(f"Rule {name}: parameter '{key}' must be a positive number, got {value}")
                return False
        if key in CONFIDENCE_PARAMS:
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                logger.error(f"Rule {name}: parameter '{key}' must be in [0, 1], got {value}")
                return False
    return True


def create_rules(rule_names: List[str], params: Dict[str, Dict] = None) -> List[Rule]:
    """
    Create Rule instances from rule names.

    Args:
        rule_names: List of rule names to create
        params: Optional dict of {rule_name: {param: value}} overrides

    Returns:
        List of Rule instances
    """
    params = params or {}
    rules = []

    for name in rule_names:
        if name not in RULE_REGISTRY:
            logger.warning(f"Unknown rule: {name}, skipping")
            continue

        # Get default params and merge with overrides
        rule_params = RULE_CONFIGS.get(name, {}).get("default_params", {}).copy()
        if name in params:
            rule_params.update(params[name])

        if not _validate_rule_params(name, rule_params):
            logger.warning(f"Skipping rule {name} due to invalid parameters")
            continue

        try:
            rule = RuleRegistry.create_rule(name, rule_params)
            rules.append(rule)
            logger.info(f"Created rule: {rule.name}")
        except Exception as e:
            logger.error(f"Failed to create rule {name}: {e}")

    return rules


def create_strategy(
    rule_names: List[str] = None,
    min_confidence: float = 0.6,
    require_consensus: bool = False,
    profit_target: float = 0.07,
    stop_loss: float = 0.05,
    rule_params: Dict[str, Dict] = None,
    allow_scale_in: bool = False,
    max_scale_ins: int = 2,
    scale_in_size: float = 0.5,
    stop_mode: str = "fixed",
    atr_multiplier: float = 2.0,
    atr_stop_min_pct: float = 3.0,
    atr_stop_max_pct: float = 15.0,
    max_price_extension_pct: float = 15.0,
    cooldown_bars: int = 5,
    max_trend_spread_pct: float = 20.0,
) -> Type[bt.Strategy]:
    """
    Create a backtrader Strategy class with decision-engine rules.

    Args:
        rule_names: List of rule names to use (default: buy_dip_in_uptrend)
        min_confidence: Minimum confidence to trigger signals
        require_consensus: Require multiple rules to agree
        profit_target: Profit target percentage
        stop_loss: Stop loss percentage
        rule_params: Optional parameter overrides per rule
        allow_scale_in: Allow averaging down on deeper dips
        max_scale_ins: Maximum number of scale-ins per position
        scale_in_size: Scale-in size relative to initial position
        stop_mode: "fixed" (% of entry) or "atr" (ATR-based)
        atr_multiplier: ATR multiplier for stop calculation
        atr_stop_min_pct: Minimum stop distance %
        atr_stop_max_pct: Maximum stop distance %
        max_price_extension_pct: Skip buy if price > X% above SMA_20
        cooldown_bars: Wait N bars after exit before re-entering
        max_trend_spread_pct: Skip buy if SMA_20/SMA_50 spread > X%

    Returns:
        Configured DecisionEngineStrategy class
    """
    if rule_names is None:
        rule_names = ["buy_dip_in_uptrend"]

    rules = create_rules(rule_names, rule_params)

    if not rules:
        raise ValueError("No valid rules created")

    # Create a subclass with the rules baked in
    class ConfiguredStrategy(DecisionEngineStrategy):
        params = (
            ("rules", rules),
            ("min_confidence", min_confidence),
            ("require_consensus", require_consensus),
            ("profit_target", profit_target),
            ("stop_loss", stop_loss),
            ("log_trades", True),
            ("allow_scale_in", allow_scale_in),
            ("max_scale_ins", max_scale_ins),
            ("scale_in_size", scale_in_size),
            ("stop_mode", stop_mode),
            ("atr_multiplier", atr_multiplier),
            ("atr_stop_min_pct", atr_stop_min_pct),
            ("atr_stop_max_pct", atr_stop_max_pct),
            ("max_price_extension_pct", max_price_extension_pct),
            ("cooldown_bars", cooldown_bars),
            ("max_trend_spread_pct", max_trend_spread_pct),
        )

    return ConfiguredStrategy
