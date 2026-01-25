"""
Strategy Factory

Creates trading strategies from configuration.
"""

import logging
import sys
from typing import Dict, List, Type

import backtrader as bt

# Import decision-engine components
sys.path.insert(0, "/Users/thomasrogers/Projects/decision-engine")

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
}


def list_available_rules() -> Dict[str, str]:
    """Get list of available rules with descriptions."""
    return {name: config["description"] for name, config in RULE_CONFIGS.items()}


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
        )

    return ConfiguredStrategy
