"""Tests for backtesting.strategies.strategy_factory — RULE_CONFIGS,
list_available_rules, _validate_rule_params, create_rules, create_strategy."""

import pytest

from backtesting.strategies.strategy_factory import (
    CONFIDENCE_PARAMS,
    POSITIVE_NUMERIC_PARAMS,
    RULE_CONFIGS,
    _validate_rule_params,
    create_rules,
    create_strategy,
    list_available_rules,
)


# ---------------------------------------------------------------------------
# RULE_CONFIGS structure
# ---------------------------------------------------------------------------


class TestRuleConfigs:
    def test_not_empty(self):
        assert len(RULE_CONFIGS) > 0

    def test_all_have_description(self):
        for name, config in RULE_CONFIGS.items():
            assert "description" in config, f"{name} missing description"
            assert isinstance(config["description"], str)
            assert len(config["description"]) > 0

    def test_all_have_default_params(self):
        for name, config in RULE_CONFIGS.items():
            assert "default_params" in config, f"{name} missing default_params"
            assert isinstance(config["default_params"], dict)

    def test_known_rules_present(self):
        expected = [
            "buy_dip_in_uptrend",
            "strong_buy_signal",
            "rsi_oversold",
            "rsi_overbought",
            "macd_bullish_crossover",
            "macd_bearish_crossover",
            "golden_cross",
            "enhanced_buy_dip",
            "momentum_reversal",
            "trend_continuation",
            "average_down",
            "commodity_breakout",
            "miner_metal_ratio",
            "dollar_weakness",
            "seasonality",
            "volume_breakout",
        ]
        for name in expected:
            assert name in RULE_CONFIGS, f"Missing rule: {name}"

    def test_total_rule_count(self):
        # 17 rules as of current code
        assert len(RULE_CONFIGS) >= 17


# ---------------------------------------------------------------------------
# list_available_rules
# ---------------------------------------------------------------------------


class TestListAvailableRules:
    def test_returns_dict(self):
        result = list_available_rules()
        assert isinstance(result, dict)

    def test_matches_config_keys(self):
        result = list_available_rules()
        assert set(result.keys()) == set(RULE_CONFIGS.keys())

    def test_values_are_descriptions(self):
        result = list_available_rules()
        for name, desc in result.items():
            assert desc == RULE_CONFIGS[name]["description"]


# ---------------------------------------------------------------------------
# _validate_rule_params
# ---------------------------------------------------------------------------


class TestValidateRuleParams:
    def test_empty_params_valid(self):
        assert _validate_rule_params("test_rule", {}) is True

    def test_none_value_invalid(self):
        assert _validate_rule_params("test", {"rsi_threshold": None}) is False

    def test_positive_numeric_valid(self):
        assert _validate_rule_params("test", {"rsi_threshold": 30.0}) is True

    def test_positive_numeric_zero_invalid(self):
        assert _validate_rule_params("test", {"rsi_threshold": 0}) is False

    def test_positive_numeric_negative_invalid(self):
        assert _validate_rule_params("test", {"rsi_threshold": -5.0}) is False

    def test_positive_numeric_string_invalid(self):
        assert _validate_rule_params("test", {"rsi_threshold": "thirty"}) is False

    def test_confidence_valid_bounds(self):
        assert _validate_rule_params("test", {"min_confidence": 0.0}) is True
        assert _validate_rule_params("test", {"min_confidence": 0.5}) is True
        assert _validate_rule_params("test", {"min_confidence": 1.0}) is True

    def test_confidence_below_zero_invalid(self):
        assert _validate_rule_params("test", {"min_confidence": -0.1}) is False

    def test_confidence_above_one_invalid(self):
        assert _validate_rule_params("test", {"min_confidence": 1.1}) is False

    def test_confidence_string_invalid(self):
        assert _validate_rule_params("test", {"confidence": "high"}) is False

    def test_unknown_param_passes(self):
        """Parameters not in POSITIVE_NUMERIC_PARAMS or CONFIDENCE_PARAMS are allowed."""
        assert _validate_rule_params("test", {"custom_flag": True}) is True
        assert _validate_rule_params("test", {"custom_flag": "text"}) is True

    def test_multiple_params_first_fails(self):
        """Returns False as soon as any param is invalid."""
        result = _validate_rule_params(
            "test", {"rsi_threshold": -1.0, "min_confidence": 0.5}
        )
        assert result is False

    def test_all_positive_numeric_param_names(self):
        """Verify the set of recognized positive-numeric parameter names."""
        expected = {
            "rsi_threshold", "rsi_oversold", "rsi_extreme", "threshold",
            "min_trend_spread", "pullback_tolerance_pct", "breakout_threshold_pct",
            "min_volume_ratio", "max_scale_ins",
        }
        assert POSITIVE_NUMERIC_PARAMS == expected

    def test_all_confidence_param_names(self):
        assert CONFIDENCE_PARAMS == {"min_confidence", "confidence"}


# ---------------------------------------------------------------------------
# create_rules
# ---------------------------------------------------------------------------


class TestCreateRules:
    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            create_rules(["nonexistent_rule_xyz"])

    def test_known_rule_created(self):
        rules = create_rules(["buy_dip_in_uptrend"])
        assert len(rules) == 1
        # Rule.name is the display name, not the registry key
        assert "buy" in rules[0].name.lower() and "dip" in rules[0].name.lower()

    def test_multiple_rules(self):
        rules = create_rules(["buy_dip_in_uptrend", "rsi_oversold"])
        assert len(rules) == 2
        names_lower = [r.name.lower() for r in rules]
        assert any("dip" in n for n in names_lower)
        assert any("rsi" in n for n in names_lower)

    def test_mixed_known_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            create_rules(["buy_dip_in_uptrend", "does_not_exist"])

    def test_param_override(self):
        rules = create_rules(
            ["buy_dip_in_uptrend"],
            params={"buy_dip_in_uptrend": {"rsi_threshold": 35.0}},
        )
        assert len(rules) == 1

    def test_invalid_param_skips_rule(self):
        rules = create_rules(
            ["rsi_oversold"],
            params={"rsi_oversold": {"threshold": -1.0}},
        )
        assert len(rules) == 0

    def test_none_params_default(self):
        rules = create_rules(["macd_bullish_crossover"], params=None)
        assert len(rules) == 1

    def test_empty_list(self):
        rules = create_rules([])
        assert rules == []


# ---------------------------------------------------------------------------
# create_strategy
# ---------------------------------------------------------------------------


class TestCreateStrategy:
    def test_returns_class(self):
        strategy_class = create_strategy(["buy_dip_in_uptrend"])
        import backtrader as bt
        assert issubclass(strategy_class, bt.Strategy)

    def test_default_rules(self):
        """Default is ['buy_dip_in_uptrend'] when rule_names=None."""
        strategy_class = create_strategy()
        # Should not raise — creates with default rule
        assert strategy_class is not None

    def test_no_valid_rules_raises(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            create_strategy(["nonexistent_xyz"])

    def test_custom_parameters_baked_in(self):
        strategy_class = create_strategy(
            ["buy_dip_in_uptrend"],
            min_confidence=0.8,
            profit_target=0.10,
            stop_loss=0.03,
        )
        # Check params are baked into the class
        params_dict = dict(strategy_class.params._getitems())
        assert params_dict["min_confidence"] == 0.8
        assert params_dict["profit_target"] == 0.10
        assert params_dict["stop_loss"] == 0.03

    def test_scale_in_params(self):
        strategy_class = create_strategy(
            ["buy_dip_in_uptrend"],
            allow_scale_in=True,
            max_scale_ins=3,
            scale_in_size=0.25,
        )
        params_dict = dict(strategy_class.params._getitems())
        assert params_dict["allow_scale_in"] is True
        assert params_dict["max_scale_ins"] == 3
        assert params_dict["scale_in_size"] == 0.25

    def test_atr_stop_params(self):
        strategy_class = create_strategy(
            ["buy_dip_in_uptrend"],
            stop_mode="atr",
            atr_multiplier=3.0,
            atr_stop_min_pct=2.0,
            atr_stop_max_pct=10.0,
        )
        params_dict = dict(strategy_class.params._getitems())
        assert params_dict["stop_mode"] == "atr"
        assert params_dict["atr_multiplier"] == 3.0

    def test_max_loss_and_cooldown(self):
        strategy_class = create_strategy(
            ["buy_dip_in_uptrend"],
            max_loss_pct=8.0,
            cooldown_bars=10,
            max_trend_spread_pct=15.0,
            max_price_extension_pct=12.0,
        )
        params_dict = dict(strategy_class.params._getitems())
        assert params_dict["max_loss_pct"] == 8.0
        assert params_dict["cooldown_bars"] == 10
        assert params_dict["max_trend_spread_pct"] == 15.0
        assert params_dict["max_price_extension_pct"] == 12.0

    def test_exit_timeframe_param(self):
        strategy_class = create_strategy(
            ["buy_dip_in_uptrend"],
            exit_timeframe="5min",
        )
        params_dict = dict(strategy_class.params._getitems())
        assert params_dict["exit_timeframe"] == "5min"
