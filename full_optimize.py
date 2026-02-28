#!/usr/bin/env python3
"""
Full 8-symbol optimization runner.
Creates {symbol}-rules.md files with top 5 configs per symbol.
Usage: python full_optimize.py SYMBOL [--rounds N]
"""

import subprocess
import sys
import os
import json
import time
import random
import tempfile
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BacktestResult:
    symbol: str
    rules: List[str]
    params: Dict
    total_return: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    command: str = ""
    label: str = ""
    avg_win: float = 0.0
    avg_loss: float = 0.0
    annual_return: float = 0.0


def run_backtest(symbol: str, rules: List[str], params: Dict, label: str = "") -> Optional[BacktestResult]:
    """Run a single backtest, parse JSON results."""
    json_out = tempfile.mktemp(suffix=".json")

    cmd = [
        sys.executable, "-m", "backtesting",
        "--symbol", symbol,
        "--start", "2021-01-01",
        "--end", "2026-02-22",
        "--timeframe", "daily",
        "--exit-timeframe", "5min",
        "--rules", ",".join(rules),
        "--profit-target", str(params.get("profit_target", 0.07)),
        "--stop-loss", str(params.get("stop_loss", 1.0)),
        "--min-confidence", str(params.get("min_confidence", 0.6)),
        "--max-loss", str(params.get("max_loss", 5.0)),
        "--cooldown-bars", str(params.get("cooldown_bars", 5)),
        "--cash", str(params.get("cash", 1000)),
        "--output", json_out,
        "--quiet",
    ]

    if params.get("stop_mode") == "atr":
        cmd.extend(["--stop-mode", "atr"])
        cmd.extend(["--atr-multiplier", str(params.get("atr_multiplier", 2.0))])

    if params.get("sizing_mode") == "risk_based":
        cmd.extend(["--sizing-mode", "risk_based"])
        cmd.extend(["--risk-pct", str(params.get("risk_pct", 5.0))])
        cmd.extend(["--max-position-pct", str(params.get("max_position_pct", 20.0))])

    if params.get("max_extension"):
        cmd.extend(["--max-extension", str(params["max_extension"])])

    if params.get("max_trend_spread"):
        cmd.extend(["--max-trend-spread", str(params["max_trend_spread"])])

    cmd_str = " ".join(cmd)

    env = os.environ.copy()
    env["DECISION_ENGINE_PATH"] = os.path.expanduser("~/Projects/decision-engine")
    env["PYTHONPATH"] = os.path.expanduser("~/Projects/decision-engine") + ":" + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                cwd=os.path.expanduser("~/Projects/backtesting-service"), env=env)
    except subprocess.TimeoutExpired:
        try: os.unlink(json_out)
        except: pass
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        try: os.unlink(json_out)
        except: pass
        return None

    try:
        with open(json_out) as f:
            data = json.load(f)
        os.unlink(json_out)
    except:
        try: os.unlink(json_out)
        except: pass
        return None

    br = BacktestResult(symbol=symbol, rules=rules, params=params, command=cmd_str, label=label)
    br.total_return = data.get("performance", {}).get("total_return", 0) * 100
    br.annual_return = data.get("performance", {}).get("annual_return", 0) * 100
    br.trades = data.get("trades", {}).get("total", 0)
    br.win_rate = data.get("trades", {}).get("win_rate", 0) * 100
    br.profit_factor = data.get("risk_metrics", {}).get("profit_factor", 0)
    br.sharpe = data.get("risk_metrics", {}).get("sharpe_ratio", 0)
    br.max_drawdown = data.get("risk_metrics", {}).get("max_drawdown_pct", 0)
    br.avg_win = data.get("trades", {}).get("avg_win", 0)
    br.avg_loss = data.get("trades", {}).get("avg_loss", 0)

    if br.trades == 0:
        return None

    return br


# =============================================================================
# SYMBOL RESEARCH CONTEXT
# =============================================================================

SYMBOL_CONTEXT = {
    "CCJ": """**Cameco (Uranium)** — Pure uranium play, beta 1.52. Uranium at 14-year highs (~$86.50/lb).
Nuclear renaissance tailwind (AI data centers, government policy). Supply deficit: 180M lbs demand vs 130M lbs production.
High correlation with URNM/UUUU. Best strategy: trend-following on uranium breakouts, buy dips to support.
Seasonal: Q1-Q2 stronger (utility procurement). At $117/share — expensive for $888 account. Avoid extended holds above resistance without catalyst.""",

    "CAT": """**Caterpillar (Industrial)** — Macro industrial bellwether, beta 1.05-1.21. Driven by construction spending,
mining cycles, infrastructure. Outperforms in inflationary/capex boom periods. Clean trending stock ideal for
dip-buying in strong trends. At $350+ — outside current account range but good for backtesting optimal rules.
Best strategy: buy oversold dips to 50-SMA within uptrend, target 2:1 R:R over 5-10 days.""",

    "WPM": """**Wheaton Precious Metals (Streaming)** — Gold/silver streamer with fixed-cost model and 1.5-2.5x leverage
to metal prices. 59% gold / 39% silver exposure. Production growth: +25% by 2026, +50% by 2030.
Seasonal strength Sept-Dec. Best strategy: trend-following on golden cross + EMA confluence. Buy pullbacks
to 50-SMA, exit on close below 21-SMA. Mean reversion works but needs tighter stops.""",

    "PPLT": """**Platinum ETF** — Physical platinum, driven by auto catalysts (38%), hydrogen economy, supply deficit (966k oz 2025).
Higher volatility than gold/silver due to thin liquidity. Outperformed gold by 10% in 2025. Strong seasonal
Jan (+1.9%) and Sep (+1.7%). Current baseline: 59.4% WR, +72%, 0.81 Sharpe — already solid.
Best strategy: mean-reversion oversold bounces + trend continuation. Size smaller due to slippage.""",

    "URNM": """**Sprott Uranium Miners ETF** — Uranium mining companies index. Structural bull from supply deficit (28% shortfall).
High correlation with CCJ. 0.75% expense ratio drag. Volatile 15-25% intra-trend swings.
Best strategy: trend-following — buy dips into support during uranium bull phases, hold 2-4 weeks.
Don't panic-sell volatility. Use 2-3x ATR stops.""",

    "SLV": """**iShares Silver Trust** — Physical silver, 52% industrial / 48% investment. Seasonal edge Dec-Feb (+7% avg).
Summer weakness Jun-Jul. 2025 rally +144%. Higher volatility than gold. Physically backed (no contango).
Best strategy: seasonal mean reversion — buy Oct-Feb weakness, sell Feb-Mar strength.
Current at resistance — consolidation/pullback likely. Use 5-10% stops for swings.""",

    "MP": """**MP Materials (Rare Earth)** — Only large-scale U.S. rare earth facility. Beta ~1.70. Extreme volatility with
binary outcomes (China export controls = gap days). 5-15% daily swings routine. Max DD -48.9%.
Best strategy: momentum on geopolitical triggers, size 1-2% risk max. Buy confirmed breakouts above resistance
on volume. Hard 5% stops. Don't hold through earnings. Take 15-25% gains quickly.""",

    "UUUU": """**Energy Fuels (Uranium + Rare Earth)** — Dual uranium + rare earth exposure. Only U.S. producer doing both.
2M lbs/year uranium by mid-2026. 15-30% monthly swings routine. High execution risk (ramping facilities).
Best strategy: trend-follow with high-conviction entries. Buy strength above resistance on volume.
Use ATR-based stops (10% or 2x ATR). Trade as quarterly swing (+15-25% over 4-6 weeks).""",
}


# =============================================================================
# RULE SETS
# =============================================================================

# Core rules available in the system
CORE_ENTRY = ["enhanced_buy_dip", "momentum_reversal", "trend_continuation"]
TREND_RULES = ["trend_alignment", "golden_cross"]
EXIT_RULES = ["trend_break_warning", "death_cross"]
RSI_RULES = ["rsi_oversold", "rsi_overbought"]
MACD_RULES = ["macd_bearish_crossover"]
MINING_RULES = ["commodity_breakout", "miner_metal_ratio", "dollar_weakness", "seasonality", "volume_breakout"]
COMPOSITE_RULES = ["rsi_macd_confluence", "dip_recovery"]
LEGACY_ENTRY = ["buy_dip_in_uptrend", "strong_buy_signal"]


# =============================================================================
# CONFIGURATION GENERATORS
# =============================================================================

def get_ccj_configs():
    """CCJ: Cameco — uranium, trend-following, high beta."""
    configs = []

    # Current baseline from rules.yaml
    configs.append(("Baseline (rules.yaml: trend_cont + seasonality + death_cross)",
        ["trend_continuation", "seasonality", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Add trend alignment to baseline
    configs.append(("Baseline + trend_alignment",
        ["trend_continuation", "seasonality", "death_cross", "trend_alignment"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Add golden cross (uranium tends to have strong trend signals)
    configs.append(("Baseline + golden_cross + trend_alignment",
        ["trend_continuation", "seasonality", "death_cross", "trend_alignment", "golden_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Full core entry rules (uranium is trending, try dip buying)
    configs.append(("Core 3 + seasonality + exits",
        CORE_ENTRY + ["seasonality"] + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    configs.append(("Core 3 + trend_alignment + exits",
        CORE_ENTRY + ["trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Mining-specific rules (CCJ is a uranium miner)
    configs.append(("Mining rules + seasonality",
        ["commodity_breakout", "volume_breakout", "miner_metal_ratio", "seasonality"] + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    configs.append(("Full mining suite + trend",
        MINING_RULES + ["trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Kitchen sink
    configs.append(("Everything: core + mining + trend + exits",
        CORE_ENTRY + MINING_RULES + TREND_RULES + EXIT_RULES + RSI_RULES + MACD_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Composites
    configs.append(("Core + composites + exits",
        CORE_ENTRY + COMPOSITE_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Profit target sweeps
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        configs.append((f"Baseline + PT {pt}",
            ["trend_continuation", "seasonality", "death_cross"],
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.50, 0.55, 0.60, 0.70, 0.75]:
        configs.append((f"Baseline + confidence {mc}",
            ["trend_continuation", "seasonality", "death_cross"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
        ))

    # Max loss sweeps
    for ml in [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        configs.append((f"Baseline + ML {ml}%",
            ["trend_continuation", "seasonality", "death_cross"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": ml, "cooldown_bars": 7, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [3, 5, 7, 10, 14]:
        configs.append((f"Baseline + cooldown {cb}",
            ["trend_continuation", "seasonality", "death_cross"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops (uranium is volatile, ATR-based stops may help)
    for atr in [1.5, 2.0, 2.5, 3.0, 3.5]:
        configs.append((f"Baseline + ATR {atr}x",
            ["trend_continuation", "seasonality", "death_cross"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Combined best guesses
    configs.append(("Core + seasonality + ATR 2.5 + PT 0.12",
        CORE_ENTRY + ["seasonality"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 8.0, "cooldown_bars": 7,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    configs.append(("Mining + golden_cross + ATR 2.0 + PT 0.15",
        ["commodity_breakout", "volume_breakout", "seasonality", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 8.0, "cooldown_bars": 7,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("Trend follow: golden_cross + trend_align + trend_cont + exits",
        ["golden_cross", "trend_alignment", "trend_continuation"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 8.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Wider PT for high-beta uranium (let winners run)
    configs.append(("Core + PT 0.20 + ML 10% (let winners run)",
        CORE_ENTRY + ["seasonality", "trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.20, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 10.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        ["trend_continuation", "seasonality", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    configs.append(("Core + risk-based 3%",
        CORE_ENTRY + ["seasonality"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 8.0, "cooldown_bars": 7,
         "sizing_mode": "risk_based", "risk_pct": 3.0, "max_position_pct": 20.0, "cash": 1000}
    ))

    return configs


def get_cat_configs():
    """CAT: Caterpillar — industrial blue-chip, strong trend follower."""
    configs = []

    # Baseline from rules.yaml
    configs.append(("Baseline (rules.yaml)",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add golden_cross + death_cross
    configs.append(("Baseline + golden_cross + death_cross",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
         "golden_cross", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PT sweeps
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15]:
        configs.append((f"Baseline + PT {pt}",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Max loss sweeps
    for ml in [4.0, 6.0, 8.0, 10.0]:
        configs.append((f"Baseline + ML {ml}%",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [3, 5, 7, 10]:
        configs.append((f"Baseline + cooldown {cb}",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops
    for atr in [1.5, 2.0, 2.5, 3.0]:
        configs.append((f"Baseline + ATR {atr}x",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Trend-heavy (CAT is a strong trend follower)
    configs.append(("Trend-heavy: trend_cont + trend_align + golden + exits",
        ["trend_continuation", "trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Minimal dip buying
    configs.append(("Minimal: enhanced_buy_dip + trend_align + exits",
        ["enhanced_buy_dip", "trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Composites
    configs.append(("Baseline + rsi_macd_confluence + dip_recovery",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
         "rsi_macd_confluence", "dip_recovery"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Combined best guesses
    configs.append(("Golden+death + PT 0.08 + ATR 2.0",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
         "golden_cross", "death_cross"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("PT 0.12 + ML 8% + cooldown 7 (let winners run)",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 7, "cash": 1000}
    ))

    configs.append(("PT 0.08 + ATR 2.0 + confidence 0.60",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 5.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    return configs


def get_wpm_configs():
    """WPM: Wheaton Precious Metals — streaming company, gold/silver leverage."""
    configs = []

    baseline_rules = [
        "enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
        "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
        "commodity_breakout", "miner_metal_ratio", "dollar_weakness", "seasonality", "volume_breakout"
    ]

    # Baseline
    configs.append(("Baseline (rules.yaml — 13 rules)",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PT sweeps
    for pt in [0.06, 0.08, 0.10, 0.15, 0.20]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Max loss sweeps
    for ml in [4.0, 5.0, 8.0, 10.0]:
        configs.append((f"Baseline + ML {ml}%",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

    # ATR stops
    for atr in [1.5, 2.0, 2.5, 3.0]:
        configs.append((f"Baseline + ATR {atr}x",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [3, 7, 10]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # Reduced rule sets
    configs.append(("Reduced: core 3 + exits",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Reduced: mining focus",
        ["commodity_breakout", "miner_metal_ratio", "volume_breakout", "dollar_weakness", "seasonality",
         "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Reduced: trend + mining",
        ["enhanced_buy_dip", "trend_continuation", "trend_alignment", "golden_cross",
         "commodity_breakout", "volume_breakout", "seasonality", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add golden/death cross to baseline
    configs.append(("Baseline + golden_cross + death_cross",
        baseline_rules + ["golden_cross", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PPLT-style (worked for another PM)
    configs.append(("PPLT-style rules",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Composites
    configs.append(("Baseline + rsi_macd_confluence + dip_recovery",
        baseline_rules + ["rsi_macd_confluence", "dip_recovery"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Combined best guesses
    configs.append(("Full rules + ATR 2.0 + PT 0.10",
        baseline_rules + ["golden_cross", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 5.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("Mining + trend + PT 0.15 + ATR 2.5",
        ["commodity_breakout", "volume_breakout", "seasonality", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    return configs


def get_pplt_configs():
    """PPLT: Platinum ETF — volatile commodity, trend follower with breakouts."""
    configs = []

    # Baseline
    baseline_rules = ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"]

    configs.append(("Baseline (rules.yaml)",
        baseline_rules,
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # PT sweeps
    for pt in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
        ))

    # Max loss sweeps
    for ml in [3.0, 4.0, 5.0, 6.0, 8.0]:
        configs.append((f"Baseline + ML {ml}%",
            baseline_rules,
            {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": ml, "cooldown_bars": 3, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [1, 3, 5, 7, 10]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops
    for atr in [1.5, 2.0, 2.5, 3.0]:
        configs.append((f"Baseline + ATR {atr}x",
            baseline_rules,
            {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Mining rules (platinum IS a commodity)
    configs.append(("Baseline + commodity_breakout + volume_breakout",
        baseline_rules + ["commodity_breakout", "volume_breakout"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("Baseline + all mining rules",
        baseline_rules + ["commodity_breakout", "volume_breakout", "miner_metal_ratio", "dollar_weakness", "seasonality"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Minimal rules
    configs.append(("Minimal: golden_cross + trend_align + death_cross",
        ["golden_cross", "trend_alignment", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("Minimal: trend_cont + trend_align + exits",
        ["trend_continuation", "trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Composites
    configs.append(("Baseline + rsi_macd_confluence + dip_recovery",
        baseline_rules + ["rsi_macd_confluence", "dip_recovery"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Combined best guesses
    configs.append(("PT 0.08 + ML 5% + cooldown 5",
        baseline_rules,
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("PT 0.08 + ML 5% + ATR 2.0",
        baseline_rules,
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 5.0, "cooldown_bars": 3,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("PT 0.10 + ML 6% + cooldown 5",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Mining rules + ATR 2.0 + PT 0.08",
        baseline_rules + ["commodity_breakout", "volume_breakout", "seasonality"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("PT 0.12 + ML 6% (let winners run)",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Lower confidence for more trades
    configs.append(("Confidence 0.50 + ML 5%",
        baseline_rules,
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Legacy rules
    configs.append(("strong_buy + momentum + trend rules",
        ["strong_buy_signal", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("buy_dip_in_uptrend + trend rules",
        ["buy_dip_in_uptrend", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        baseline_rules,
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 30.0, "cash": 1000}
    ))

    return configs


def get_urnm_configs():
    """URNM: Sprott Uranium Miners ETF — trend-following uranium bull."""
    configs = []

    # Baseline from rules.yaml (3 entry rules only)
    baseline_rules = ["enhanced_buy_dip", "momentum_reversal", "trend_continuation"]

    configs.append(("Baseline (rules.yaml: core 3 only)",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add trend rules (URNM is a trend follower)
    configs.append(("Core + trend_alignment + exits",
        baseline_rules + ["trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Core + golden_cross + trend_alignment + exits",
        baseline_rules + ["trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Mining rules (URNM IS a mining ETF)
    configs.append(("Core + mining rules + exits",
        baseline_rules + MINING_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Mining focus: breakout + volume + seasonality + exits",
        ["commodity_breakout", "volume_breakout", "seasonality"] + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Full kitchen sink
    configs.append(("Everything: core + mining + trend + RSI + MACD + exits",
        baseline_rules + MINING_RULES + TREND_RULES + EXIT_RULES + RSI_RULES + MACD_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PT sweeps
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Max loss sweeps (URNM is volatile — wider stops may help)
    for ml in [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        configs.append((f"Baseline + ML {ml}%",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [3, 5, 7, 10, 14]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops (important for volatile uranium)
    for atr in [1.5, 2.0, 2.5, 3.0, 3.5]:
        configs.append((f"Baseline + ATR {atr}x",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Combined best guesses
    configs.append(("Core + trend + PT 0.15 + ML 8% + ATR 2.5 (trend follow)",
        baseline_rules + ["trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 7,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    configs.append(("Mining + trend + PT 0.12 + ATR 2.0",
        ["commodity_breakout", "volume_breakout", "seasonality", "trend_continuation",
         "trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 6.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("PT 0.20 + ML 10% + ATR 3.0 (let uranium run)",
        baseline_rules + ["trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.20, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 10.0, "cooldown_bars": 7,
         "stop_mode": "atr", "atr_multiplier": 3.0, "cash": 1000}
    ))

    # Composites
    configs.append(("Core + composites + exits",
        baseline_rules + COMPOSITE_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    configs.append(("Core + trend + risk-based 3%",
        baseline_rules + ["trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 7,
         "sizing_mode": "risk_based", "risk_pct": 3.0, "max_position_pct": 20.0, "cash": 1000}
    ))

    return configs


def get_slv_configs():
    """SLV: iShares Silver Trust — seasonal mean reversion + trend following."""
    configs = []

    # Baseline from rules.yaml (13 rules — kitchen sink)
    baseline_rules = [
        "enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
        "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
        "commodity_breakout", "miner_metal_ratio", "dollar_weakness", "seasonality", "volume_breakout"
    ]

    configs.append(("Baseline (rules.yaml — 13 rules)",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PT sweeps
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Max loss sweeps
    for ml in [4.0, 5.0, 6.0, 8.0, 10.0]:
        configs.append((f"Baseline + ML {ml}%",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [3, 5, 7, 10]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops
    for atr in [1.5, 2.0, 2.5, 3.0]:
        configs.append((f"Baseline + ATR {atr}x",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Reduced: core only (silver may not need mining-specific rules)
    configs.append(("Reduced: core 3 + RSI + MACD + trend + exits",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation",
         "rsi_oversold", "rsi_overbought", "macd_bearish_crossover",
         "trend_alignment", "trend_break_warning"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Seasonality focus (silver has strong seasonal patterns)
    configs.append(("Seasonality focus: core + seasonality + dollar_weakness + exits",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation",
         "seasonality", "dollar_weakness", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PPLT-style
    configs.append(("PPLT-style: core + trend + golden/death",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Trend-heavy
    configs.append(("Trend-heavy: trend_cont + align + golden + exits",
        ["trend_continuation", "trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Composites
    configs.append(("Baseline + rsi_macd_confluence + dip_recovery",
        baseline_rules + ["rsi_macd_confluence", "dip_recovery"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add golden/death cross
    configs.append(("Baseline + golden_cross + death_cross",
        baseline_rules + ["golden_cross", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Combined best guesses
    configs.append(("Full + ATR 2.0 + PT 0.10 + ML 5%",
        baseline_rules + ["golden_cross", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 5.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("Seasonal + trend + PT 0.15 + ATR 2.5",
        ["enhanced_buy_dip", "trend_continuation", "seasonality", "dollar_weakness",
         "trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    configs.append(("PT 0.20 + ML 10% (let silver run)",
        baseline_rules,
        {"profit_target": 0.20, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 10.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    return configs


def get_mp_configs():
    """MP: MP Materials — rare earth, high-beta, momentum plays."""
    configs = []

    # Baseline from rules.yaml
    baseline_rules = ["enhanced_buy_dip", "momentum_reversal", "trend_continuation"]

    configs.append(("Baseline (rules.yaml: core 3)",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add trend + exit rules
    configs.append(("Core + trend_alignment + exits",
        baseline_rules + ["trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Core + golden_cross + trend_alignment + exits",
        baseline_rules + ["trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Mining rules
    configs.append(("Core + mining rules + exits",
        baseline_rules + MINING_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # RSI + MACD (MP is volatile — RSI signals may be strong)
    configs.append(("Core + RSI + MACD + trend + exits",
        baseline_rules + RSI_RULES + MACD_RULES + ["trend_alignment", "trend_break_warning"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Kitchen sink
    configs.append(("Everything: core + mining + trend + RSI + MACD + exits",
        baseline_rules + MINING_RULES + TREND_RULES + EXIT_RULES + RSI_RULES + MACD_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PT sweeps (MP is high-beta — wider PTs may be better)
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Max loss sweeps (MP has -39% DD — need to find the right balance)
    for ml in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
        configs.append((f"Baseline + ML {ml}%",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

    # Cooldown sweeps (MP is volatile — longer cooldowns may prevent whipsaw)
    for cb in [3, 5, 7, 10, 14, 20]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops (critical for high-beta stock)
    for atr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        configs.append((f"Baseline + ATR {atr}x",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Composites
    configs.append(("Core + composites + exits",
        baseline_rules + COMPOSITE_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Combined best guesses for high-beta
    configs.append(("Core + trend + PT 0.15 + ML 8% + ATR 3.0 (let winners run)",
        baseline_rules + ["trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 7,
         "stop_mode": "atr", "atr_multiplier": 3.0, "cash": 1000}
    ))

    configs.append(("Core + trend + PT 0.20 + ML 10% + ATR 3.5 (high-beta)",
        baseline_rules + ["trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.20, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 10.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 3.5, "cash": 1000}
    ))

    configs.append(("Mining + momentum + PT 0.12 + ML 6%",
        ["commodity_breakout", "volume_breakout", "momentum_reversal"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    configs.append(("Tight: core + PT 0.06 + ML 3% + cooldown 3 (scalp approach)",
        baseline_rules + EXIT_RULES,
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 3.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Risk-based sizing (important for high-beta)
    configs.append(("Baseline + risk-based 3%",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 3.0, "max_position_pct": 20.0, "cash": 1000}
    ))

    configs.append(("Core + trend + risk-based 5% + ATR 2.5",
        baseline_rules + ["trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 7,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    return configs


def get_uuuu_configs():
    """UUUU: Energy Fuels — uranium + rare earth, high-beta trend follower."""
    configs = []

    # Baseline from rules.yaml
    baseline_rules = ["enhanced_buy_dip", "momentum_reversal", "trend_continuation"]

    configs.append(("Baseline (rules.yaml: core 3)",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add trend + exit rules
    configs.append(("Core + trend_alignment + exits",
        baseline_rules + ["trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Core + golden_cross + trend_alignment + exits",
        baseline_rules + ["trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Mining rules (UUUU is a uranium miner)
    configs.append(("Core + mining rules + exits",
        baseline_rules + MINING_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Mining focus: breakout + volume + seasonality + exits",
        ["commodity_breakout", "volume_breakout", "seasonality"] + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # RSI + MACD
    configs.append(("Core + RSI + MACD + trend + exits",
        baseline_rules + RSI_RULES + MACD_RULES + ["trend_alignment", "trend_break_warning"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Kitchen sink
    configs.append(("Everything: core + mining + trend + RSI + MACD + exits",
        baseline_rules + MINING_RULES + TREND_RULES + EXIT_RULES + RSI_RULES + MACD_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PT sweeps (UUUU is high-beta — wider PTs)
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Confidence sweeps
    for mc in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    # Max loss sweeps
    for ml in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
        configs.append((f"Baseline + ML {ml}%",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

    # Cooldown sweeps
    for cb in [3, 5, 7, 10, 14, 20]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR stops (critical for volatile uranium/RE stock)
    for atr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        configs.append((f"Baseline + ATR {atr}x",
            baseline_rules,
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Composites
    configs.append(("Core + composites + exits",
        baseline_rules + COMPOSITE_RULES + EXIT_RULES,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Combined best guesses for high-beta uranium
    configs.append(("Core + trend + PT 0.15 + ML 8% + ATR 2.5",
        baseline_rules + ["trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 7,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    configs.append(("Core + trend + PT 0.20 + ML 10% + ATR 3.0 (let winners run)",
        baseline_rules + ["trend_alignment", "golden_cross"] + EXIT_RULES,
        {"profit_target": 0.20, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 10.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 3.0, "cash": 1000}
    ))

    configs.append(("Mining + trend + PT 0.12 + ML 6% + ATR 2.0",
        ["commodity_breakout", "volume_breakout", "seasonality", "trend_continuation",
         "trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 6.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("Tight: core + PT 0.06 + ML 3% (scalp approach)",
        baseline_rules + EXIT_RULES,
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 3.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # CCJ-style (worked for another uranium stock)
    configs.append(("CCJ-style: trend_cont + seasonality + death_cross",
        ["trend_continuation", "seasonality", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 3%",
        baseline_rules,
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 3.0, "max_position_pct": 20.0, "cash": 1000}
    ))

    configs.append(("Core + trend + risk-based 5% + ATR 2.5",
        baseline_rules + ["trend_alignment"] + EXIT_RULES,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 7,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    return configs


# =============================================================================
# REFINEMENT
# =============================================================================

ALL_ENTRY_RULES = [
    "buy_dip_in_uptrend", "strong_buy_signal", "enhanced_buy_dip", "momentum_reversal",
    "trend_continuation", "rsi_oversold", "rsi_macd_confluence", "dip_recovery",
    "golden_cross", "trend_alignment", "commodity_breakout", "miner_metal_ratio",
    "dollar_weakness", "volume_breakout", "macd_bullish_crossover",
]
ALL_EXIT_RULES = ["rsi_overbought", "macd_bearish_crossover", "trend_break_warning", "death_cross"]


def generate_refinements(top_results: List[BacktestResult], round_num: int) -> list:
    """Generate new configs by mutating the best results from previous round."""
    configs = []

    for rank, base in enumerate(top_results[:3], 1):
        p = base.params.copy()
        r = list(base.rules)

        # Param tweaks
        for pt_delta in [-0.02, -0.01, 0.01, 0.02, 0.03]:
            new_pt = round(p.get("profit_target", 0.07) + pt_delta, 2)
            if 0.03 <= new_pt <= 0.30:
                np_ = {**p, "profit_target": new_pt}
                configs.append((f"R{round_num} #{rank} PT {new_pt}", r, np_))

        for mc_delta in [-0.05, 0.05, 0.10]:
            new_mc = round(p.get("min_confidence", 0.6) + mc_delta, 2)
            if 0.35 <= new_mc <= 0.80:
                np_ = {**p, "min_confidence": new_mc}
                configs.append((f"R{round_num} #{rank} MC {new_mc}", r, np_))

        for ml_delta in [-2, -1, 1, 2]:
            new_ml = p.get("max_loss", 5.0) + ml_delta
            if 3 <= new_ml <= 20:
                np_ = {**p, "max_loss": new_ml}
                configs.append((f"R{round_num} #{rank} ML {new_ml}", r, np_))

        for cb_delta in [-2, -1, 1, 2, 3]:
            new_cb = int(p.get("cooldown_bars", 5) + cb_delta)
            if 1 <= new_cb <= 20:
                np_ = {**p, "cooldown_bars": new_cb}
                configs.append((f"R{round_num} #{rank} CB {new_cb}", r, np_))

        # ATR
        if p.get("stop_mode") != "atr":
            for atr in [1.5, 2.0, 2.5]:
                np_ = {**p, "stop_mode": "atr", "atr_multiplier": atr}
                configs.append((f"R{round_num} #{rank} +ATR {atr}x", r, np_))
        else:
            cur_atr = p.get("atr_multiplier", 2.0)
            for atr_d in [-0.5, -0.25, 0.25, 0.5]:
                new_atr = round(cur_atr + atr_d, 2)
                if 1.0 <= new_atr <= 4.5:
                    np_ = {**p, "atr_multiplier": new_atr}
                    configs.append((f"R{round_num} #{rank} ATR {new_atr}x", r, np_))

        # Rule mutations
        for _ in range(3):
            candidate = random.choice(ALL_ENTRY_RULES)
            if candidate not in r:
                configs.append((f"R{round_num} #{rank} +{candidate}", r + [candidate], p))

        for _ in range(2):
            candidate = random.choice(ALL_EXIT_RULES)
            if candidate not in r:
                configs.append((f"R{round_num} #{rank} +{candidate}", r + [candidate], p))

        entry_rules_in = [rule for rule in r if rule in ALL_ENTRY_RULES]
        if len(entry_rules_in) > 2:
            for _ in range(2):
                to_remove = random.choice(entry_rules_in)
                new_rules = [rule for rule in r if rule != to_remove]
                configs.append((f"R{round_num} #{rank} -{to_remove}", new_rules, p))

        if p.get("sizing_mode") != "risk_based":
            np_ = {**p, "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0}
            configs.append((f"R{round_num} #{rank} +risk_based", r, np_))

    return configs


# =============================================================================
# MARKDOWN WRITER
# =============================================================================

def format_params(params: Dict) -> str:
    skip = {"cash", "stop_loss"}
    parts = []
    for k, v in sorted(params.items()):
        if k not in skip:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def write_md(symbol: str, results: List[BacktestResult], baseline: Optional[BacktestResult]):
    """Write comprehensive markdown file with top 5 results."""
    results.sort(key=lambda r: (r.sharpe, r.profit_factor), reverse=True)
    top5 = results[:5]

    filepath = os.path.join(os.path.expanduser("~/Projects/backtesting-service"), f"{symbol.lower()}-rules.md")

    with open(filepath, "w") as f:
        f.write(f"# {symbol} — Optimization Results\n\n")
        f.write(f"**Date:** 2026-02-23\n")
        f.write(f"**Period:** 2021-01-01 to 2026-02-22 (5 years)\n")
        f.write(f"**Mode:** Daily entries + 5-min exits (multi-timeframe)\n")
        f.write(f"**Starting Capital:** $1,000\n")
        f.write(f"**Configurations Tested:** {len(results)}\n\n")

        # Research context
        if symbol in SYMBOL_CONTEXT:
            f.write(f"## Market Context\n\n")
            f.write(SYMBOL_CONTEXT[symbol] + "\n\n")

        f.write("---\n\n")

        # Baseline
        if baseline:
            f.write(f"## Current Baseline (rules.yaml)\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| **Rules** | {', '.join(baseline.rules)} |\n")
            f.write(f"| **Profit Target** | {baseline.params.get('profit_target', 'N/A')} |\n")
            f.write(f"| **Min Confidence** | {baseline.params.get('min_confidence', 'N/A')} |\n")
            f.write(f"| **Max Loss** | {baseline.params.get('max_loss', 'N/A')}% |\n")
            f.write(f"| **Cooldown Bars** | {baseline.params.get('cooldown_bars', 'N/A')} |\n")
            f.write(f"| **Total Return** | {baseline.total_return:+.1f}% |\n")
            f.write(f"| **Trades** | {baseline.trades} |\n")
            f.write(f"| **Win Rate** | {baseline.win_rate:.1f}% |\n")
            f.write(f"| **Profit Factor** | {baseline.profit_factor:.2f} |\n")
            f.write(f"| **Sharpe Ratio** | {baseline.sharpe:.2f} |\n")
            f.write(f"| **Max Drawdown** | -{baseline.max_drawdown:.1f}% |\n\n")
            f.write("---\n\n")

        # Top 5
        f.write("## Top 5 Optimized Configurations\n\n")

        for i, r in enumerate(top5, 1):
            medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(i, f"{i}th")
            f.write(f"### #{i} — {r.label}\n\n")

            f.write(f"| Metric | Value |")
            if baseline:
                f.write(f" vs Baseline |")
            f.write(f"\n|--------|-------|")
            if baseline:
                f.write(f"-------------|")
            f.write(f"\n")

            f.write(f"| **Total Return** | {r.total_return:+.1f}% |")
            if baseline:
                delta = r.total_return - baseline.total_return
                f.write(f" {delta:+.1f}% |")
            f.write(f"\n")

            f.write(f"| **Trades** | {r.trades} |")
            if baseline:
                f.write(f" {r.trades - baseline.trades:+d} |")
            f.write(f"\n")

            f.write(f"| **Win Rate** | {r.win_rate:.1f}% |")
            if baseline:
                f.write(f" {r.win_rate - baseline.win_rate:+.1f}% |")
            f.write(f"\n")

            f.write(f"| **Profit Factor** | {r.profit_factor:.2f} |")
            if baseline:
                f.write(f" {r.profit_factor - baseline.profit_factor:+.2f} |")
            f.write(f"\n")

            f.write(f"| **Sharpe Ratio** | {r.sharpe:.2f} |")
            if baseline:
                f.write(f" {r.sharpe - baseline.sharpe:+.2f} |")
            f.write(f"\n")

            f.write(f"| **Max Drawdown** | -{r.max_drawdown:.1f}% |")
            if baseline:
                dd_diff = baseline.max_drawdown - r.max_drawdown
                better = "better" if dd_diff > 0 else "worse"
                f.write(f" {abs(dd_diff):.1f}% {better} |")
            f.write(f"\n\n")

            f.write(f"**Rules:** `{', '.join(r.rules)}`\n\n")
            f.write(f"**Parameters:**\n")
            f.write(f"- Profit Target: {r.params.get('profit_target', 'N/A')}\n")
            f.write(f"- Min Confidence: {r.params.get('min_confidence', 'N/A')}\n")
            f.write(f"- Max Loss: {r.params.get('max_loss', 'N/A')}%\n")
            f.write(f"- Cooldown Bars: {r.params.get('cooldown_bars', 'N/A')}\n")
            if r.params.get("stop_mode") == "atr":
                f.write(f"- Stop Mode: ATR (multiplier={r.params.get('atr_multiplier', 2.0)})\n")
            if r.params.get("sizing_mode") == "risk_based":
                f.write(f"- Sizing: Risk-based (risk_pct={r.params.get('risk_pct', 5.0)}%, max_pos={r.params.get('max_position_pct', 20.0)}%)\n")
            f.write(f"\n")

            f.write(f"**Backtest Command:**\n```bash\n{r.command}\n```\n\n")

            if i < 5:
                f.write("---\n\n")

        # Summary comparison table
        f.write("\n\n## Summary Comparison\n\n")
        f.write("| Rank | Label | Return | WR | PF | Sharpe | DD |\n")
        f.write("|------|-------|--------|----|----|--------|----|\n")
        if baseline:
            f.write(f"| BASE | {baseline.label[:40]} | {baseline.total_return:+.1f}% | {baseline.win_rate:.1f}% | {baseline.profit_factor:.2f} | {baseline.sharpe:.2f} | -{baseline.max_drawdown:.1f}% |\n")
        for i, r in enumerate(top5, 1):
            f.write(f"| #{i} | {r.label[:40]} | {r.total_return:+.1f}% | {r.win_rate:.1f}% | {r.profit_factor:.2f} | {r.sharpe:.2f} | -{r.max_drawdown:.1f}% |\n")
        f.write("\n")

    print(f"  Wrote {filepath}")


# =============================================================================
# MAIN
# =============================================================================

SYMBOL_CONFIGS = {
    "CCJ": get_ccj_configs,
    "CAT": get_cat_configs,
    "WPM": get_wpm_configs,
    "PPLT": get_pplt_configs,
    "URNM": get_urnm_configs,
    "SLV": get_slv_configs,
    "MP": get_mp_configs,
    "UUUU": get_uuuu_configs,
}


def main():
    parser = argparse.ArgumentParser(description="Full symbol optimization")
    parser.add_argument("symbol", help="Symbol to optimize")
    parser.add_argument("--rounds", type=int, default=4, help="Number of refinement rounds")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol: {symbol}. Available: {', '.join(SYMBOL_CONFIGS.keys())}")
        sys.exit(1)

    max_rounds = args.rounds
    all_results: List[BacktestResult] = []
    baseline: Optional[BacktestResult] = None

    print(f"\n{'='*60}")
    print(f"  OPTIMIZING {symbol}")
    print(f"  Max rounds: {max_rounds}")
    print(f"{'='*60}")

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  ROUND {round_num}/{max_rounds}")
        print(f"{'#'*60}")

        if round_num == 1:
            configs = SYMBOL_CONFIGS[symbol]()
        else:
            top = sorted(all_results, key=lambda r: (r.sharpe, r.profit_factor), reverse=True)[:5]
            if not top:
                print("  No results to refine")
                break
            configs = generate_refinements(top, round_num)

        if not configs:
            print("  No configs generated")
            break

        total = len(configs)
        print(f"  Running {total} configurations...\n")

        for i, (label, rules, params) in enumerate(configs, 1):
            print(f"  [{symbol}] {i}/{total}: {label}...", end=" ", flush=True)
            start = time.time()
            result = run_backtest(symbol, rules, params, label)
            elapsed = time.time() - start

            if result:
                all_results.append(result)
                if round_num == 1 and i == 1:
                    baseline = result
                print(f"Sharpe={result.sharpe:.2f} PF={result.profit_factor:.2f} "
                      f"WR={result.win_rate:.1f}% Ret={result.total_return:+.1f}% "
                      f"DD=-{result.max_drawdown:.1f}% ({elapsed:.1f}s)")
            else:
                print(f"FAILED ({elapsed:.1f}s)")

        # Write updated results after each round
        if all_results:
            write_md(symbol, all_results, baseline)
            best = max(all_results, key=lambda r: r.sharpe)
            print(f"\n  Round {round_num} done. Total tested: {len(all_results)}. "
                  f"Best Sharpe: {best.sharpe:.2f} ({best.label})")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  {symbol} OPTIMIZATION COMPLETE")
    print(f"  Total configurations tested: {len(all_results)}")
    if all_results:
        best = max(all_results, key=lambda r: r.sharpe)
        print(f"  Best: Sharpe={best.sharpe:.2f} PF={best.profit_factor:.2f} "
              f"WR={best.win_rate:.1f}% Ret={best.total_return:+.1f}% "
              f"DD=-{best.max_drawdown:.1f}%")
        print(f"  Config: {best.label}")
    print(f"  Output: {symbol.lower()}-rules.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
