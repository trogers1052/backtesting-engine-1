#!/usr/bin/env python3
"""
Multi-symbol strategy optimizer.
Runs systematic backtests across rule combinations and parameters,
captures the best 5 results per symbol to markdown files.
"""

import subprocess
import sys
import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
    raw_output: str = ""
    label: str = ""

import tempfile
import random

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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.expanduser("~/Projects/backtesting-service"),
            env=env,
        )
    except subprocess.TimeoutExpired:
        try: os.unlink(json_out)
        except: pass
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        try: os.unlink(json_out)
        except: pass
        return None

    # Parse JSON results
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
    br.trades = data.get("trades", {}).get("total", 0)
    br.win_rate = data.get("trades", {}).get("win_rate", 0) * 100
    br.profit_factor = data.get("risk_metrics", {}).get("profit_factor", 0)
    br.sharpe = data.get("risk_metrics", {}).get("sharpe_ratio", 0)
    br.max_drawdown = data.get("risk_metrics", {}).get("max_drawdown_pct", 0)

    if br.trades == 0:
        return None

    return br


def write_results(symbol: str, results: List[BacktestResult], baseline: Optional[BacktestResult]):
    """Write top 5 results to markdown file."""
    # Sort by Sharpe (primary), then PF (secondary)
    results.sort(key=lambda r: (r.sharpe, r.profit_factor), reverse=True)
    top5 = results[:5]

    filename = f"{symbol.lower()}-rules.md"
    filepath = os.path.join(os.path.expanduser("~/Projects/backtesting-service"), filename)

    with open(filepath, "w") as f:
        f.write(f"# {symbol} Optimization Results\n")
        f.write(f"## Date: 2026-02-22\n")
        f.write(f"## Total configurations tested: {len(results)}\n\n")

        if baseline:
            f.write(f"### Current Baseline\n")
            f.write(f"**Rules:** {', '.join(baseline.rules)}\n")
            f.write(f"**Parameters:** {_format_params(baseline.params)}\n")
            f.write(f"**Results:** Return={baseline.total_return:+.1f}%, Trades={baseline.trades}, ")
            f.write(f"WR={baseline.win_rate:.1f}%, PF={baseline.profit_factor:.2f}, ")
            f.write(f"Sharpe={baseline.sharpe:.2f}, DD=-{baseline.max_drawdown:.1f}%\n\n")
            f.write("---\n\n")

        for i, r in enumerate(top5, 1):
            f.write(f"### Result #{i}")
            if i == 1:
                f.write(" (Best)")
            f.write(f"\n")
            f.write(f"**Label:** {r.label}\n")
            f.write(f"**Rules:** {', '.join(r.rules)}\n")
            f.write(f"**Parameters:** {_format_params(r.params)}\n")

            risk_engine = []
            if r.params.get("stop_mode") == "atr":
                risk_engine.append(f"ATR stops (multiplier={r.params.get('atr_multiplier', 2.0)})")
            if r.params.get("sizing_mode") == "risk_based":
                risk_engine.append(f"Risk-based sizing (risk_pct={r.params.get('risk_pct', 5.0)}%, max_pos={r.params.get('max_position_pct', 20.0)}%)")
            f.write(f"**Risk Engine:** {', '.join(risk_engine) if risk_engine else 'No (fixed stops)'}\n")

            f.write(f"**Results:** Return={r.total_return:+.1f}%, Trades={r.trades}, ")
            f.write(f"WR={r.win_rate:.1f}%, PF={r.profit_factor:.2f}, ")
            f.write(f"Sharpe={r.sharpe:.2f}, DD=-{r.max_drawdown:.1f}%\n")

            # Comparison to baseline
            if baseline and baseline.sharpe > 0:
                sharpe_delta = r.sharpe - baseline.sharpe
                f.write(f"**vs Baseline:** Sharpe {sharpe_delta:+.2f}, ")
                f.write(f"DD {'better' if r.max_drawdown < baseline.max_drawdown else 'worse'} ")
                f.write(f"({-r.max_drawdown:.1f}% vs -{baseline.max_drawdown:.1f}%)\n")

            f.write(f"**Command:**\n```\n{r.command}\n```\n\n")

    print(f"  Wrote {filepath}")


def _format_params(params: Dict) -> str:
    skip = {"cash", "stop_loss"}
    parts = []
    for k, v in params.items():
        if k not in skip:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# =============================================================================
# SYMBOL CONFIGURATIONS
# =============================================================================

def get_pplt_configs():
    """PPLT: Platinum ETF — volatile commodity, trend follower with breakouts."""
    configs = []

    # Baseline
    configs.append(("Baseline (rules.yaml)",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 1: Tighter confidence
    configs.append(("Baseline + higher confidence 0.70",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.70, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 2: Wider profit target
    configs.append(("Baseline + PT 0.08",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("Baseline + PT 0.10",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 3: Wider max loss
    configs.append(("Baseline + ML 5%",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 5.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("Baseline + ML 6%",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 4: Different cooldown
    configs.append(("Baseline + cooldown 5",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Baseline + cooldown 7",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 7, "cash": 1000}
    ))

    # Variation 5: ATR stops
    configs.append(("Baseline + ATR stops 2.0x",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("Baseline + ATR stops 2.5x",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    # Variation 6: Risk-based sizing
    configs.append(("Baseline + risk-based sizing 5%",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 30.0, "cash": 1000}
    ))

    # Variation 7: Add mining-specific rules
    configs.append(("Baseline + commodity_breakout + volume_breakout",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross", "commodity_breakout", "volume_breakout"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("Baseline + all mining rules",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross",
         "commodity_breakout", "volume_breakout", "miner_metal_ratio", "dollar_weakness", "seasonality"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 8: Minimal rule sets
    configs.append(("Minimal: golden_cross + death_cross only",
        ["golden_cross", "trend_alignment", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    configs.append(("Minimal: trend_continuation + trend_break",
        ["trend_continuation", "trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 9: Composite rules
    configs.append(("Baseline + rsi_macd_confluence + dip_recovery",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross", "rsi_macd_confluence", "dip_recovery"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 10: Lower confidence for more trades
    configs.append(("Baseline + confidence 0.50",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 11: Combined best params guesses
    configs.append(("PT 0.08 + ML 5% + cooldown 5",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("PT 0.08 + ML 5% + ATR 2.0",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 5.0, "cooldown_bars": 3,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    configs.append(("Mining rules + ATR 2.0 + PT 0.08",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross", "commodity_breakout", "volume_breakout", "seasonality"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    # Variation 12: PT 0.12 (let winners run)
    configs.append(("PT 0.12 + ML 6% (let winners run)",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 6.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 13: strong_buy_signal instead of enhanced_buy_dip
    configs.append(("strong_buy + momentum + trend rules",
        ["strong_buy_signal", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 14: buy_dip_in_uptrend (original rule)
    configs.append(("Original buy_dip_in_uptrend + exits",
        ["buy_dip_in_uptrend", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Variation 15: Confidence 0.60
    configs.append(("Baseline + confidence 0.60",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    return configs


def get_cat_configs():
    """CAT: Caterpillar — strong trend follower, industrial blue-chip."""
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

    # Parameter sweeps
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15]:
        configs.append((f"Baseline + PT {pt}",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    for mc in [0.40, 0.55, 0.60, 0.65, 0.70]:
        configs.append((f"Baseline + confidence {mc}",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
        ))

    for ml in [4.0, 6.0, 8.0, 10.0]:
        configs.append((f"Baseline + max_loss {ml}%",
            ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
             "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
            {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 5, "cash": 1000}
        ))

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

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    # Trend-heavy approach
    configs.append(("Trend-heavy: trend rules + exits",
        ["trend_continuation", "trend_alignment", "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Minimal
    configs.append(("Minimal: enhanced_buy_dip + trend_alignment + death_cross",
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

    # Combined best guess
    configs.append(("Golden+death + PT 0.08 + ATR 2.0",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
         "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
         "golden_cross", "death_cross"],
        {"profit_target": 0.08, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    return configs


def get_wpm_configs():
    """WPM: Wheaton Precious Metals — streaming company, gold/silver correlated."""
    configs = []

    # Baseline (13 rules — kitchen sink)
    baseline_rules = [
        "enhanced_buy_dip", "momentum_reversal", "trend_continuation", "rsi_oversold",
        "rsi_overbought", "macd_bearish_crossover", "trend_alignment", "trend_break_warning",
        "commodity_breakout", "miner_metal_ratio", "dollar_weakness", "seasonality", "volume_breakout"
    ]

    configs.append(("Baseline (rules.yaml — 13 rules)",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Parameter sweeps on baseline
    for pt in [0.06, 0.08, 0.10, 0.15]:
        configs.append((f"Baseline + PT {pt}",
            baseline_rules,
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
        ))

    for mc in [0.40, 0.55, 0.60, 0.65]:
        configs.append((f"Baseline + confidence {mc}",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
        ))

    for ml in [4.0, 5.0, 8.0, 10.0]:
        configs.append((f"Baseline + max_loss {ml}%",
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

    # Reduced rule sets — fewer but higher quality
    configs.append(("Reduced: core 3 + exits",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation",
         "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Reduced: mining focus",
        ["commodity_breakout", "miner_metal_ratio", "volume_breakout", "dollar_weakness", "seasonality",
         "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    configs.append(("Reduced: trend + mining",
        ["enhanced_buy_dip", "trend_continuation", "trend_alignment", "golden_cross",
         "commodity_breakout", "volume_breakout", "seasonality",
         "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Add golden_cross + death_cross to baseline
    configs.append(("Baseline + golden_cross + death_cross",
        baseline_rules + ["golden_cross", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # PPLT-style (worked well for another precious metals play)
    configs.append(("PPLT-style rules",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.06, "stop_loss": 1.0, "min_confidence": 0.65, "max_loss": 4.0, "cooldown_bars": 3, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Baseline + risk-based 5%",
        baseline_rules,
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    # Composites
    configs.append(("Baseline + rsi_macd_confluence + dip_recovery",
        baseline_rules + ["rsi_macd_confluence", "dip_recovery"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Combined best guess
    configs.append(("Full rules + ATR 2.0 + PT 0.10",
        baseline_rules + ["golden_cross", "death_cross"],
        {"profit_target": 0.10, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 5.0, "cooldown_bars": 5,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    # Cooldown sweep
    for cb in [3, 7, 10]:
        configs.append((f"Baseline + cooldown {cb}",
            baseline_rules,
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 6.0, "cooldown_bars": cb, "cash": 1000}
        ))

    return configs


def get_fcx_configs():
    """FCX: Freeport-McMoRan — high-beta copper miner. Needs different approach."""
    configs = []

    # Old baseline (poor results)
    configs.append(("Old baseline (poor 0.18 Sharpe)",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation"],
        {"profit_target": 0.07, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 5.0, "cooldown_bars": 5, "cash": 1000}
    ))

    # Mean reversion approach
    configs.append(("Mean reversion: miner_metal_ratio + rsi_oversold",
        ["miner_metal_ratio", "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10, "cash": 1000}
    ))

    configs.append(("Mean reversion + ATR 2.5x",
        ["miner_metal_ratio", "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    configs.append(("Mean reversion + ATR 3.0x",
        ["miner_metal_ratio", "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 10.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 3.0, "cash": 1000}
    ))

    # Commodity breakout
    configs.append(("Commodity breakout + volume",
        ["commodity_breakout", "volume_breakout", "trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 10, "cash": 1000}
    ))

    configs.append(("Commodity breakout + ATR 2.5x",
        ["commodity_breakout", "volume_breakout", "trend_alignment", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    # Mining full suite
    configs.append(("Full mining rules",
        ["commodity_breakout", "miner_metal_ratio", "dollar_weakness", "volume_breakout", "seasonality",
         "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10, "cash": 1000}
    ))

    configs.append(("Full mining + ATR 2.0x",
        ["commodity_breakout", "miner_metal_ratio", "dollar_weakness", "volume_breakout", "seasonality",
         "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 2.0, "cash": 1000}
    ))

    # Wider profit targets for high-beta
    for pt in [0.10, 0.15, 0.20]:
        configs.append((f"Mining rules + PT {pt}",
            ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
             "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
            {"profit_target": pt, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10, "cash": 1000}
        ))

    # Max loss sweep
    for ml in [6.0, 8.0, 10.0, 12.0, 15.0]:
        configs.append((f"Mining rules + ML {ml}%",
            ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
             "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": ml, "cooldown_bars": 10, "cash": 1000}
        ))

    # Confidence sweep
    for mc in [0.40, 0.50, 0.60, 0.65, 0.70]:
        configs.append((f"Mining rules + confidence {mc}",
            ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
             "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": mc, "max_loss": 8.0, "cooldown_bars": 10, "cash": 1000}
        ))

    # Cooldown sweep
    for cb in [5, 7, 10, 15]:
        configs.append((f"Mining rules + cooldown {cb}",
            ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
             "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": cb, "cash": 1000}
        ))

    # ATR multiplier sweep
    for atr in [1.5, 2.0, 2.5, 3.0, 3.5]:
        configs.append((f"Mining rules + ATR {atr}x",
            ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
             "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
            {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10,
             "stop_mode": "atr", "atr_multiplier": atr, "cash": 1000}
        ))

    # Trend following with wider stops
    configs.append(("Trend follow + wide: PT 0.15, ML 10%, ATR 3.0",
        ["enhanced_buy_dip", "momentum_reversal", "trend_continuation", "trend_alignment",
         "golden_cross", "trend_break_warning", "death_cross"],
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.60, "max_loss": 10.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 3.0, "cash": 1000}
    ))

    # Momentum approach
    configs.append(("Momentum: rsi_macd_confluence + momentum_reversal",
        ["rsi_macd_confluence", "momentum_reversal", "dip_recovery",
         "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 8.0, "cooldown_bars": 10, "cash": 1000}
    ))

    # Risk-based sizing
    configs.append(("Mining rules + risk-based 3%",
        ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
         "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10,
         "sizing_mode": "risk_based", "risk_pct": 3.0, "max_position_pct": 20.0, "cash": 1000}
    ))

    configs.append(("Mining rules + risk-based 5%",
        ["commodity_breakout", "miner_metal_ratio", "volume_breakout",
         "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.12, "stop_loss": 1.0, "min_confidence": 0.50, "max_loss": 8.0, "cooldown_bars": 10,
         "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0, "cash": 1000}
    ))

    # Combined best guess
    configs.append(("Best guess: mining + ATR 2.5 + PT 0.15 + ML 10 + CD 10",
        ["commodity_breakout", "miner_metal_ratio", "volume_breakout", "dollar_weakness", "seasonality",
         "rsi_oversold", "rsi_overbought", "trend_break_warning", "death_cross"],
        {"profit_target": 0.15, "stop_loss": 1.0, "min_confidence": 0.55, "max_loss": 10.0, "cooldown_bars": 10,
         "stop_mode": "atr", "atr_multiplier": 2.5, "cash": 1000}
    ))

    return configs


# =============================================================================
# MAIN
# =============================================================================

def optimize_symbol(symbol: str, configs):
    """Run all configs for a symbol and return results."""
    results = []
    baseline = None
    total = len(configs)

    print(f"\n{'='*60}")
    print(f"  OPTIMIZING {symbol} — {total} configurations")
    print(f"{'='*60}")

    for i, (label, rules, params) in enumerate(configs, 1):
        print(f"  [{symbol}] {i}/{total}: {label}...", end=" ", flush=True)
        start = time.time()
        result = run_backtest(symbol, rules, params, label)
        elapsed = time.time() - start

        if result:
            results.append(result)
            if i == 1:
                baseline = result
            print(f"Sharpe={result.sharpe:.2f} PF={result.profit_factor:.2f} "
                  f"WR={result.win_rate:.1f}% DD=-{result.max_drawdown:.1f}% "
                  f"({elapsed:.1f}s)")
        else:
            print(f"FAILED ({elapsed:.1f}s)")

    return results, baseline


def generate_refinements(symbol: str, top_results: List[BacktestResult], round_num: int) -> list:
    """Generate new configs by mutating the best results from previous round."""
    configs = []

    ALL_ENTRY_RULES = [
        "buy_dip_in_uptrend", "strong_buy_signal", "enhanced_buy_dip", "momentum_reversal",
        "trend_continuation", "rsi_oversold", "rsi_macd_confluence", "dip_recovery",
        "golden_cross", "trend_alignment", "commodity_breakout", "miner_metal_ratio",
        "dollar_weakness", "volume_breakout", "macd_bullish_crossover", "macd_momentum",
    ]
    ALL_EXIT_RULES = ["rsi_overbought", "macd_bearish_crossover", "trend_break_warning", "death_cross"]

    for rank, base in enumerate(top_results[:3], 1):
        p = base.params.copy()
        r = list(base.rules)

        # Param tweaks around the best
        for pt_delta in [-0.02, -0.01, 0.01, 0.02, 0.03]:
            new_pt = round(p.get("profit_target", 0.07) + pt_delta, 2)
            if 0.03 <= new_pt <= 0.25:
                np = {**p, "profit_target": new_pt}
                configs.append((f"R{round_num} #{rank} PT {new_pt}", r, np))

        for mc_delta in [-0.05, 0.05, 0.10]:
            new_mc = round(p.get("min_confidence", 0.6) + mc_delta, 2)
            if 0.35 <= new_mc <= 0.80:
                np = {**p, "min_confidence": new_mc}
                configs.append((f"R{round_num} #{rank} MC {new_mc}", r, np))

        for ml_delta in [-2, -1, 1, 2]:
            new_ml = p.get("max_loss", 5.0) + ml_delta
            if 3 <= new_ml <= 20:
                np = {**p, "max_loss": new_ml}
                configs.append((f"R{round_num} #{rank} ML {new_ml}", r, np))

        for cb_delta in [-2, -1, 1, 2, 3]:
            new_cb = int(p.get("cooldown_bars", 5) + cb_delta)
            if 1 <= new_cb <= 20:
                np = {**p, "cooldown_bars": new_cb}
                configs.append((f"R{round_num} #{rank} CB {new_cb}", r, np))

        # ATR variations
        if p.get("stop_mode") != "atr":
            for atr in [1.5, 2.0, 2.5]:
                np = {**p, "stop_mode": "atr", "atr_multiplier": atr}
                configs.append((f"R{round_num} #{rank} +ATR {atr}x", r, np))
        else:
            cur_atr = p.get("atr_multiplier", 2.0)
            for atr_d in [-0.5, 0.5]:
                new_atr = round(cur_atr + atr_d, 1)
                if 1.0 <= new_atr <= 4.0:
                    np = {**p, "atr_multiplier": new_atr}
                    configs.append((f"R{round_num} #{rank} ATR {new_atr}x", r, np))

        # Rule mutations — add a random entry or exit rule
        for _ in range(3):
            candidate = random.choice(ALL_ENTRY_RULES)
            if candidate not in r:
                new_rules = r + [candidate]
                configs.append((f"R{round_num} #{rank} +{candidate}", new_rules, p))

        for _ in range(2):
            candidate = random.choice(ALL_EXIT_RULES)
            if candidate not in r:
                new_rules = r + [candidate]
                configs.append((f"R{round_num} #{rank} +{candidate}", new_rules, p))

        # Rule mutations — remove a random entry rule
        entry_rules_in = [rule for rule in r if rule in ALL_ENTRY_RULES]
        if len(entry_rules_in) > 2:
            for _ in range(2):
                to_remove = random.choice(entry_rules_in)
                new_rules = [rule for rule in r if rule != to_remove]
                configs.append((f"R{round_num} #{rank} -{to_remove}", new_rules, p))

        # Risk-based sizing toggle
        if p.get("sizing_mode") != "risk_based":
            np = {**p, "sizing_mode": "risk_based", "risk_pct": 5.0, "max_position_pct": 25.0}
            configs.append((f"R{round_num} #{rank} +risk_based", r, np))

    return configs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="*", default=["PPLT", "CAT", "WPM", "FCX"])
    parser.add_argument("--hours", type=float, default=4.0, help="Hours to run")
    args = parser.parse_args()

    selected = [s.upper() for s in args.symbols]
    run_hours = args.hours
    deadline = time.time() + run_hours * 3600

    symbols_configs = {
        "PPLT": get_pplt_configs,
        "CAT": get_cat_configs,
        "WPM": get_wpm_configs,
        "FCX": get_fcx_configs,
    }

    # Track all results per symbol across all rounds
    all_results: Dict[str, List[BacktestResult]] = {s: [] for s in selected}
    baselines: Dict[str, Optional[BacktestResult]] = {s: None for s in selected}

    print(f"{'='*60}")
    print(f"  4-HOUR OPTIMIZATION ENGINE")
    print(f"  Symbols: {', '.join(selected)}")
    print(f"  Runtime: {run_hours} hours (until {time.strftime('%H:%M', time.localtime(deadline))})")
    print(f"{'='*60}")

    round_num = 0

    while time.time() < deadline:
        round_num += 1
        elapsed_hrs = (time.time() - (deadline - run_hours * 3600)) / 3600
        remaining_hrs = (deadline - time.time()) / 3600

        print(f"\n{'#'*60}")
        print(f"  ROUND {round_num} | Elapsed: {elapsed_hrs:.1f}h | Remaining: {remaining_hrs:.1f}h")
        print(f"{'#'*60}")

        for symbol in selected:
            if time.time() >= deadline:
                break

            if round_num == 1:
                # First round: use predefined configs
                configs = symbols_configs[symbol]()
            else:
                # Subsequent rounds: refine around top results
                top = sorted(all_results[symbol], key=lambda r: (r.sharpe, r.profit_factor), reverse=True)[:5]
                if not top:
                    print(f"  [{symbol}] No results to refine, skipping")
                    continue
                configs = generate_refinements(symbol, top, round_num)

            if not configs:
                continue

            results, baseline = optimize_symbol(symbol, configs)
            all_results[symbol].extend(results)
            if baseline and baselines[symbol] is None:
                baselines[symbol] = baseline

            # Write updated results after each symbol
            if all_results[symbol]:
                write_results(symbol, all_results[symbol], baselines[symbol])
                best = max(all_results[symbol], key=lambda r: r.sharpe)
                total_tested = len(all_results[symbol])
                print(f"\n  [{symbol}] Round {round_num} done. "
                      f"Total tested: {total_tested}. "
                      f"Best Sharpe: {best.sharpe:.2f} ({best.label})")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE — {round_num} rounds")
    print(f"{'='*60}")
    for symbol in selected:
        if all_results[symbol]:
            best = max(all_results[symbol], key=lambda r: r.sharpe)
            total = len(all_results[symbol])
            print(f"  {symbol}: {total} configs tested | "
                  f"Best Sharpe={best.sharpe:.2f} PF={best.profit_factor:.2f} "
                  f"WR={best.win_rate:.1f}% DD=-{best.max_drawdown:.1f}% "
                  f"| {best.label}")
        else:
            print(f"  {symbol}: No valid results")
    print(f"\nResults written to: {', '.join(s.lower() + '-rules.md' for s in selected)}")
