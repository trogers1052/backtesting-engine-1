#!/usr/bin/env python3
"""
Generate Tier Rankings (S-F) from Backtesting Validation Results

Reads all *-validated.md files and produces a composite score per symbol.

Composite Score Formula (max 100):
  score = (gates/4 * 35) + (sharpe_norm * 20) + (return_norm * 15) + (wr_norm * 10) + (pf_norm * 10) + (trades_norm * 10)

Normalization (0-1, capped):
  sharpe_norm  = clamp(sharpe / 1.2, 0, 1)
  return_norm  = clamp(return_pct / 150, 0, 1)
  wr_norm      = clamp((win_rate - 30) / 40, 0, 1)
  pf_norm      = clamp((pf - 1.0) / 2.0, 0, 1)
  trades_norm  = clamp((trades - 30) / 70, 0, 1)  — 30 = CLT floor, 100+ = full score

Statistical basis for trade count (López de Prado, Bailey 2014):
  <30 trades:   Statistically unreliable (below CLT floor)
  30-50:        Bare minimum, limited confidence
  50-100:       Basic reliability
  100-200:      Solid confidence
  200+:         Institutional-grade (rarely achieved in 5yr daily signals)

Tier Cutoffs:
  S (85-100): Elite — full deployment, max conviction
  A (70-84):  Strong — deploy with standard sizing
  B (55-69):  Good — deploy with reduced sizing or regime restrictions
  C (40-54):  Average — watchlist, conditional use only
  D (25-39):  Weak — avoid or very restricted
  F (0-24):   Failed — blacklist

Usage:
    python generate_tier_rankings.py
"""

import os
import re
import glob
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class SymbolResult:
    symbol: str
    name: str = ""
    category: str = ""
    gates_passed: int = 0
    gates_total: int = 4
    sharpe: float = 0.0
    total_return: float = 0.0  # as percentage, e.g. 65.7
    win_rate: float = 0.0      # as percentage, e.g. 63.6
    profit_factor: float = 0.0
    max_drawdown: float = 0.0  # as percentage
    trades: int = 0
    best_config: str = ""
    wf_pass: bool = False
    bs_pass: bool = False
    mc_pass: bool = False
    regime_pass: bool = False
    blacklisted: bool = False
    zero_trades: bool = False
    score: float = 0.0
    tier: str = "F"


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def compute_score(r: SymbolResult) -> float:
    if r.blacklisted or r.zero_trades:
        return 0.0

    gates_score = (r.gates_passed / 4) * 35
    sharpe_norm = clamp(r.sharpe / 1.2) if r.sharpe > 0 else 0.0
    return_norm = clamp(r.total_return / 150) if r.total_return > 0 else 0.0
    wr_norm = clamp((r.win_rate - 30) / 40)
    pf_norm = clamp((r.profit_factor - 1.0) / 2.0) if r.profit_factor > 1.0 else 0.0

    # Trade count: 30 = CLT statistical floor (0.0), 100+ = full score (1.0)
    # Below 30 trades: penalty applied (negative norm reduces score)
    trades_norm = clamp((r.trades - 30) / 70, -0.5, 1.0)

    raw_score = (gates_score
                 + (sharpe_norm * 20)
                 + (return_norm * 15)
                 + (wr_norm * 10)
                 + (pf_norm * 10)
                 + (trades_norm * 10))

    return round(max(raw_score, 0.0), 1)


def assign_tier(score: float) -> str:
    if score >= 85:
        return "S"
    elif score >= 70:
        return "A"
    elif score >= 55:
        return "B"
    elif score >= 40:
        return "C"
    elif score >= 25:
        return "D"
    else:
        return "F"


BLACKLISTED_SYMBOLS = {"RXRX", "VKTX"}


def parse_validated_file(filepath: str) -> Optional[SymbolResult]:
    """Parse a *-validated.md file and extract key metrics."""
    filename = os.path.basename(filepath)
    symbol = filename.replace("-validated.md", "").upper()
    if symbol in ("BRK.B",):
        symbol = "BRK.B"
    elif "." in filename.replace("-validated.md", ""):
        symbol = filename.replace("-validated.md", "").upper()

    r = SymbolResult(symbol=symbol)

    with open(filepath, "r") as f:
        content = f.read()

    if not content.strip():
        r.zero_trades = True
        return r

    if symbol in BLACKLISTED_SYMBOLS:
        r.blacklisted = True
        # Try to extract name
        title_match = re.search(r"^# .+?\((.+?)\)", content, re.MULTILINE)
        if title_match:
            r.name = title_match.group(1)
        return r

    # Extract name from title: "# SYMBOL (Name) Validated..."
    title_match = re.search(r"^# .+?\((.+?)\)", content, re.MULTILINE)
    if title_match:
        r.name = title_match.group(1)

    # Extract category
    cat_match = re.search(r"\*\*Category:\*\*\s*(.+)", content)
    if cat_match:
        r.category = cat_match.group(1).strip()

    # Find the Summary Table — section numbering varies (4 or 5)
    # Format: | Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
    summary_section = re.search(r"## \d+\. Summary Table(.*?)(?=## \d+\.|$)", content, re.DOTALL)

    best_row = None
    best_gates = -1
    best_sharpe = -999
    # Column indices — detect from header row
    col_sharpe = 5
    col_return = 6
    col_trades = 7

    if summary_section:
        table_text = summary_section.group(1)
        # Parse table rows
        rows = re.findall(r"\|(.+)\|", table_text)

        # Detect column layout from header row
        for row in rows:
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c]
            if cells and cells[0].strip().startswith("Config"):
                # Find column indices by header name
                for idx, cell in enumerate(cells):
                    header = cell.strip().lower()
                    if header == "sharpe":
                        col_sharpe = idx
                    elif header == "return":
                        col_return = idx
                    elif header == "trades":
                        col_trades = idx
                break

        for row in rows:
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c]
            if len(cells) < 8:
                continue
            # Skip header/separator rows
            if cells[0].startswith("Config") or cells[0].startswith("---"):
                continue
            if "---" in cells[1]:
                continue

            # Count gates passed
            gates = 0
            wf = "PASS" in cells[1]
            bs = "PASS" in cells[2]
            mc = "PASS" in cells[3]
            regime = "PASS" in cells[4]
            gates = sum([wf, bs, mc, regime])

            # Extract sharpe using detected column index
            sharpe_str = cells[col_sharpe].replace("*", "").strip() if col_sharpe < len(cells) else "0"
            try:
                sharpe = float(sharpe_str)
            except (ValueError, IndexError):
                sharpe = 0.0

            # Pick best row by gates, then sharpe
            if gates > best_gates or (gates == best_gates and sharpe > best_sharpe):
                best_gates = gates
                best_sharpe = sharpe
                best_row = cells
                r.wf_pass = wf
                r.bs_pass = bs
                r.mc_pass = mc
                r.regime_pass = regime

    if best_row and len(best_row) >= 8:
        r.gates_passed = best_gates
        r.best_config = best_row[0].replace("*", "").strip()

        try:
            r.sharpe = float(best_row[col_sharpe].replace("*", "").strip())
        except (ValueError, IndexError):
            pass

        return_str = best_row[col_return].replace("*", "").replace("%", "").replace("+", "").strip() if col_return < len(best_row) else "0"
        try:
            r.total_return = float(return_str)
        except ValueError:
            pass

        try:
            r.trades = int(best_row[col_trades].replace("*", "").strip()) if col_trades < len(best_row) else 0
        except (ValueError, IndexError):
            pass
    else:
        # Fallback: try to find gates from "Verdict: X/4" or "Result: X/4"
        verdict_match = re.search(r"(?:Verdict|Result):\s*(\d)/4", content)
        if verdict_match:
            r.gates_passed = int(verdict_match.group(1))

    # If no trades found in summary, check final recommendation
    if r.trades == 0:
        trades_match = re.search(r"Trades[=:]\s*(\d+)", content)
        if trades_match:
            r.trades = int(trades_match.group(1))

    # Extract win rate from best config section or final recommendation
    # Look in the validation detail sections
    if r.best_config:
        # Find the section for the best config
        config_pattern = re.escape(r.best_config)
        section = re.search(
            rf"### {config_pattern}.*?(?=###|## \d|$)",
            content, re.DOTALL
        )
        if section:
            perf_match = re.search(
                r"WR=(\d+\.?\d*)%.*?PF=(\d+\.?\d*).*?DD=-?(\d+\.?\d*)%",
                section.group(0)
            )
            if perf_match:
                r.win_rate = float(perf_match.group(1))
                r.profit_factor = float(perf_match.group(2))
                r.max_drawdown = float(perf_match.group(3))

    # Fallback: find win rate, PF, DD anywhere near the best config mention
    if r.win_rate == 0:
        perf_match = re.search(
            r"WR=(\d+\.?\d*)%",
            content
        )
        if perf_match:
            r.win_rate = float(perf_match.group(1))

    if r.profit_factor == 0:
        pf_match = re.search(r"PF=(\d+\.?\d*)", content)
        if pf_match:
            r.profit_factor = float(pf_match.group(1))

    if r.max_drawdown == 0:
        dd_match = re.search(r"DD=-?(\d+\.?\d*)%", content)
        if dd_match:
            r.max_drawdown = float(dd_match.group(1))

    # Check for zero trades
    if r.trades == 0:
        r.zero_trades = True

    return r


def get_sector_group(r: SymbolResult) -> str:
    """Map a symbol to a broad sector group."""
    cat = (r.category or "").lower()
    sym = r.symbol

    # Mining / Materials
    mining = {"AEM", "CCJ", "COPX", "FCX", "GDX", "GLD", "GUSH", "HL", "IAUM",
              "MP", "PAAS", "PPLT", "REMX", "SIL", "SLV", "URA", "URNM", "UUUU",
              "USAR", "WPM", "B"}
    if sym in mining or "mining" in cat or "gold" in cat or "uranium" in cat or "rare earth" in cat:
        return "Mining & Materials"

    # Energy
    energy = {"CNQ", "COP", "CVX", "EOG", "EPD", "EQNR", "ET", "FET", "OXY",
              "XLE", "XOM", "CEG", "VST", "OKLO"}
    if sym in energy or "energy" in cat or "oil" in cat or "midstream" in cat:
        return "Energy"

    # Tech / Growth
    tech = {"AAPL", "ALAB", "AMAT", "AMD", "AMZN", "APH", "CRM", "CRWD", "GOOGL",
            "LRCX", "MELI", "MU", "NET", "NOW", "PANW", "S", "SDGR", "SNDK", "TEM",
            "XLK", "QQQ"}
    if sym in tech or "tech" in cat or "semi" in cat or "saas" in cat or "cyber" in cat:
        return "Tech & Growth"

    # Financials
    financials = {"BRK.B", "CB", "GS", "JPM", "MA", "MCO", "NU", "PGR", "SOFI",
                  "SPGI", "V", "XLF"}
    if sym in financials or "financial" in cat or "bank" in cat or "insurance" in cat or "payment" in cat:
        return "Financials"

    # Healthcare
    healthcare = {"ABBV", "JNJ", "RXRX", "SYK", "TMDX", "UNH", "VKTX", "VRTX", "XLV"}
    if sym in healthcare or "healthcare" in cat or "pharma" in cat or "biotech" in cat or "medical" in cat:
        return "Healthcare"

    # Defense
    defense = {"AVAV", "ITA", "LMT", "RTX"}
    if sym in defense or "defense" in cat or "aerospace" in cat:
        return "Defense"

    # Industrials
    industrials = {"CAT", "ETN", "WLDN", "XLI"}
    if sym in industrials or "industrial" in cat:
        return "Industrials"

    # Utilities / Power
    utilities = {"AWK", "BEP", "CEG", "CWEN", "D", "NEE", "O", "SO", "VPU", "VST",
                 "XLU"}
    if sym in utilities or "utility" in cat or "power" in cat or "yieldco" in cat:
        return "Utilities & Power"

    # Consumer
    consumer = {"CHD", "CL", "COST", "KMB", "KO", "PG", "WMT", "XLP", "XLY", "AMT"}
    if sym in consumer or "consumer" in cat or "retail" in cat or "household" in cat:
        return "Consumer"

    # International
    international = {"EWY"}
    if sym in international:
        return "International"

    return "Other"


def write_report(results: List[SymbolResult], filepath: str):
    """Write the tier rankings report."""
    results.sort(key=lambda r: r.score, reverse=True)

    tier_counts = {}
    for r in results:
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    with open(filepath, "w") as f:
        f.write(f"# Backtesting Tier Rankings\n\n")
        f.write(f"**Generated:** {date.today()}\n")
        f.write(f"**Symbols:** {len(results)}\n")
        f.write(f"**Scoring:** Composite (gates 35% + Sharpe 20% + return 15% + trades 10% + win rate 10% + profit factor 10%)\n\n")

        # Tier summary
        f.write("## Tier Summary\n\n")
        f.write("| Tier | Range | Count | Description |\n")
        f.write("|------|-------|-------|-------------|\n")
        for tier, desc in [("S", "Elite — full deployment, max conviction"),
                           ("A", "Strong — deploy with standard sizing"),
                           ("B", "Good — deploy with reduced sizing or regime restrictions"),
                           ("C", "Average — watchlist, conditional use only"),
                           ("D", "Weak — avoid or very restricted"),
                           ("F", "Failed — blacklist")]:
            ranges = {"S": "85-100", "A": "70-84", "B": "55-69", "C": "40-54", "D": "25-39", "F": "0-24"}
            count = tier_counts.get(tier, 0)
            f.write(f"| **{tier}** | {ranges[tier]} | {count} | {desc} |\n")
        f.write("\n---\n\n")

        # Full ranked table
        f.write("## Full Rankings\n\n")
        f.write("| Rank | Tier | Score | Symbol | Name | Gates | Sharpe | Return | WR | PF | Trades | Max DD | Category |\n")
        f.write("|------|------|-------|--------|------|-------|--------|--------|-----|-----|--------|--------|----------|\n")
        for i, r in enumerate(results, 1):
            gates_str = f"{r.gates_passed}/4"
            if r.blacklisted:
                gates_str = "BL"
            elif r.zero_trades:
                gates_str = "0T"

            name = r.name[:20] if r.name else ""
            cat = r.category[:25] if r.category else ""

            tier_bold = f"**{r.tier}**" if r.tier in ("S", "A") else r.tier

            ret_str = f"+{r.total_return:.1f}%" if r.total_return > 0 else f"{r.total_return:.1f}%"
            if r.zero_trades or r.blacklisted:
                ret_str = "-"

            f.write(f"| {i} | {tier_bold} | {r.score:.1f} | **{r.symbol}** | {name} | "
                    f"{gates_str} | {r.sharpe:.2f} | {ret_str} | {r.win_rate:.0f}% | "
                    f"{r.profit_factor:.2f} | {r.trades} | -{r.max_drawdown:.0f}% | {cat} |\n")

        f.write("\n---\n\n")

        # Per-tier breakdown
        for tier in ["S", "A", "B", "C", "D", "F"]:
            tier_results = [r for r in results if r.tier == tier]
            if not tier_results:
                continue
            tier_names = {"S": "S Tier — Elite", "A": "A Tier — Strong", "B": "B Tier — Good",
                          "C": "C Tier — Average", "D": "D Tier — Weak", "F": "F Tier — Failed"}
            f.write(f"## {tier_names[tier]} ({len(tier_results)} symbols)\n\n")
            for r in tier_results:
                gates_detail = []
                if r.wf_pass: gates_detail.append("WF")
                if r.bs_pass: gates_detail.append("BS")
                if r.mc_pass: gates_detail.append("MC")
                if r.regime_pass: gates_detail.append("Regime")
                gates_str = ", ".join(gates_detail) if gates_detail else "none"

                if r.blacklisted:
                    f.write(f"- **{r.symbol}** ({r.name}) — BLACKLISTED (binary risk)\n")
                elif r.zero_trades:
                    f.write(f"- **{r.symbol}** ({r.name}) — Zero trades generated\n")
                else:
                    f.write(f"- **{r.symbol}** ({r.name}): score={r.score:.1f}, "
                            f"{r.gates_passed}/4 gates [{gates_str}], "
                            f"Sharpe={r.sharpe:.2f}, return={r.total_return:+.1f}%, "
                            f"WR={r.win_rate:.0f}%, {r.trades} trades\n")
            f.write("\n---\n\n")

        # Per-sector breakdown
        f.write("## Sector Breakdown\n\n")
        sector_groups = {}
        for r in results:
            sector = get_sector_group(r)
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(r)

        for sector in sorted(sector_groups.keys()):
            group = sector_groups[sector]
            group.sort(key=lambda r: r.score, reverse=True)
            avg_score = sum(r.score for r in group) / len(group) if group else 0
            f.write(f"### {sector} (avg score: {avg_score:.0f})\n\n")
            f.write("| Symbol | Tier | Score | Gates | Sharpe | Return |\n")
            f.write("|--------|------|-------|-------|--------|--------|\n")
            for r in group:
                ret_str = f"+{r.total_return:.1f}%" if r.total_return > 0 else f"{r.total_return:.1f}%"
                if r.zero_trades or r.blacklisted:
                    ret_str = "-"
                f.write(f"| {r.symbol} | {r.tier} | {r.score:.0f} | {r.gates_passed}/4 | {r.sharpe:.2f} | {ret_str} |\n")
            f.write("\n")

    print(f"Report written to: {filepath}")


def main():
    base_dir = os.path.expanduser("~/Projects/backtesting-service")
    pattern = os.path.join(base_dir, "*-validated.md")
    files = sorted(glob.glob(pattern))

    print(f"Found {len(files)} validation reports")

    results = []
    for fpath in files:
        r = parse_validated_file(fpath)
        if r:
            r.score = compute_score(r)
            r.tier = assign_tier(r.score)
            results.append(r)

    print(f"Parsed {len(results)} symbols")

    # Print quick summary to console
    tier_counts = {}
    for r in results:
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    print(f"\nTier Distribution:")
    for tier in ["S", "A", "B", "C", "D", "F"]:
        count = tier_counts.get(tier, 0)
        symbols = [r.symbol for r in sorted(
            [x for x in results if x.tier == tier],
            key=lambda x: x.score, reverse=True
        )]
        print(f"  {tier}: {count:2d} symbols — {', '.join(symbols[:10])}" +
              (f" + {len(symbols)-10} more" if len(symbols) > 10 else ""))

    output_path = os.path.join(base_dir, "tier-rankings.md")
    write_report(results, output_path)


if __name__ == "__main__":
    main()
