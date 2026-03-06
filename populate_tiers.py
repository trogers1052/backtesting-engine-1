#!/usr/bin/env python3
"""
Populate Tier Rankings into Stock-Service

Reads all *-validated.md files, computes composite scores and tiers,
then sends the data to stock-service via PUT /api/v1/tiers/bulk.

This populates both PostgreSQL and Redis (stock-service handles both).

Usage:
    python populate_tiers.py
    python populate_tiers.py --dry-run
    python populate_tiers.py --stock-service-url http://pi.local:8080
"""

import argparse
import json
import os
import glob
from datetime import date

import requests

from generate_tier_rankings import (
    parse_validated_file,
    compute_score,
    assign_tier,
)

# Tier → confidence multiplier mapping
CONFIDENCE_MULTIPLIERS = {
    "S": 1.15,
    "A": 1.05,
    "B": 1.00,
    "C": 0.85,
    "D": 0.65,
    "F": 0.00,
}

# Tier → position size multiplier mapping
POSITION_SIZE_MULTIPLIERS = {
    "S": 1.00,
    "A": 1.00,
    "B": 0.75,
    "C": 0.50,
    "D": 0.25,
    "F": 0.00,
}

# Symbols with special allowed_regimes overrides from validated reports
REGIME_OVERRIDES = {
    "SLV": ["BULL", "CHOP"],
    "UUUU": ["BULL", "CHOP"],
}


def build_tier_payload(r) -> dict:
    """Convert a SymbolResult into the stock-service tier API payload."""
    # Determine allowed_regimes
    allowed_regimes = None
    if r.symbol in REGIME_OVERRIDES:
        allowed_regimes = REGIME_OVERRIDES[r.symbol]
    elif not r.regime_pass and not r.blacklisted and not r.zero_trades:
        # Failed regime gate → restrict to bull-only
        allowed_regimes = ["BULL"]
    # regime_pass == True or blacklisted/zero_trades → None (unrestricted or suppressed)

    payload = {
        "symbol": r.symbol,
        "tier": r.tier,
        "composite_score": r.score,
        "gates_passed": r.gates_passed,
        "gates_total": r.gates_total,
        "regime_pass": r.regime_pass,
        "sharpe": r.sharpe if r.sharpe != 0 else None,
        "total_return": r.total_return if r.total_return != 0 else None,
        "win_rate": r.win_rate if r.win_rate != 0 else None,
        "profit_factor": r.profit_factor if r.profit_factor != 0 else None,
        "max_drawdown": r.max_drawdown if r.max_drawdown != 0 else None,
        "trade_count": r.trades,
        "confidence_multiplier": CONFIDENCE_MULTIPLIERS[r.tier],
        "position_size_multiplier": POSITION_SIZE_MULTIPLIERS[r.tier],
        "blacklisted": r.blacklisted or r.zero_trades,
        "ranking_date": date.today().isoformat(),
    }

    if allowed_regimes is not None:
        payload["allowed_regimes"] = allowed_regimes

    # Build notes
    notes_parts = []
    if r.blacklisted:
        notes_parts.append("BLACKLISTED: binary risk")
    if r.zero_trades:
        notes_parts.append("Zero trades generated")
    if r.best_config:
        notes_parts.append(f"Best config: {r.best_config}")
    if not r.regime_pass and not r.blacklisted and not r.zero_trades:
        regimes = allowed_regimes or []
        notes_parts.append(f"Regime-conditional: {','.join(regimes)}-only")
    if notes_parts:
        payload["notes"] = "; ".join(notes_parts)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Populate tier rankings into stock-service")
    parser.add_argument(
        "--stock-service-url",
        default="http://localhost:8080",
        help="Stock-service base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads without sending",
    )
    args = parser.parse_args()

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

    # Build payloads
    payloads = [build_tier_payload(r) for r in results]

    # Print summary
    tier_counts = {}
    regime_conditional = 0
    blacklisted = 0
    for p in payloads:
        tier_counts[p["tier"]] = tier_counts.get(p["tier"], 0) + 1
        if p.get("allowed_regimes"):
            regime_conditional += 1
        if p["blacklisted"]:
            blacklisted += 1

    print(f"\nTier Distribution:")
    for tier in ["S", "A", "B", "C", "D", "F"]:
        count = tier_counts.get(tier, 0)
        print(f"  {tier}: {count}")
    print(f"\nRegime-conditional (BULL-only): {regime_conditional}")
    print(f"Blacklisted: {blacklisted}")

    if args.dry_run:
        print(f"\n--- DRY RUN: {len(payloads)} payloads ---")
        for p in payloads:
            regimes = p.get("allowed_regimes", "unrestricted")
            print(f"  {p['symbol']:6s} | {p['tier']} | score={p['composite_score']:5.1f} | "
                  f"conf={p['confidence_multiplier']:.2f} | pos={p['position_size_multiplier']:.2f} | "
                  f"bl={p['blacklisted']} | regimes={regimes}")
        return

    # Send bulk PUT
    url = f"{args.stock_service_url}/api/v1/tiers/bulk"
    print(f"\nSending {len(payloads)} tiers to {url}...")

    try:
        resp = requests.put(url, json=payloads, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        print(f"Result: {result['succeeded']}/{result['total']} succeeded")
        if result.get("errors"):
            print(f"Errors:")
            for err in result["errors"]:
                print(f"  - {err}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to stock-service at {args.stock_service_url}")
        print("Is the service running?")
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {e.response.status_code}: {e.response.text}")


if __name__ == "__main__":
    main()
