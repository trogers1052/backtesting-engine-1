#!/usr/bin/env python3
"""Backfill VIX data from FRED API into TimescaleDB.

FRED series VIXCLS provides daily VIX closing values. This script fetches
historical data and inserts it into the ohlcv_1min table as symbol "VIX"
so that the RegimeClassifier can use it for volatility overlay.

Usage:
    python scripts/backfill_vix.py                          # default: 2018-01-01 to today
    python scripts/backfill_vix.py --start 2020-01-01       # custom start
    python scripts/backfill_vix.py --dry-run                # preview without inserting
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from urllib.parse import urlencode
from urllib.request import urlopen

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FRED_API_KEY = "aa00ca527b2ecf48ecf96047934d25a3"
FRED_SERIES = "VIXCLS"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# Database defaults (match backtesting config.py)
DB_HOST = "localhost"
DB_PORT = 5433
DB_USER = "ingestor"
DB_PASSWORD = "ingestor"
DB_NAME = "stock_db"


def fetch_vix_from_fred(
    start_date: str, end_date: str
) -> list[dict]:
    """Fetch VIXCLS observations from FRED API.

    Returns list of dicts with 'date' and 'value' keys.
    Skips entries where value is '.' (missing/holiday).
    """
    params = {
        "series_id": FRED_SERIES,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }

    logger.info(f"Fetching VIXCLS from FRED: {start_date} to {end_date}")
    url = f"{FRED_URL}?{urlencode(params)}"
    with urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    observations = data.get("observations", [])

    rows = []
    skipped = 0
    for obs in observations:
        val = obs["value"]
        if val == ".":
            skipped += 1
            continue
        rows.append({
            "date": obs["date"],
            "value": float(val),
        })

    logger.info(
        f"Fetched {len(rows)} VIX observations ({skipped} missing/holiday days skipped)"
    )
    return rows


def insert_vix_rows(rows: list[dict], dry_run: bool = False) -> int:
    """Insert VIX rows into ohlcv_1min table.

    Uses ON CONFLICT DO UPDATE to handle re-runs safely.
    VIX is stored as: open=high=low=close=vix_value, volume=0.
    Timestamp set to market open (14:30 UTC / 9:30 ET) for each date.
    """
    if not rows:
        logger.warning("No rows to insert")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(rows)} rows")
        for r in rows[:5]:
            logger.info(f"  {r['date']} -> VIX={r['value']:.2f}")
        if len(rows) > 5:
            logger.info(f"  ... and {len(rows) - 5} more")
        return len(rows)

    upsert_sql = """
        INSERT INTO ohlcv_1min (time, symbol, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
    """

    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            connect_timeout=10,
        )

        inserted = 0
        with conn.cursor() as cur:
            for row in rows:
                # Use 14:30 UTC (9:30 AM ET market open) as the timestamp
                ts = datetime.strptime(row["date"], "%Y-%m-%d").replace(
                    hour=14, minute=30
                )
                vix = row["value"]
                cur.execute(upsert_sql, (ts, "VIX", vix, vix, vix, vix, 0))
                inserted += 1

            conn.commit()

        logger.info(f"Inserted/updated {inserted} VIX rows into ohlcv_1min")
        return inserted

    except Exception as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def verify_data() -> None:
    """Print a summary of VIX data in the database."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    MIN(time) as first_date,
                    MAX(time) as last_date,
                    COUNT(*) as total_rows,
                    MIN(close) as min_vix,
                    MAX(close) as max_vix,
                    AVG(close) as avg_vix
                FROM ohlcv_1min
                WHERE symbol = 'VIX';
            """)
            row = cur.fetchone()

        if row and row[0]:
            logger.info(
                f"VIX data in DB: {row[0].date()} to {row[1].date()} | "
                f"{row[2]} rows | "
                f"min={row[3]:.2f} max={row[4]:.2f} avg={row[5]:.2f}"
            )
        else:
            logger.warning("No VIX data found in database")

    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill VIX data from FRED API")
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Start date (YYYY-MM-DD, default: 2018-01-01)",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview data without inserting into database",
    )
    args = parser.parse_args()

    rows = fetch_vix_from_fred(args.start, args.end)

    if not rows:
        logger.error("No data fetched from FRED. Exiting.")
        sys.exit(1)

    count = insert_vix_rows(rows, dry_run=args.dry_run)

    if not args.dry_run:
        verify_data()

    logger.info(f"Done. {count} rows processed.")


if __name__ == "__main__":
    main()
