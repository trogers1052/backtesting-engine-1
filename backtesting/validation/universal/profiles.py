"""
Symbol profile persistence in PostgreSQL.

Stores winning rule combos, params, and gate results per symbol so
validation runs can skip discovery phases and go straight to tuning.

Two tiers:
  - varsity: configs that went through full 4-gate validation
  - jv: combos that showed promise in combo screen but didn't make the
    top cut for sweep/validation — included in retune to see if they improve
"""
import hashlib
import json
import logging
from typing import List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from ...config import settings
from .config import GENERIC_ENTRY_RULES, MODIFIER_RULES, get_sector_config
from .models import SymbolResult

logger = logging.getLogger(__name__)

# JV threshold: combos with 5+ trades and positive Sharpe that didn't
# make the top_combos cut still get saved for future retune runs.
JV_MIN_TRADES = 5
JV_MIN_SHARPE = 0.0


class ProfileManager:
    """Manages symbol validation profiles in PostgreSQL."""

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        """Get or create a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=settings.trading_db_host,
                port=settings.trading_db_port,
                user=settings.trading_db_user,
                password=settings.trading_db_password,
                database=settings.trading_db_name,
                connect_timeout=10,
            )
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    @staticmethod
    def compute_config_hash(sector: str) -> str:
        """Hash the sector's rules + combos to detect config changes."""
        config = get_sector_config(sector)
        data = {
            "sector_rules": sorted(config.sector_rules),
            "rule_combos": {k: sorted(v) for k, v in sorted(config.rule_combos.items())},
            "generic_entry_rules": sorted(GENERIC_ENTRY_RULES),
            "modifier_rules": sorted(MODIFIER_RULES),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def _upsert_profile(
        self, conn, symbol: str, sector: str, config_hash: str, run_mode: str,
        tier: str, label: str, combo_name: str, rules: list, kwargs_dict: dict,
        total_return=None, total_trades=None, win_rate=None,
        sharpe_ratio=None, profit_factor=None, max_drawdown_pct=None,
        pass_count=0, gates_dict=None, recommendation=None,
        individual_count=None, combo_count=None, sweep_count=None,
    ) -> bool:
        """Upsert a single profile row. Returns True on success."""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO symbol_profiles (
                        symbol, sector, config_hash, run_mode, tier,
                        label, combo_name, rules, kwargs,
                        total_return, total_trades, win_rate,
                        sharpe_ratio, profit_factor, max_drawdown_pct,
                        pass_count, gates, recommendation,
                        individual_rules_tested, combos_tested, sweep_configs_tested,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        NOW()
                    )
                    ON CONFLICT (symbol, label) DO UPDATE SET
                        sector = EXCLUDED.sector,
                        config_hash = EXCLUDED.config_hash,
                        run_mode = EXCLUDED.run_mode,
                        tier = EXCLUDED.tier,
                        combo_name = EXCLUDED.combo_name,
                        rules = EXCLUDED.rules,
                        kwargs = EXCLUDED.kwargs,
                        total_return = EXCLUDED.total_return,
                        total_trades = EXCLUDED.total_trades,
                        win_rate = EXCLUDED.win_rate,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        profit_factor = EXCLUDED.profit_factor,
                        max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                        pass_count = EXCLUDED.pass_count,
                        gates = EXCLUDED.gates,
                        recommendation = EXCLUDED.recommendation,
                        individual_rules_tested = EXCLUDED.individual_rules_tested,
                        combos_tested = EXCLUDED.combos_tested,
                        sweep_configs_tested = EXCLUDED.sweep_configs_tested,
                        updated_at = NOW()
                """, (
                    symbol, sector, config_hash, run_mode, tier,
                    label, combo_name, rules, json.dumps(kwargs_dict),
                    total_return, total_trades, win_rate,
                    sharpe_ratio, profit_factor, max_drawdown_pct,
                    pass_count, json.dumps(gates_dict or {}), recommendation,
                    individual_count, combo_count, sweep_count,
                ))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert profile {symbol}/{label}: {e}")
            return False

    def save_profile(self, symbol_result: SymbolResult, run_mode: str = "full") -> int:
        """Save validated configs + JV combos from a SymbolResult.

        Varsity: configs that went through full 4-gate validation.
        JV: combos from Step 2 that showed promise but didn't make the cut.

        Returns total rows saved.
        """
        conn = self._get_connection()
        config_hash = self.compute_config_hash(symbol_result.sector)
        saved = 0

        individual_count = len(symbol_result.individual_results)
        combo_count = len(symbol_result.combo_results)
        sweep_count = len(symbol_result.sweep_results)

        # ── Varsity: validated configs ──
        for report in symbol_result.validated_reports:
            if not report.backtest:
                continue

            kwargs = report.kwargs.copy()
            rules = kwargs.pop("rules", [])
            combo_name = report.label.split()[0] if report.label else "unknown"
            gates_dict = {
                g.name: {"passed": g.passed, "detail": g.detail}
                for g in report.gates
            }

            ok = self._upsert_profile(
                conn, symbol_result.symbol, symbol_result.sector,
                config_hash, run_mode, "varsity",
                report.label, combo_name, rules, kwargs,
                total_return=report.backtest.total_return,
                total_trades=report.backtest.total_trades,
                win_rate=report.backtest.win_rate,
                sharpe_ratio=report.backtest.sharpe_ratio,
                profit_factor=report.backtest.profit_factor,
                max_drawdown_pct=report.backtest.max_drawdown_pct,
                pass_count=report.pass_count,
                gates_dict=gates_dict,
                recommendation=symbol_result.recommendation,
                individual_count=individual_count,
                combo_count=combo_count,
                sweep_count=sweep_count,
            )
            if ok:
                saved += 1

        # ── JV: promising combos that didn't make the top cut ──
        # Identify which combo names are already in varsity
        varsity_combos = set()
        for report in symbol_result.validated_reports:
            if report.label:
                varsity_combos.add(report.label.split()[0])

        for combo_name, kwargs, result in symbol_result.combo_results:
            # Skip if already represented in varsity
            if combo_name in varsity_combos:
                continue
            # JV threshold: must have enough trades and non-negative Sharpe
            if result.total_trades < JV_MIN_TRADES:
                continue
            if (result.sharpe_ratio or 0) < JV_MIN_SHARPE:
                continue

            rules = kwargs.get("rules", [])
            jv_kwargs = {k: v for k, v in kwargs.items() if k != "rules"}
            label = f"{combo_name} [jv]"

            ok = self._upsert_profile(
                conn, symbol_result.symbol, symbol_result.sector,
                config_hash, run_mode, "jv",
                label, combo_name, rules, jv_kwargs,
                total_return=result.total_return,
                total_trades=result.total_trades,
                win_rate=result.win_rate,
                sharpe_ratio=result.sharpe_ratio,
                profit_factor=result.profit_factor,
                max_drawdown_pct=result.max_drawdown_pct,
                recommendation="jv",
            )
            if ok:
                saved += 1

        logger.info(
            f"Saved {saved} profile(s) for {symbol_result.symbol} "
            f"({len(varsity_combos)} varsity, {saved - len(varsity_combos)} jv)"
        )
        return saved

    def has_profile(self, symbol: str) -> bool:
        """Check if any profile exists for this symbol."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM symbol_profiles WHERE symbol = %s LIMIT 1", (symbol,))
            return cur.fetchone() is not None

    def is_stale(self, symbol: str, sector: str) -> bool:
        """Check if the profile was built with a different config version."""
        current_hash = self.compute_config_hash(sector)
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT config_hash FROM symbol_profiles WHERE symbol = %s LIMIT 1",
                (symbol,),
            )
            row = cur.fetchone()
            if not row:
                return True
            return row[0] != current_hash

    def get_known_combos(self, symbol: str) -> Optional[List[Tuple[str, dict]]]:
        """Get combos for retune mode — both varsity AND JV.

        Returns list of (combo_name, base_kwargs) where base_kwargs has 'rules'.
        JV combos get a second chance on every retune run.
        """
        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT combo_name, rules, tier
                FROM symbol_profiles
                WHERE symbol = %s
                ORDER BY tier, combo_name
            """, (symbol,))
            rows = cur.fetchall()

        if not rows:
            return None

        combos = []
        for row in rows:
            base_kwargs = {"rules": row["rules"]}
            combos.append((row["combo_name"], base_kwargs))

        varsity_count = sum(1 for r in rows if r["tier"] == "varsity")
        jv_count = sum(1 for r in rows if r["tier"] == "jv")
        logger.info(f"{symbol}: loading {varsity_count} varsity + {jv_count} jv combos for retune")
        return combos

    def get_winning_configs(self, symbol: str) -> Optional[List[Tuple[str, dict]]]:
        """Get exact winning kwargs for revalidate mode (varsity only).

        Returns list of (label, full_kwargs) pairs.
        """
        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT label, rules, kwargs
                FROM symbol_profiles
                WHERE symbol = %s AND tier = 'varsity'
                ORDER BY pass_count DESC, sharpe_ratio DESC
            """, (symbol,))
            rows = cur.fetchall()

        if not rows:
            return None

        configs = []
        for row in rows:
            kwargs = json.loads(row["kwargs"]) if isinstance(row["kwargs"], str) else row["kwargs"]
            kwargs["rules"] = row["rules"]
            configs.append((row["label"], kwargs))
        return configs

    def list_profiles(self) -> List[dict]:
        """List all profiled symbols with summary info."""
        conn = self._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT symbol, sector, config_hash,
                       MAX(pass_count) as best_gates,
                       MAX(sharpe_ratio) as best_sharpe,
                       MAX(recommendation) FILTER (WHERE tier = 'varsity') as recommendation,
                       MAX(updated_at) as last_updated,
                       COUNT(*) FILTER (WHERE tier = 'varsity') as varsity_count,
                       COUNT(*) FILTER (WHERE tier = 'jv') as jv_count
                FROM symbol_profiles
                GROUP BY symbol, sector, config_hash
                ORDER BY MAX(pass_count) DESC, MAX(sharpe_ratio) DESC
            """)
            return cur.fetchall()

    def delete_symbol(self, symbol: str) -> int:
        """Delete all profiles for a symbol. Returns rows deleted."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM symbol_profiles WHERE symbol = %s", (symbol,))
            deleted = cur.rowcount
        conn.commit()
        return deleted

    def promote_jv(self, symbol: str, combo_name: str) -> bool:
        """Promote a JV combo to varsity after it passes validation."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE symbol_profiles
                SET tier = 'varsity', updated_at = NOW()
                WHERE symbol = %s AND combo_name = %s AND tier = 'jv'
            """, (symbol, combo_name))
            updated = cur.rowcount > 0
        conn.commit()
        return updated
