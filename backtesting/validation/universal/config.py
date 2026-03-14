"""
Sector configuration registry for universal validation.

Maps sectors to their rules, combos, and stock metadata.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# Generic rules available to ALL sectors
# ============================================================================

GENERIC_ENTRY_RULES = [
    "buy_dip_in_uptrend",
    "enhanced_buy_dip",
    "momentum_reversal",
    "trend_continuation",
    "dip_recovery",
    "rsi_oversold",
    "macd_bullish_crossover",
    "macd_momentum",
    "rsi_macd_confluence",
]

MODIFIER_RULES = [
    "death_cross",
    "trend_alignment",
    "golden_cross",
    "trend_break_warning",
    "rsi_overbought",
    "macd_bearish_crossover",
]

# Standard parameter sweep grids (fine — used in Phase 2 neighbor search)
CONFIDENCE_VALUES = [0.40, 0.45, 0.50, 0.55, 0.60]
PROFIT_TARGET_VALUES = [0.07, 0.08, 0.10, 0.12, 0.15]
MAX_LOSS_VALUES = [3.0, 4.0, 5.0, 6.0, 8.0]
COOLDOWN_VALUES = [3, 5, 7, 10]

# Coarse grid for Phase 1 of two-phase adaptive sweep (~54 configs vs 500)
COARSE_CONFIDENCE = [0.40, 0.50, 0.60]
COARSE_PROFIT_TARGET = [0.07, 0.10, 0.15]
COARSE_MAX_LOSS = [3.0, 5.0, 8.0]
COARSE_COOLDOWN = [3, 7]


# ============================================================================
# Sector configurations
# ============================================================================

@dataclass
class SectorConfig:
    """Everything needed to validate stocks in a sector."""
    name: str
    sector_rules: List[str]  # Sector-specific entry rules
    rule_combos: Dict[str, List[str]]  # Pre-built combos to test
    sub_sector_rules: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def individual_rules(self) -> List[str]:
        """All entry rules to test individually: sector + generic."""
        return self.sector_rules + GENERIC_ENTRY_RULES


# ── Mining (gold, silver, platinum, uranium, rare earth) ─────────────────

MINING_CONFIG = SectorConfig(
    name="mining",
    sector_rules=[
        "commodity_breakout",
        "miner_metal_ratio",
        "dollar_weakness",
        "seasonality",
        "volume_breakout",
    ],
    rule_combos={
        # Lean mining combos (proven on AEM)
        "mining_lean_4": ["momentum_reversal", "commodity_breakout", "seasonality", "death_cross"],
        "mining_core_3": ["commodity_breakout", "seasonality", "death_cross"],
        "mining_momentum_3": ["momentum_reversal", "seasonality", "death_cross"],
        # Wider mining combos
        "mining_5": ["commodity_breakout", "miner_metal_ratio", "seasonality", "death_cross", "trend_alignment"],
        "mining_dip_4": ["enhanced_buy_dip", "commodity_breakout", "seasonality", "death_cross"],
        "mining_full_7": [
            "commodity_breakout", "miner_metal_ratio", "dollar_weakness",
            "seasonality", "volume_breakout", "death_cross", "trend_alignment",
        ],
        # Generic combos for comparison
        "mining_generic_wide": [
            "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
            "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
            "golden_cross", "trend_break_warning", "death_cross", "seasonality",
        ],
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "gold_miner": ["commodity_breakout", "miner_metal_ratio", "dollar_weakness"],
        "silver": ["commodity_breakout", "dollar_weakness"],
        "platinum": ["commodity_breakout"],
        "uranium": ["commodity_breakout", "volume_breakout"],
        "rare_earth": ["commodity_breakout", "volume_breakout"],
    },
)

# ── Energy (integrated, upstream, midstream, refining) ───────────────────

ENERGY_CONFIG = SectorConfig(
    name="energy",
    sector_rules=[
        "energy_momentum",
        "energy_mean_reversion",
        "energy_seasonality",
        "midstream_yield_reversion",
    ],
    rule_combos={
        # Pure energy combos
        "energy_core_3": ["energy_momentum", "energy_seasonality", "death_cross"],
        "energy_mean_rev_3": ["energy_mean_reversion", "energy_seasonality", "death_cross"],
        "energy_full_4": ["energy_momentum", "energy_mean_reversion", "energy_seasonality", "death_cross"],
        # Midstream-specific
        "midstream_yield_3": ["midstream_yield_reversion", "energy_seasonality", "death_cross"],
        "midstream_full_4": ["midstream_yield_reversion", "energy_momentum", "energy_seasonality", "death_cross"],
        # Energy + generic hybrids
        "energy_dip_4": ["energy_mean_reversion", "enhanced_buy_dip", "energy_seasonality", "death_cross"],
        "energy_momentum_dip": ["energy_momentum", "dip_recovery", "energy_seasonality", "death_cross"],
        "energy_rsi_4": ["energy_mean_reversion", "rsi_oversold", "energy_seasonality", "death_cross"],
        "energy_macd_4": ["energy_momentum", "macd_bullish_crossover", "energy_seasonality", "death_cross"],
        "energy_trend_4": ["energy_momentum", "trend_continuation", "energy_seasonality", "death_cross"],
        # Wider energy combos
        "energy_7": [
            "energy_momentum", "energy_mean_reversion", "energy_seasonality",
            "midstream_yield_reversion", "death_cross", "trend_alignment", "golden_cross",
        ],
        "energy_wide_10": [
            "energy_momentum", "energy_mean_reversion", "energy_seasonality",
            "midstream_yield_reversion", "enhanced_buy_dip", "momentum_reversal",
            "trend_continuation", "death_cross", "trend_alignment", "macd_bearish_crossover",
        ],
        # Generic baselines
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "midstream": ["midstream_yield_reversion"],
        "integrated": [],
        "upstream": [],
        "refining": [],
    },
)

# ── Industrial (machinery, conglomerate, electrical) ─────────────────────

INDUSTRIAL_CONFIG = SectorConfig(
    name="industrial",
    sector_rules=[
        "industrial_mean_reversion",
        "industrial_pullback",
        "industrial_seasonality",
    ],
    rule_combos={
        "industrial_core_3": ["industrial_mean_reversion", "industrial_seasonality", "death_cross"],
        "industrial_pullback_3": ["industrial_pullback", "industrial_seasonality", "death_cross"],
        "industrial_full_4": ["industrial_mean_reversion", "industrial_pullback", "industrial_seasonality", "death_cross"],
        "industrial_dip_4": ["industrial_mean_reversion", "enhanced_buy_dip", "industrial_seasonality", "death_cross"],
        "industrial_momentum_4": ["industrial_pullback", "momentum_reversal", "industrial_seasonality", "death_cross"],
        "industrial_generic_wide": [
            "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
            "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
            "trend_break_warning", "death_cross", "industrial_seasonality",
        ],
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "heavy_equipment": [],
        "conglomerate": [],
        "electrical": [],
        "aerospace": [],
    },
)

# ── Defense/Aerospace (primes, defense tech, ETFs) ───────────────────────

DEFENSE_CONFIG = SectorConfig(
    name="defense",
    sector_rules=[
        "defense_budget_cycle",
        "defense_mean_reversion",
        "defense_momentum",
        "defense_counter_cyclical",
    ],
    rule_combos={
        # Pure defense combos
        "defense_core_3": ["defense_budget_cycle", "defense_momentum", "death_cross"],
        "defense_mean_rev_3": ["defense_mean_reversion", "defense_budget_cycle", "death_cross"],
        "defense_full_4": ["defense_budget_cycle", "defense_mean_reversion", "defense_momentum", "death_cross"],
        "defense_counter_4": ["defense_counter_cyclical", "defense_budget_cycle", "defense_mean_reversion", "death_cross"],
        # Defense + generic hybrids (user requested)
        "defense_dip_4": ["defense_mean_reversion", "enhanced_buy_dip", "defense_budget_cycle", "death_cross"],
        "defense_momentum_dip": ["defense_momentum", "dip_recovery", "defense_budget_cycle", "death_cross"],
        "defense_trend_4": ["defense_momentum", "trend_continuation", "defense_budget_cycle", "death_cross"],
        "defense_macd_4": ["defense_momentum", "macd_bullish_crossover", "defense_budget_cycle", "death_cross"],
        "defense_rsi_4": ["defense_mean_reversion", "rsi_oversold", "defense_budget_cycle", "death_cross"],
        # Wider combos
        "defense_wide_7": [
            "defense_budget_cycle", "defense_mean_reversion", "defense_momentum",
            "defense_counter_cyclical", "death_cross", "trend_alignment", "golden_cross",
        ],
        "defense_generic_wide": [
            "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
            "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
            "trend_break_warning", "death_cross", "defense_budget_cycle",
        ],
        # Generic baselines (for comparison)
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "prime": ["defense_counter_cyclical"],
        "defense_tech": ["defense_momentum"],
        "defense_services": ["defense_counter_cyclical"],
        "defense_etf": [],
    },
)

# ── Tech/Growth ─────────────────────────────────────────────────────────

TECH_CONFIG = SectorConfig(
    name="tech",
    sector_rules=[
        "tech_ema_pullback",
        "tech_mean_reversion",
        "tech_seasonality",
        "semi_cycle",
    ],
    rule_combos={
        "tech_core_3": ["tech_ema_pullback", "tech_seasonality", "death_cross"],
        "tech_mean_rev_3": ["tech_mean_reversion", "tech_seasonality", "death_cross"],
        "tech_full_4": ["tech_ema_pullback", "tech_mean_reversion", "tech_seasonality", "death_cross"],
        "tech_dip_4": ["tech_mean_reversion", "enhanced_buy_dip", "tech_seasonality", "death_cross"],
        "tech_momentum_4": ["tech_ema_pullback", "momentum_reversal", "tech_seasonality", "death_cross"],
        # Semi-specific combos (AMAT, LRCX, MU, AMD)
        "semi_cycle_3": ["semi_cycle", "tech_seasonality", "death_cross"],
        "semi_cycle_ema_4": ["semi_cycle", "tech_ema_pullback", "tech_seasonality", "death_cross"],
        "semi_cycle_dip_4": ["semi_cycle", "enhanced_buy_dip", "tech_seasonality", "death_cross"],
        "semi_cycle_full_5": ["semi_cycle", "tech_ema_pullback", "tech_mean_reversion", "tech_seasonality", "death_cross"],
        # Wide combos
        "tech_wide_7": [
            "tech_ema_pullback", "tech_mean_reversion", "tech_seasonality",
            "enhanced_buy_dip", "momentum_reversal", "death_cross", "trend_alignment",
        ],
        "tech_semi_wide_8": [
            "semi_cycle", "tech_ema_pullback", "tech_mean_reversion", "tech_seasonality",
            "enhanced_buy_dip", "momentum_reversal", "death_cross", "trend_alignment",
        ],
        # Generic baselines
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "semi_equip": ["semi_cycle"],
        "memory": ["semi_cycle"],
        "mega_cap": ["tech_ema_pullback"],
        "saas": ["tech_ema_pullback"],
        "low_beta_tech": ["tech_mean_reversion"],
        "ai_speculative": [],
        "fintech": [],
        "tech_etf": ["tech_mean_reversion"],
    },
)

# ── Financial ───────────────────────────────────────────────────────────

FINANCIAL_CONFIG = SectorConfig(
    name="financial",
    sector_rules=[
        "financial_mean_reversion",
        "financial_pullback",
        "financial_seasonality",
    ],
    rule_combos={
        "financial_core_3": ["financial_mean_reversion", "financial_seasonality", "death_cross"],
        "financial_pullback_3": ["financial_pullback", "financial_seasonality", "death_cross"],
        "financial_full_4": ["financial_mean_reversion", "financial_pullback", "financial_seasonality", "death_cross"],
        "financial_dip_4": ["financial_mean_reversion", "enhanced_buy_dip", "financial_seasonality", "death_cross"],
        "financial_pullback_dip_4": ["financial_pullback", "enhanced_buy_dip", "financial_seasonality", "death_cross"],
        "financial_momentum_4": ["financial_pullback", "momentum_reversal", "financial_seasonality", "death_cross"],
        "financial_pullback_macd_4": ["financial_pullback", "macd_bullish_crossover", "financial_seasonality", "death_cross"],
        "financial_wide_7": [
            "financial_mean_reversion", "financial_pullback", "financial_seasonality",
            "enhanced_buy_dip", "momentum_reversal", "death_cross", "trend_alignment",
        ],
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "bank": ["financial_mean_reversion"],
        "investment_bank": ["financial_pullback"],
        "insurance": ["financial_mean_reversion"],
        "payments": ["financial_pullback"],
        "financial_data": ["financial_pullback"],
        "fintech": [],
        "reit": ["financial_mean_reversion"],
        "sector_etf": [],
    },
)

# ── Utility ─────────────────────────────────────────────────────────────

UTILITY_CONFIG = SectorConfig(
    name="utility",
    sector_rules=[
        "utility_mean_reversion",
        "utility_rate_reversion",
        "utility_seasonality",
        "nuclear_power_momentum",
    ],
    rule_combos={
        # Traditional utility combos
        "utility_core_3": ["utility_mean_reversion", "utility_seasonality", "death_cross"],
        "utility_rate_3": ["utility_rate_reversion", "utility_seasonality", "death_cross"],
        "utility_full_4": ["utility_mean_reversion", "utility_rate_reversion", "utility_seasonality", "death_cross"],
        "utility_dip_4": ["utility_mean_reversion", "enhanced_buy_dip", "utility_seasonality", "death_cross"],
        # Nuclear/AI power combos (VST, CEG, OKLO)
        "nuclear_momentum_3": ["nuclear_power_momentum", "utility_seasonality", "death_cross"],
        "nuclear_momentum_dip_4": ["nuclear_power_momentum", "dip_recovery", "utility_seasonality", "death_cross"],
        "nuclear_momentum_macd_4": ["nuclear_power_momentum", "macd_bullish_crossover", "utility_seasonality", "death_cross"],
        "nuclear_momentum_trend_4": ["nuclear_power_momentum", "trend_continuation", "utility_seasonality", "death_cross"],
        "nuclear_full_5": [
            "nuclear_power_momentum", "utility_seasonality",
            "momentum_reversal", "trend_continuation", "death_cross",
        ],
        # Wide combos
        "utility_wide_7": [
            "utility_mean_reversion", "utility_rate_reversion", "utility_seasonality",
            "nuclear_power_momentum", "enhanced_buy_dip", "death_cross", "trend_alignment",
        ],
        # Generic baselines
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
    sub_sector_rules={
        "regulated": ["utility_mean_reversion", "utility_rate_reversion"],
        "water": ["utility_mean_reversion"],
        "yieldco": ["utility_mean_reversion"],
        "nuclear_power": ["nuclear_power_momentum"],
        "utility_etf": ["utility_mean_reversion"],
    },
)

# ── Consumer Staples ────────────────────────────────────────────────────

CONSUMER_STAPLES_CONFIG = SectorConfig(
    name="consumer_staples",
    sector_rules=[
        "consumer_staples_mean_reversion",
        "consumer_staples_pullback",
        "consumer_staples_seasonality",
    ],
    rule_combos={
        "staples_core_3": ["consumer_staples_mean_reversion", "consumer_staples_seasonality", "death_cross"],
        "staples_pullback_3": ["consumer_staples_pullback", "consumer_staples_seasonality", "death_cross"],
        "staples_full_4": [
            "consumer_staples_mean_reversion", "consumer_staples_pullback",
            "consumer_staples_seasonality", "death_cross",
        ],
        "staples_dip_4": [
            "consumer_staples_mean_reversion", "enhanced_buy_dip",
            "consumer_staples_seasonality", "death_cross",
        ],
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
)

# ── Healthcare ──────────────────────────────────────────────────────────

HEALTHCARE_CONFIG = SectorConfig(
    name="healthcare",
    sector_rules=[
        "healthcare_mean_reversion",
        "healthcare_pullback",
        "healthcare_seasonality",
    ],
    rule_combos={
        "health_core_3": ["healthcare_mean_reversion", "healthcare_seasonality", "death_cross"],
        "health_pullback_3": ["healthcare_pullback", "healthcare_seasonality", "death_cross"],
        "health_full_4": [
            "healthcare_mean_reversion", "healthcare_pullback",
            "healthcare_seasonality", "death_cross",
        ],
        "health_dip_4": [
            "healthcare_mean_reversion", "enhanced_buy_dip",
            "healthcare_seasonality", "death_cross",
        ],
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
    },
)

# ── Generic (ETFs and stocks without sector-specific rules) ─────────────

GENERIC_CONFIG = SectorConfig(
    name="generic",
    sector_rules=[],  # No sector-specific rules — use only generic
    rule_combos={
        "generic_wide_10": [
            "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
            "rsi_oversold", "macd_bearish_crossover", "trend_alignment",
            "golden_cross", "trend_break_warning", "death_cross",
        ],
        "generic_lean_5": [
            "enhanced_buy_dip", "momentum_reversal", "trend_continuation",
            "death_cross", "trend_alignment",
        ],
        "generic_dip_3": ["enhanced_buy_dip", "trend_continuation", "death_cross"],
        "generic_momentum_3": ["momentum_reversal", "trend_continuation", "death_cross"],
        "generic_rsi_3": ["rsi_oversold", "trend_continuation", "death_cross"],
        "generic_macd_3": ["macd_bullish_crossover", "trend_continuation", "death_cross"],
        "generic_dip_momentum_4": [
            "enhanced_buy_dip", "momentum_reversal",
            "trend_continuation", "death_cross",
        ],
        "generic_full_trend_5": [
            "momentum_reversal", "trend_continuation", "trend_alignment",
            "golden_cross", "death_cross",
        ],
    },
)


# ============================================================================
# Registry: sector name → config
# ============================================================================

SECTOR_REGISTRY: Dict[str, SectorConfig] = {
    "mining": MINING_CONFIG,
    "energy": ENERGY_CONFIG,
    "industrial": INDUSTRIAL_CONFIG,
    "defense": DEFENSE_CONFIG,
    "tech": TECH_CONFIG,
    "financial": FINANCIAL_CONFIG,
    "utility": UTILITY_CONFIG,
    "consumer_staples": CONSUMER_STAPLES_CONFIG,
    "healthcare": HEALTHCARE_CONFIG,
    "generic": GENERIC_CONFIG,
}


# ============================================================================
# Symbol → sector/sub-sector mapping
# ============================================================================

SYMBOL_SECTORS: Dict[str, Dict] = {
    # ── Mining & Materials (20 validated) ────────────────────────────────
    # Gold
    "AEM": {"sector": "mining", "sub_sector": "gold_miner", "name": "Agnico Eagle Mines", "tier": "A"},
    "WPM": {"sector": "mining", "sub_sector": "gold_miner", "name": "Wheaton Precious Metals", "tier": "A"},
    "GLD": {"sector": "mining", "sub_sector": "gold_miner", "name": "SPDR Gold Trust", "tier": "A"},
    "GDX": {"sector": "mining", "sub_sector": "gold_miner", "name": "VanEck Gold Miners ETF", "tier": "B"},
    "IAUM": {"sector": "mining", "sub_sector": "gold_miner", "name": "iShares Gold Micro ETF", "tier": "B"},
    # Silver
    "SLV": {"sector": "mining", "sub_sector": "silver", "name": "iShares Silver Trust", "tier": "C"},
    "PAAS": {"sector": "mining", "sub_sector": "silver", "name": "Pan American Silver", "tier": "C"},
    "SIL": {"sector": "mining", "sub_sector": "silver", "name": "Global X Silver Miners ETF", "tier": "D"},
    "HL": {"sector": "mining", "sub_sector": "silver", "name": "Hecla Mining", "tier": "F"},
    # Platinum
    "PPLT": {"sector": "mining", "sub_sector": "platinum", "name": "Platinum ETF", "tier": "A"},
    # Uranium
    "CCJ": {"sector": "mining", "sub_sector": "uranium", "name": "Cameco - Uranium", "tier": "B"},
    "URNM": {"sector": "mining", "sub_sector": "uranium", "name": "Sprott Uranium Miners ETF", "tier": "B"},
    "UUUU": {"sector": "mining", "sub_sector": "uranium", "name": "Energy Fuels - Uranium", "tier": "B"},
    "URA": {"sector": "mining", "sub_sector": "uranium", "name": "Global X Uranium ETF", "tier": "F"},
    # Copper
    "COPX": {"sector": "mining", "sub_sector": "copper", "name": "Global X Copper Miners ETF", "tier": "B"},
    "FCX": {"sector": "mining", "sub_sector": "copper", "name": "Freeport-McMoRan", "tier": "D"},
    # Rare earth
    "MP": {"sector": "mining", "sub_sector": "rare_earth", "name": "MP Materials - Rare Earth", "tier": "C"},
    "REMX": {"sector": "mining", "sub_sector": "rare_earth", "name": "VanEck Rare Earth ETF", "tier": "F"},
    "USAR": {"sector": "mining", "sub_sector": "rare_earth", "name": "USA Rare Earth", "tier": "F"},
    # Leveraged (blacklisted)
    "GUSH": {"sector": "energy", "sub_sector": "leveraged_etf", "name": "Direxion Daily S&P Oil & Gas Bull 2X", "tier": "F"},

    # ── Energy — Fossil (15 validated) ───────────────────────────────────
    "OKE": {"sector": "energy", "sub_sector": "midstream", "name": "ONEOK - Midstream", "tier": "A"},
    "PSX": {"sector": "energy", "sub_sector": "integrated", "name": "Phillips 66 - Integrated", "tier": "A"},
    "FET": {"sector": "energy", "sub_sector": "oilfield_services", "name": "Forum Energy Technologies", "tier": "C"},
    "XOM": {"sector": "energy", "sub_sector": "integrated", "name": "ExxonMobil", "tier": "D"},
    "EPD": {"sector": "energy", "sub_sector": "midstream", "name": "Enterprise Products Partners", "tier": "D"},
    "XLE": {"sector": "energy", "sub_sector": "energy_etf", "name": "Energy Select Sector SPDR ETF", "tier": "D"},
    "ET": {"sector": "energy", "sub_sector": "midstream", "name": "Energy Transfer", "tier": "D"},
    "EQNR": {"sector": "energy", "sub_sector": "integrated", "name": "Equinor", "tier": "D"},
    "CVX": {"sector": "energy", "sub_sector": "integrated", "name": "Chevron", "tier": "F"},
    "COP": {"sector": "energy", "sub_sector": "upstream", "name": "ConocoPhillips", "tier": "F"},
    "OXY": {"sector": "energy", "sub_sector": "upstream", "name": "Occidental Petroleum", "tier": "F"},
    "EOG": {"sector": "energy", "sub_sector": "upstream", "name": "EOG Resources", "tier": "F"},
    "KMI": {"sector": "energy", "sub_sector": "midstream", "name": "Kinder Morgan", "tier": "Watchlist"},
    "FANG": {"sector": "energy", "sub_sector": "upstream", "name": "Diamondback Energy", "tier": "Watchlist"},

    # ── Industrial (4 validated) ─────────────────────────────────────────
    "ETN": {"sector": "industrial", "sub_sector": "electrical", "name": "Eaton Corporation", "tier": "A"},
    "XLI": {"sector": "industrial", "sub_sector": "industrial_etf", "name": "Industrial Select Sector SPDR", "tier": "A"},
    "CAT": {"sector": "industrial", "sub_sector": "heavy_equipment", "name": "Caterpillar", "tier": "B"},
    "WLDN": {"sector": "industrial", "sub_sector": "electrical", "name": "Willdan Group - Infrastructure", "tier": "B"},

    # ── Defense / Aerospace (4 validated + watchlist) ────────────────────
    "RTX": {"sector": "defense", "sub_sector": "prime", "name": "RTX Corp - Defense/Aerospace", "tier": "C"},
    "ITA": {"sector": "defense", "sub_sector": "defense_etf", "name": "iShares US Aerospace & Defense ETF", "tier": "C"},
    "AVAV": {"sector": "defense", "sub_sector": "defense_tech", "name": "AeroVironment - Defense Drones", "tier": "C"},
    "LMT": {"sector": "defense", "sub_sector": "prime", "name": "Lockheed Martin", "tier": "F"},
    "NOC": {"sector": "defense", "sub_sector": "prime", "name": "Northrop Grumman", "tier": "Watchlist"},
    "GD": {"sector": "defense", "sub_sector": "prime", "name": "General Dynamics", "tier": "Watchlist"},
    "XAR": {"sector": "defense", "sub_sector": "defense_etf", "name": "SPDR S&P Aerospace & Defense ETF", "tier": "Watchlist"},

    # ── Tech & Growth (28 validated) ─────────────────────────────────────
    # S tier
    "APH": {"sector": "tech", "sub_sector": "low_beta_tech", "name": "Amphenol - Electronic Components", "tier": "S"},
    # A tier
    "AMAT": {"sector": "tech", "sub_sector": "semi_equip", "name": "Applied Materials - Semi Equipment", "tier": "A"},
    "LRCX": {"sector": "tech", "sub_sector": "semi_equip", "name": "Lam Research - Semi Equipment", "tier": "A"},
    "GOOGL": {"sector": "tech", "sub_sector": "mega_cap", "name": "Alphabet/Google", "tier": "A"},
    # B tier
    "MU": {"sector": "tech", "sub_sector": "memory", "name": "Micron Technology - Memory", "tier": "B"},
    "VRTX": {"sector": "healthcare", "sub_sector": "large_biotech", "name": "Vertex Pharmaceuticals", "tier": "B"},
    "CRWD": {"sector": "tech", "sub_sector": "saas", "name": "CrowdStrike - Cybersecurity", "tier": "B"},
    # C tier
    "XLK": {"sector": "tech", "sub_sector": "tech_etf", "name": "Technology Select Sector SPDR ETF", "tier": "C"},
    "NU": {"sector": "financial", "sub_sector": "fintech", "name": "Nu Holdings - Fintech", "tier": "C"},
    "QQQ": {"sector": "tech", "sub_sector": "tech_etf", "name": "Invesco QQQ Trust", "tier": "C"},
    "EWY": {"sector": "tech", "sub_sector": "tech_etf", "name": "iShares MSCI South Korea ETF", "tier": "C"},
    "SDGR": {"sector": "tech", "sub_sector": "saas", "name": "Schrödinger - Computational Biology", "tier": "C"},
    # D tier
    "CRM": {"sector": "tech", "sub_sector": "saas", "name": "Salesforce", "tier": "D"},
    "AMD": {"sector": "tech", "sub_sector": "semi_equip", "name": "Advanced Micro Devices", "tier": "D"},
    "NOW": {"sector": "tech", "sub_sector": "saas", "name": "ServiceNow", "tier": "D"},
    "PANW": {"sector": "tech", "sub_sector": "saas", "name": "Palo Alto Networks - Cybersecurity", "tier": "D"},
    "AMZN": {"sector": "tech", "sub_sector": "mega_cap", "name": "Amazon", "tier": "D"},
    "NET": {"sector": "tech", "sub_sector": "saas", "name": "Cloudflare", "tier": "D"},
    # F tier
    "AAPL": {"sector": "tech", "sub_sector": "mega_cap", "name": "Apple", "tier": "F"},
    "SOFI": {"sector": "tech", "sub_sector": "fintech", "name": "SoFi Technologies", "tier": "F"},
    "ALAB": {"sector": "tech", "sub_sector": "ai_speculative", "name": "Astera Labs", "tier": "F"},
    "S": {"sector": "tech", "sub_sector": "saas", "name": "SentinelOne - Cybersecurity", "tier": "F"},
    "MELI": {"sector": "generic", "sub_sector": "unknown", "name": "MercadoLibre - LatAm E-commerce", "tier": "F"},
    "SNDK": {"sector": "tech", "sub_sector": "memory", "name": "SanDisk", "tier": "F"},
    "TEM": {"sector": "tech", "sub_sector": "ai_speculative", "name": "Tempus AI", "tier": "F"},
    "TMDX": {"sector": "healthcare", "sub_sector": "med_growth", "name": "TransMedics - Med-Tech", "tier": "F"},

    # ── Financial (10 validated) ─────────────────────────────────────────
    "JPM": {"sector": "financial", "sub_sector": "bank", "name": "JPMorgan Chase", "tier": "A"},
    "V": {"sector": "financial", "sub_sector": "payments", "name": "Visa", "tier": "B"},
    "BRK.B": {"sector": "financial", "sub_sector": "insurance", "name": "Berkshire Hathaway", "tier": "C"},
    "XLF": {"sector": "financial", "sub_sector": "sector_etf", "name": "Financial Select Sector SPDR ETF", "tier": "C"},
    "GS": {"sector": "financial", "sub_sector": "investment_bank", "name": "Goldman Sachs", "tier": "C"},
    "CB": {"sector": "financial", "sub_sector": "insurance", "name": "Chubb Limited", "tier": "D"},
    "MA": {"sector": "financial", "sub_sector": "payments", "name": "Mastercard", "tier": "D"},
    "PGR": {"sector": "financial", "sub_sector": "insurance", "name": "Progressive Corporation", "tier": "D"},
    "MCO": {"sector": "financial", "sub_sector": "financial_data", "name": "Moody's Corporation", "tier": "D"},
    "SPGI": {"sector": "financial", "sub_sector": "financial_data", "name": "S&P Global", "tier": "F"},

    # ── Healthcare (7 validated) ─────────────────────────────────────────
    "SYK": {"sector": "healthcare", "sub_sector": "med_devices", "name": "Stryker - Medical Devices", "tier": "C"},
    "JNJ": {"sector": "healthcare", "sub_sector": "diversified", "name": "Johnson & Johnson", "tier": "C"},
    "ABBV": {"sector": "healthcare", "sub_sector": "pharma", "name": "AbbVie", "tier": "D"},
    "XLV": {"sector": "healthcare", "sub_sector": "healthcare_etf", "name": "Health Care Select Sector SPDR", "tier": "F"},
    "UNH": {"sector": "healthcare", "sub_sector": "managed_care", "name": "UnitedHealth Group", "tier": "F"},
    "RXRX": {"sector": "healthcare", "sub_sector": "clinical_biotech", "name": "Recursion Pharmaceuticals", "tier": "F"},
    "VKTX": {"sector": "healthcare", "sub_sector": "clinical_biotech", "name": "Viking Therapeutics", "tier": "F"},

    # ── Consumer Staples & Discretionary (10 validated) ──────────────────
    "WMT": {"sector": "consumer_staples", "sub_sector": "mass_retail", "name": "Walmart", "tier": "A"},
    "CHD": {"sector": "consumer_staples", "sub_sector": "household", "name": "Church & Dwight", "tier": "B"},
    "CL": {"sector": "consumer_staples", "sub_sector": "household", "name": "Colgate-Palmolive", "tier": "B"},
    "KO": {"sector": "consumer_staples", "sub_sector": "beverages", "name": "Coca-Cola", "tier": "B"},
    "COST": {"sector": "consumer_staples", "sub_sector": "retail_growth", "name": "Costco", "tier": "C"},
    "XLY": {"sector": "generic", "sub_sector": "consumer_disc_etf", "name": "Consumer Discretionary SPDR ETF", "tier": "D"},
    "KMB": {"sector": "consumer_staples", "sub_sector": "household", "name": "Kimberly-Clark", "tier": "F"},
    "XLP": {"sector": "consumer_staples", "sub_sector": "staples_etf", "name": "Consumer Staples SPDR ETF", "tier": "F"},
    "PG": {"sector": "consumer_staples", "sub_sector": "household", "name": "Procter & Gamble", "tier": "F"},

    # ── Utilities & Power (11 validated) ─────────────────────────────────
    "VPU": {"sector": "utility", "sub_sector": "utility_etf", "name": "Vanguard Utilities ETF", "tier": "B"},
    "XLU": {"sector": "utility", "sub_sector": "utility_etf", "name": "Utilities Select Sector SPDR ETF", "tier": "C"},
    "VST": {"sector": "utility", "sub_sector": "nuclear_power", "name": "Vistra Corp - Nuclear/Gas Power", "tier": "C"},
    "BEP": {"sector": "utility", "sub_sector": "yieldco", "name": "Brookfield Renewable Partners", "tier": "C"},
    "CWEN": {"sector": "utility", "sub_sector": "yieldco", "name": "Clearway Energy", "tier": "D"},
    "D": {"sector": "utility", "sub_sector": "regulated", "name": "Dominion Energy", "tier": "C"},
    "SO": {"sector": "utility", "sub_sector": "regulated", "name": "Southern Company", "tier": "D"},
    "NEE": {"sector": "utility", "sub_sector": "regulated", "name": "NextEra Energy", "tier": "D"},
    "CEG": {"sector": "utility", "sub_sector": "nuclear_power", "name": "Constellation Energy - Nuclear", "tier": "D"},
    "O": {"sector": "financial", "sub_sector": "reit", "name": "Realty Income REIT", "tier": "D"},
    "AMT": {"sector": "financial", "sub_sector": "reit", "name": "American Tower REIT", "tier": "C"},
    "AWK": {"sector": "utility", "sub_sector": "water", "name": "American Water Works", "tier": "F"},
    "OKLO": {"sector": "utility", "sub_sector": "nuclear_power", "name": "Oklo Inc - Nuclear SMR", "tier": "F"},
}


def get_sector_config(sector: str) -> SectorConfig:
    """Get sector config, fall back to generic."""
    return SECTOR_REGISTRY.get(sector, GENERIC_CONFIG)


def get_symbol_sector(symbol: str) -> Dict:
    """Get symbol metadata. Returns generic defaults if unknown."""
    if symbol in SYMBOL_SECTORS:
        return SYMBOL_SECTORS[symbol]
    return {"sector": "generic", "sub_sector": "unknown", "name": symbol, "tier": "Unknown"}
