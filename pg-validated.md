# PG (Procter & Gamble) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 24.4 minutes
**Category:** Household products

---

## Methodology

Validate-then-tune approach with daily-only screening for speed, multi-TF re-validation for final config.

### Validation Gates

| Gate | Method | Pass Criteria | Purpose |
|------|--------|---------------|---------|
| Walk-Forward | 70/30 train/test split, 5-day embargo, 10-day purge | Test Sharpe >= 50% of Train Sharpe | Detects parameter overfitting |
| Bootstrap | 10,000 resamples with replacement | p < 0.05 AND Sharpe 95% CI excludes zero | Tests statistical significance |
| Monte Carlo | 10,000 trade-order permutations | Ruin probability < 10% AND P95 drawdown < 40% | Measures worst-case risk |
| Regime | SPY SMA_50/SMA_200 + VIX classification | No single regime contributes >70% of total profit | Detects regime dependency |

---

## 1. Baseline Screening

PG — Consumer staple bellwether — beta 0.15, household products leader. Household products.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 28.6% | -16.6% | -0.93 | 0.36 | -17.4% |
| Alt A: Full general rules (10 rules, 10%/5%) | 8 | 25.0% | -10.7% | -0.62 | 0.52 | -13.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 25.0% | -14.3% | -0.81 | 0.37 | -15.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 28.6% | -15.0% | -0.81 | 0.42 | -15.7% |
| Alt D: Staples rules (13 rules, 10%/5%) | 17 | 47.1% | -12.3% | -0.41 | 0.58 | -18.8% |
| Alt E: household lean (3 rules, 10%/5%) | 14 | 50.0% | -14.7% | -0.68 | 0.52 | -20.7% |
| Alt F: Household tight (6%/3%, conf=0.55) | 18 | 44.4% | -11.6% | -0.44 | 0.59 | -18.2% |
| Alt G: Household moderate (7%/4%) | 21 | 47.6% | -10.0% | -0.28 | 0.76 | -19.1% |

**Best baseline selected for validation: Alt G: Household moderate (7%/4%)**

---

## 2. Full Validation

### Alt G: Household moderate (7%/4%)

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-10.0%, Trades=21, WR=47.6%, Sharpe=-0.28, PF=0.76, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.27, Test Sharpe=-0.06, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6092, Sharpe CI=[-4.19, 2.65], WR CI=[28.6%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.5%, Median equity=$928, Survival=100.0% |
| Regime | FAIL | bull:16t/+8.9%, bear:3t/-4.2%, chop:2t/-9.9% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 18 | 55.6% | -2.3% | -0.13 |
| BS tune: staples rules (13) | 24 | 50.0% | -4.0% | -0.17 |
| BS tune: full rules (10) | 14 | 42.9% | +0.5% | -0.17 |
| WF tune: cooldown=3 | 22 | 50.0% | -6.5% | -0.20 |
| WF tune: conf=0.6 | 20 | 50.0% | -5.1% | -0.21 |
| WF tune: ATR stops x2.5 | 21 | 47.6% | -6.5% | -0.24 |
| WF tune: conf=0.45 | 21 | 47.6% | -10.0% | -0.28 |
| WF tune: conf=0.55 | 21 | 47.6% | -10.0% | -0.28 |
| BS tune: conf=0.4 | 21 | 47.6% | -10.0% | -0.28 |
| WF tune: conf=0.65 | 16 | 37.5% | -12.6% | -0.28 |
| WF tune: PT=6% | 21 | 47.6% | -11.3% | -0.32 |
| WF tune: PT=12% | 15 | 46.7% | -10.9% | -0.35 |
| WF tune: PT=15% | 13 | 53.8% | -8.2% | -0.36 |
| BS tune: household rules | 18 | 44.4% | -15.1% | -0.47 |
| WF tune: PT=8% | 17 | 41.2% | -20.8% | -0.82 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 7 bars

**Performance:** Return=-2.3%, Trades=18, WR=55.6%, Sharpe=-0.13, PF=0.93, DD=-15.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.17, Test Sharpe=-0.06, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4391, Sharpe CI=[-3.44, 3.88], WR CI=[33.3%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$1,023, Survival=100.0% |
| Regime | FAIL | bull:14t/+14.2%, bear:3t/-3.5%, chop:1t/-6.4% |

**Result: 1/4 gates passed**

---

### BS tune: staples rules (13)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=-4.0%, Trades=24, WR=50.0%, Sharpe=-0.17, PF=0.90, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.17, Test Sharpe=-0.06, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4538, Sharpe CI=[-3.08, 3.07], WR CI=[29.2%, 70.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.0%, Median equity=$1,016, Survival=100.0% |
| Regime | FAIL | bull:18t/+15.7%, bear:4t/-1.2%, chop:2t/-10.5% |

**Result: 1/4 gates passed**

---

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.5%, Trades=14, WR=42.9%, Sharpe=-0.17, PF=1.02, DD=-10.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.51, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4480, Sharpe CI=[-4.12, 4.16], WR CI=[21.4%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.6%, Median equity=$1,012, Survival=100.0% |
| Regime | FAIL | bull:12t/+13.4%, bear:1t/-4.6%, chop:1t/-5.5% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=7** | FAIL | FAIL | **PASS** | FAIL | **-0.13** | **-2.3%** | 18 |
| BS tune: staples rules (13) | FAIL | FAIL | **PASS** | FAIL | -0.17 | -4.0% | 24 |
| BS tune: full rules (10) | FAIL | FAIL | **PASS** | FAIL | -0.17 | +0.5% | 14 |
| Alt G: Household moderate (7%/4%) | FAIL | FAIL | **PASS** | FAIL | -0.28 | -10.0% | 21 |

---

## 5. Final Recommendation

**PG partially validates.** Best config: WF tune: cooldown=7 (1/4 gates).

### WF tune: cooldown=7

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 7 bars

**Performance:** Return=-2.3%, Trades=18, WR=55.6%, Sharpe=-0.13, PF=0.93, DD=-15.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.17, Test Sharpe=-0.06, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4391, Sharpe CI=[-3.44, 3.88], WR CI=[33.3%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$1,023, Survival=100.0% |
| Regime | FAIL | bull:14t/+14.2%, bear:3t/-3.5%, chop:1t/-6.4% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

