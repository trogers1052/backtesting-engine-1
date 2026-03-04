# GOOGL (Alphabet/Google) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 48.5 minutes
**Category:** Mega-cap tech

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

GOOGL — Search ads, YouTube, Google Cloud, AI — mega-cap mean-reverter. Mega-cap tech.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 17 | 64.7% | +110.5% | 1.03 | 3.09 | -15.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 23 | 52.2% | +87.9% | 0.70 | 1.98 | -23.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 19 | 52.6% | +94.4% | 0.80 | 2.98 | -16.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 14 | 57.1% | +92.1% | 0.81 | 3.17 | -15.1% |
| Alt D: Tech rules (13 rules, 10%/5%) | 30 | 50.0% | +79.5% | 0.59 | 1.79 | -22.2% |
| Alt E: mega_cap rules (3 rules, 10%/5%) | 24 | 54.2% | +72.9% | 0.72 | 2.06 | -17.9% |
| Alt F: Mega-cap balanced (8%/4%, conf=0.55) | 25 | 56.0% | +76.9% | 0.69 | 2.08 | -11.2% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+110.5%, Trades=17, WR=64.7%, Sharpe=1.03, PF=3.09, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.88, Test Sharpe=0.90, Ratio=103% (need >=50%) |
| Bootstrap | **PASS** | p=0.0095, Sharpe CI=[0.75, 10.50], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$2,362, Survival=100.0% |
| Regime | FAIL | bull:17t/+95.2% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: conf=0.65 | 17 | 64.7% | +110.0% | 1.04 |
| Regime tune: + tech_ema_pullback | 17 | 64.7% | +110.5% | 1.03 |
| Regime tune: + tech_mean_reversion | 17 | 64.7% | +110.5% | 1.03 |
| Regime tune: PT=15% | 12 | 66.7% | +149.3% | 0.92 |
| Regime tune: PT=12% | 14 | 57.1% | +92.1% | 0.81 |
| Regime tune: tighter stop 4% | 19 | 52.6% | +94.4% | 0.80 |
| Regime tune: full rules (10) | 23 | 52.2% | +87.9% | 0.70 |
| Regime tune: tech rules (13) | 30 | 50.0% | +79.5% | 0.59 |
| Regime tune: conf=0.65 [multi-TF] | 17 | 58.8% | +72.3% | 0.98 |

### Full Validation of Top Candidates

### Regime tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+110.0%, Trades=17, WR=64.7%, Sharpe=1.04, PF=3.04, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.88, Test Sharpe=0.90, Ratio=103% (need >=50%) |
| Bootstrap | **PASS** | p=0.0088, Sharpe CI=[0.79, 10.53], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$2,374, Survival=100.0% |
| Regime | FAIL | bull:17t/+95.7% |

**Result: 3/4 gates passed**

---

### Regime tune: + tech_ema_pullback

- **Rules:** `trend_continuation, seasonality, death_cross, tech_ema_pullback`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+110.5%, Trades=17, WR=64.7%, Sharpe=1.03, PF=3.09, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.88, Test Sharpe=0.90, Ratio=103% (need >=50%) |
| Bootstrap | **PASS** | p=0.0095, Sharpe CI=[0.75, 10.50], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$2,362, Survival=100.0% |
| Regime | FAIL | bull:17t/+95.2% |

**Result: 3/4 gates passed**

---

### Regime tune: + tech_mean_reversion

- **Rules:** `trend_continuation, seasonality, death_cross, tech_mean_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+110.5%, Trades=17, WR=64.7%, Sharpe=1.03, PF=3.09, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.88, Test Sharpe=0.90, Ratio=103% (need >=50%) |
| Bootstrap | **PASS** | p=0.0095, Sharpe CI=[0.75, 10.50], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$2,362, Survival=100.0% |
| Regime | FAIL | bull:17t/+95.2% |

**Result: 3/4 gates passed**

---

### Regime tune: conf=0.65 [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+72.3%, Trades=17, WR=58.8%, Sharpe=0.98, PF=2.76, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.74, Test Sharpe=0.97, Ratio=130% (need >=50%) |
| Bootstrap | **PASS** | p=0.0140, Sharpe CI=[0.36, 9.07], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.8%, Median equity=$1,911, Survival=100.0% |
| Regime | FAIL | bull:17t/+71.0% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: conf=0.65** | **PASS** | **PASS** | **PASS** | FAIL | **1.04** | **+110.0%** | 17 |
| Lean 3 rules baseline (10%/5%, conf=0.50) | **PASS** | **PASS** | **PASS** | FAIL | 1.03 | +110.5% | 17 |
| Regime tune: + tech_ema_pullback | **PASS** | **PASS** | **PASS** | FAIL | 1.03 | +110.5% | 17 |
| Regime tune: + tech_mean_reversion | **PASS** | **PASS** | **PASS** | FAIL | 1.03 | +110.5% | 17 |
| Regime tune: conf=0.65 [multi-TF] | **PASS** | **PASS** | **PASS** | FAIL | 0.98 | +72.3% | 17 |

---

## 5. Final Recommendation

**GOOGL partially validates.** Best config: Regime tune: conf=0.65 (3/4 gates).

### Regime tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+110.0%, Trades=17, WR=64.7%, Sharpe=1.04, PF=3.04, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.88, Test Sharpe=0.90, Ratio=103% (need >=50%) |
| Bootstrap | **PASS** | p=0.0088, Sharpe CI=[0.79, 10.53], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$2,374, Survival=100.0% |
| Regime | FAIL | bull:17t/+95.7% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

