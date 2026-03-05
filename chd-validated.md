# CHD (Church & Dwight) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 29.1 minutes
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

CHD — Household products — beta 0.02-0.47, Arm & Hammer, ultra-low vol. Household products.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 28.6% | -0.5% | -0.69 | 0.98 | -16.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 12 | 41.7% | -4.4% | -0.81 | 0.85 | -16.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 25.0% | -1.4% | -0.49 | 0.93 | -17.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 28.6% | +7.1% | 0.06 | 1.37 | -14.3% |
| Alt D: Staples rules (13 rules, 10%/5%) | 23 | 56.5% | +26.5% | 0.79 | 1.69 | -15.5% |
| Alt E: household lean (3 rules, 10%/5%) | 16 | 56.2% | +23.5% | 0.38 | 1.76 | -20.4% |
| Alt F: Household tight (6%/3%, conf=0.55) | 20 | 55.0% | +20.9% | 0.35 | 1.66 | -16.6% |
| Alt G: Household moderate (7%/4%) | 19 | 63.2% | +39.0% | 1.15 | 2.82 | -11.9% |

**Best baseline selected for validation: Alt G: Household moderate (7%/4%)**

---

## 2. Full Validation

### Alt G: Household moderate (7%/4%)

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+39.0%, Trades=19, WR=63.2%, Sharpe=1.15, PF=2.82, DD=-11.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.92, Test Sharpe=0.49, Ratio=53% (need >=50%) |
| Bootstrap | **PASS** | p=0.0144, Sharpe CI=[0.35, 7.39], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.0%, Median equity=$1,489, Survival=100.0% |
| Regime | FAIL | bull:15t/+30.6%, bear:3t/+12.6%, chop:1t/-1.2% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=8% | 20 | 60.0% | +32.3% | 0.80 |
| Regime tune: conf=0.65 | 16 | 56.2% | +27.2% | 0.76 |
| Regime tune: PT=12% | 17 | 58.8% | +28.1% | 0.74 |
| Regime tune: PT=15% | 15 | 60.0% | +32.9% | 0.73 |
| Regime tune: staples rules (13) | 25 | 56.0% | +33.6% | 0.71 |
| Regime tune: full rules (10) | 13 | 46.2% | +8.5% | 0.17 |
| Alt G: Household moderate (7%/4%) [multi-TF] | 43 | 53.5% | -6.5% | -0.29 |

### Full Validation of Top Candidates

### Regime tune: PT=8%

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+32.3%, Trades=20, WR=60.0%, Sharpe=0.80, PF=2.27, DD=-13.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.56, Test Sharpe=0.49, Ratio=86% (need >=50%) |
| Bootstrap | FAIL | p=0.0356, Sharpe CI=[-0.25, 6.44], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-13.8%, Median equity=$1,406, Survival=100.0% |
| Regime | **PASS** | bull:16t/+24.9%, bear:3t/+12.6%, chop:1t/-1.2% |

**Result: 3/4 gates passed**

---

### Regime tune: conf=0.65

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.65
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+27.2%, Trades=16, WR=56.2%, Sharpe=0.76, PF=2.12, DD=-12.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.42, Test Sharpe=0.50, Ratio=120% (need >=50%) |
| Bootstrap | FAIL | p=0.0571, Sharpe CI=[-0.76, 7.36], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.9%, Median equity=$1,348, Survival=100.0% |
| Regime | **PASS** | bull:13t/+16.1%, bear:2t/+17.1%, chop:1t/-1.2% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=12%

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+28.1%, Trades=17, WR=58.8%, Sharpe=0.74, PF=2.24, DD=-11.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.01, Test Sharpe=0.49, Ratio=48% (need >=50%) |
| Bootstrap | FAIL | p=0.0582, Sharpe CI=[-0.80, 6.00], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-13.3%, Median equity=$1,354, Survival=100.0% |
| Regime | **PASS** | bull:14t/+15.4%, bear:3t/+17.3% |

**Result: 2/4 gates passed**

---

### Alt G: Household moderate (7%/4%) [multi-TF]

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-6.5%, Trades=43, WR=53.5%, Sharpe=-0.29, PF=0.90, DD=-30.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.65, Test Sharpe=0.02, Ratio=3% (need >=50%) |
| Bootstrap | FAIL | p=0.4547, Sharpe CI=[-2.08, 2.34], WR CI=[41.9%, 69.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.2%, Median equity=$1,017, Survival=100.0% |
| Regime | **PASS** | bull:31t/+8.8%, bear:6t/+2.6%, chop:4t/-9.5%, volatile:2t/+2.9% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt G: Household moderate (7%/4%)** | **PASS** | **PASS** | **PASS** | FAIL | **1.15** | **+39.0%** | 19 |
| Regime tune: PT=8% | **PASS** | FAIL | **PASS** | **PASS** | 0.80 | +32.3% | 20 |
| Regime tune: conf=0.65 | **PASS** | FAIL | **PASS** | **PASS** | 0.76 | +27.2% | 16 |
| Regime tune: PT=12% | FAIL | FAIL | **PASS** | **PASS** | 0.74 | +28.1% | 17 |
| Alt G: Household moderate (7%/4%) [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | -0.29 | -6.5% | 43 |

---

## 5. Final Recommendation

**CHD partially validates.** Best config: Alt G: Household moderate (7%/4%) (3/4 gates).

### Alt G: Household moderate (7%/4%)

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+39.0%, Trades=19, WR=63.2%, Sharpe=1.15, PF=2.82, DD=-11.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.92, Test Sharpe=0.49, Ratio=53% (need >=50%) |
| Bootstrap | **PASS** | p=0.0144, Sharpe CI=[0.35, 7.39], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.0%, Median equity=$1,489, Survival=100.0% |
| Regime | FAIL | bull:15t/+30.6%, bear:3t/+12.6%, chop:1t/-1.2% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

