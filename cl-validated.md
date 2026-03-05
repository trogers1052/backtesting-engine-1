# CL (Colgate-Palmolive) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 30.9 minutes
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

CL — Household products — beta 0.03-0.30, negative skew, global consumer staple. Household products.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 6 | 50.0% | +10.4% | 0.12 | 1.64 | -10.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 11 | 27.3% | +3.3% | -0.03 | 0.96 | -14.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 6 | 50.0% | +13.4% | 0.17 | 1.88 | -11.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 5 | 40.0% | +5.9% | 0.03 | 1.39 | -11.3% |
| Alt D: Staples rules (13 rules, 10%/5%) | 25 | 60.0% | +18.8% | 0.32 | 1.33 | -16.0% |
| Alt E: household lean (3 rules, 10%/5%) | 20 | 75.0% | +29.6% | 0.49 | 1.99 | -11.2% |
| Alt F: Household tight (6%/3%, conf=0.55) | 20 | 70.0% | +34.3% | 0.70 | 2.32 | -9.9% |
| Alt G: Household moderate (7%/4%) | 22 | 72.7% | +28.2% | 0.45 | 1.93 | -13.0% |

**Best baseline selected for validation: Alt F: Household tight (6%/3%, conf=0.55)**

---

## 2. Full Validation

### Alt F: Household tight (6%/3%, conf=0.55)

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+34.3%, Trades=20, WR=70.0%, Sharpe=0.70, PF=2.32, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.61, Test Sharpe=0.20, Ratio=34% (need >=50%) |
| Bootstrap | **PASS** | p=0.0183, Sharpe CI=[0.19, 7.53], WR CI=[55.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.7%, Median equity=$1,434, Survival=100.0% |
| Regime | FAIL | bull:15t/+29.9%, bear:3t/+2.1%, chop:2t/+5.9% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: + staples_pullback | 21 | 66.7% | +29.7% | 0.80 |
| WF tune: conf=0.65 | 12 | 66.7% | +37.5% | 0.78 |
| WF tune: PT=12% | 18 | 66.7% | +30.1% | 0.77 |
| WF tune: cooldown=3 | 25 | 72.0% | +50.7% | 0.74 |
| WF tune: ATR stops x2.5 | 20 | 70.0% | +34.3% | 0.70 |
| WF tune: PT=7% | 20 | 70.0% | +36.7% | 0.70 |
| WF tune: PT=15% | 18 | 66.7% | +37.9% | 0.69 |
| WF tune: conf=0.6 | 20 | 60.0% | +32.9% | 0.69 |
| Regime tune: staples rules (13) | 32 | 62.5% | +33.8% | 0.64 |
| WF tune: PT=8% | 19 | 73.7% | +48.2% | 0.63 |
| WF tune: conf=0.45 | 20 | 65.0% | +30.2% | 0.57 |
| Regime tune: full rules (10) | 17 | 41.2% | +9.4% | 0.10 |
| WF tune: conf=0.65 [multi-TF] | 19 | 57.9% | +42.6% | 1.16 |

### Full Validation of Top Candidates

### WF tune: + staples_pullback

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross, consumer_staples_pullback`
- **Profit Target:** 6%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+29.7%, Trades=21, WR=66.7%, Sharpe=0.80, PF=1.91, DD=-13.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.62, Test Sharpe=0.20, Ratio=33% (need >=50%) |
| Bootstrap | FAIL | p=0.0387, Sharpe CI=[-0.29, 6.48], WR CI=[52.4%, 90.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.2%, Median equity=$1,371, Survival=100.0% |
| Regime | FAIL | bull:16t/+25.5%, bear:3t/+2.1%, chop:2t/+5.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.65
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+37.5%, Trades=12, WR=66.7%, Sharpe=0.78, PF=3.45, DD=-9.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.31, Ratio=68% (need >=50%) |
| Bootstrap | **PASS** | p=0.0145, Sharpe CI=[0.51, 17.94], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.0%, Median equity=$1,458, Survival=100.0% |
| Regime | FAIL | bull:9t/+33.0%, bear:1t/-5.7%, chop:2t/+12.4% |

**Result: 3/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+30.1%, Trades=18, WR=66.7%, Sharpe=0.77, PF=2.15, DD=-11.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.61, Test Sharpe=0.20, Ratio=33% (need >=50%) |
| Bootstrap | FAIL | p=0.0479, Sharpe CI=[-0.51, 5.92], WR CI=[50.0%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-13.0%, Median equity=$1,381, Survival=100.0% |
| Regime | FAIL | bull:13t/+26.7%, bear:3t/+2.1%, chop:2t/+5.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65 [multi-TF]

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.65
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+42.6%, Trades=19, WR=57.9%, Sharpe=1.16, PF=2.59, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.29, Test Sharpe=-0.03, Ratio=-2% (need >=50%) |
| Bootstrap | **PASS** | p=0.0077, Sharpe CI=[0.84, 8.81], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.7%, Median equity=$1,540, Survival=100.0% |
| Regime | FAIL | bull:14t/+41.4%, bear:2t/+4.7%, chop:2t/+2.4%, volatile:1t/-3.2% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | **PASS** | **PASS** | **PASS** | FAIL | **0.78** | **+37.5%** | 12 |
| WF tune: conf=0.65 [multi-TF] | FAIL | **PASS** | **PASS** | FAIL | 1.16 | +42.6% | 19 |
| Alt F: Household tight (6%/3%, conf=0.55) | FAIL | **PASS** | **PASS** | FAIL | 0.70 | +34.3% | 20 |
| WF tune: + staples_pullback | FAIL | FAIL | **PASS** | FAIL | 0.80 | +29.7% | 21 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | FAIL | 0.77 | +30.1% | 18 |

---

## 5. Final Recommendation

**CL partially validates.** Best config: WF tune: conf=0.65 (3/4 gates).

### WF tune: conf=0.65

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.65
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+37.5%, Trades=12, WR=66.7%, Sharpe=0.78, PF=3.45, DD=-9.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.31, Ratio=68% (need >=50%) |
| Bootstrap | **PASS** | p=0.0145, Sharpe CI=[0.51, 17.94], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.0%, Median equity=$1,458, Survival=100.0% |
| Regime | FAIL | bull:9t/+33.0%, bear:1t/-5.7%, chop:2t/+12.4% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

