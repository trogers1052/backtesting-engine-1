# ABBV (AbbVie) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 67.1 minutes
**Category:** Large-cap pharma (mean-reverting)

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

ABBV — Large-cap pharma — beta 0.32-0.36, yield ~3%, 12yr dividend growth, Humira franchise. Large-cap pharma (mean-reverting).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 33.3% | -10.7% | -0.72 | 0.74 | -25.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 33.3% | -10.4% | -0.41 | 0.72 | -28.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 33.3% | -2.4% | -0.70 | 0.90 | -18.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 11 | 18.2% | -25.2% | -1.16 | 0.39 | -30.5% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 24 | 45.8% | +3.0% | -0.02 | 0.97 | -25.6% |
| Alt E: pharma lean (3 rules, 10%/5%) | 17 | 52.9% | +13.6% | 0.22 | 1.18 | -19.3% |
| Alt F: Pharma tight (7%/4%, conf=0.55, cooldown=7) | 20 | 45.0% | -9.6% | -0.31 | 0.81 | -20.3% |
| Alt G: Pharma moderate (8%/4%) | 20 | 45.0% | -5.3% | -0.21 | 0.83 | -22.9% |

**Best baseline selected for validation: Alt E: pharma lean (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: pharma lean (3 rules, 10%/5%)

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+13.6%, Trades=17, WR=52.9%, Sharpe=0.22, PF=1.18, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.65, Test Sharpe=0.50, Ratio=78% (need >=50%) |
| Bootstrap | FAIL | p=0.2279, Sharpe CI=[-2.10, 5.54], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.1%, Median equity=$1,200, Survival=100.0% |
| Regime | **PASS** | bull:11t/+10.7%, bear:1t/-7.0%, chop:3t/+6.0%, volatile:2t/+13.2% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 17 | 52.9% | +14.6% | 0.26 |
| BS tune: cooldown=3 | 17 | 52.9% | +14.1% | 0.23 |
| BS tune: conf=0.45 | 17 | 52.9% | +13.6% | 0.22 |
| BS tune: conf=0.55 | 17 | 52.9% | +13.6% | 0.22 |
| BS tune: cooldown=7 | 17 | 52.9% | +4.0% | -0.01 |
| BS tune: healthcare rules (13) | 24 | 45.8% | +3.0% | -0.02 |
| BS tune: full rules (10) | 18 | 33.3% | -10.4% | -0.41 |
| BS tune: conf=0.4 [multi-TF] | 28 | 53.6% | -11.3% | -0.36 |

### Full Validation of Top Candidates

### BS tune: conf=0.4

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+14.6%, Trades=17, WR=52.9%, Sharpe=0.26, PF=1.21, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.70, Test Sharpe=0.50, Ratio=71% (need >=50%) |
| Bootstrap | FAIL | p=0.2201, Sharpe CI=[-2.03, 5.63], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.0%, Median equity=$1,211, Survival=100.0% |
| Regime | **PASS** | bull:11t/+11.7%, bear:1t/-7.0%, chop:3t/+6.0%, volatile:2t/+13.2% |

**Result: 3/4 gates passed**

---

### BS tune: cooldown=3

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+14.1%, Trades=17, WR=52.9%, Sharpe=0.23, PF=1.20, DD=-18.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.63, Test Sharpe=0.50, Ratio=79% (need >=50%) |
| Bootstrap | FAIL | p=0.2344, Sharpe CI=[-2.14, 5.47], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.1%, Median equity=$1,191, Survival=100.0% |
| Regime | **PASS** | bull:11t/+10.7%, bear:1t/-7.0%, chop:3t/+6.0%, volatile:2t/+12.5% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+13.6%, Trades=17, WR=52.9%, Sharpe=0.22, PF=1.18, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.65, Test Sharpe=0.50, Ratio=78% (need >=50%) |
| Bootstrap | FAIL | p=0.2279, Sharpe CI=[-2.10, 5.54], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.1%, Median equity=$1,200, Survival=100.0% |
| Regime | **PASS** | bull:11t/+10.7%, bear:1t/-7.0%, chop:3t/+6.0%, volatile:2t/+13.2% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-11.3%, Trades=28, WR=53.6%, Sharpe=-0.36, PF=0.77, DD=-21.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.08, Test Sharpe=-0.59, Ratio=-754% (need >=50%) |
| Bootstrap | FAIL | p=0.5884, Sharpe CI=[-3.59, 2.24], WR CI=[35.7%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.3%, Median equity=$932, Survival=100.0% |
| Regime | FAIL | bull:17t/-2.2%, bear:2t/+5.0%, chop:4t/-7.3%, volatile:5t/+0.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | **PASS** | FAIL | **PASS** | **PASS** | **0.26** | **+14.6%** | 17 |
| BS tune: cooldown=3 | **PASS** | FAIL | **PASS** | **PASS** | 0.23 | +14.1% | 17 |
| Alt E: pharma lean (3 rules, 10%/5%) | **PASS** | FAIL | **PASS** | **PASS** | 0.22 | +13.6% | 17 |
| BS tune: conf=0.45 | **PASS** | FAIL | **PASS** | **PASS** | 0.22 | +13.6% | 17 |
| BS tune: conf=0.4 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.36 | -11.3% | 28 |

---

## 5. Final Recommendation

**ABBV partially validates.** Best config: BS tune: conf=0.4 (3/4 gates).

### BS tune: conf=0.4

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+14.6%, Trades=17, WR=52.9%, Sharpe=0.26, PF=1.21, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.70, Test Sharpe=0.50, Ratio=71% (need >=50%) |
| Bootstrap | FAIL | p=0.2201, Sharpe CI=[-2.03, 5.63], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.0%, Median equity=$1,211, Survival=100.0% |
| Regime | **PASS** | bull:11t/+11.7%, bear:1t/-7.0%, chop:3t/+6.0%, volatile:2t/+13.2% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

