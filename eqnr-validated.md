# EQNR (Equinor) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 29.8 minutes
**Category:** Large-cap international oil

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

EQNR — Norwegian state oil company — North Sea, offshore wind, LNG. Large-cap international oil.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 6 | 16.7% | -17.5% | -0.58 | 0.02 | -26.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 8 | 12.5% | -26.5% | -0.98 | 0.04 | -38.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 6 | 16.7% | -15.8% | -0.55 | 0.02 | -25.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 6 | 16.7% | -17.5% | -0.58 | 0.02 | -26.5% |
| Alt D: Energy rules (14 rules, 10%/5%) | 17 | 35.3% | -2.6% | -0.11 | 0.89 | -29.9% |
| Alt E: integrated sector rules (3 rules, 10%/5%) | 14 | 42.9% | +15.9% | 0.22 | 1.20 | -15.3% |

**Best baseline selected for validation: Alt E: integrated sector rules (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: integrated sector rules (3 rules, 10%/5%)

- **Rules:** `energy_mean_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+15.9%, Trades=14, WR=42.9%, Sharpe=0.22, PF=1.20, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.59, Test Sharpe=0.55, Ratio=94% (need >=50%) |
| Bootstrap | FAIL | p=0.2377, Sharpe CI=[-2.55, 5.85], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.5%, Median equity=$1,201, Survival=100.0% |
| Regime | FAIL | bull:9t/+30.0%, bear:2t/-0.8%, chop:2t/-0.1%, volatile:1t/-6.2% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 14 | 42.9% | +16.4% | 0.23 |
| BS tune: conf=0.4 | 14 | 42.9% | +15.9% | 0.22 |
| BS tune: conf=0.45 | 14 | 42.9% | +15.9% | 0.22 |
| BS tune: conf=0.55 | 14 | 42.9% | +15.9% | 0.22 |
| BS tune: + energy_momentum | 14 | 42.9% | +15.9% | 0.22 |
| Regime tune: conf=0.65 | 14 | 35.7% | +12.5% | 0.17 |
| Regime tune: PT=12% | 13 | 38.5% | +12.1% | 0.15 |
| BS tune: cooldown=7 | 12 | 41.7% | +7.4% | 0.07 |
| BS tune: cooldown=3 | 14 | 35.7% | -0.1% | -0.06 |
| BS tune: energy rules (14) | 17 | 35.3% | -2.6% | -0.11 |
| Regime tune: PT=15% | 13 | 30.8% | -5.4% | -0.13 |
| BS tune: full rules (10) | 8 | 12.5% | -26.5% | -0.98 |
| Regime tune: tighter stop 4% [multi-TF] | 33 | 51.5% | -16.0% | -1.24 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `energy_mean_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+16.4%, Trades=14, WR=42.9%, Sharpe=0.23, PF=1.22, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.59, Test Sharpe=0.58, Ratio=98% (need >=50%) |
| Bootstrap | FAIL | p=0.2326, Sharpe CI=[-2.54, 5.91], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.2%, Median equity=$1,207, Survival=100.0% |
| Regime | FAIL | bull:9t/+30.0%, bear:2t/-0.8%, chop:2t/+0.3%, volatile:1t/-6.2% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `energy_mean_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+15.9%, Trades=14, WR=42.9%, Sharpe=0.22, PF=1.20, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.59, Test Sharpe=0.55, Ratio=94% (need >=50%) |
| Bootstrap | FAIL | p=0.2377, Sharpe CI=[-2.55, 5.85], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.5%, Median equity=$1,201, Survival=100.0% |
| Regime | FAIL | bull:9t/+30.0%, bear:2t/-0.8%, chop:2t/-0.1%, volatile:1t/-6.2% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `energy_mean_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+15.9%, Trades=14, WR=42.9%, Sharpe=0.22, PF=1.20, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.59, Test Sharpe=0.55, Ratio=94% (need >=50%) |
| Bootstrap | FAIL | p=0.2377, Sharpe CI=[-2.55, 5.85], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.5%, Median equity=$1,201, Survival=100.0% |
| Regime | FAIL | bull:9t/+30.0%, bear:2t/-0.8%, chop:2t/-0.1%, volatile:1t/-6.2% |

**Result: 2/4 gates passed**

---

### Regime tune: tighter stop 4% [multi-TF]

- **Rules:** `energy_mean_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-16.0%, Trades=33, WR=51.5%, Sharpe=-1.24, PF=0.55, DD=-23.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.16, Test Sharpe=-1.68, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7763, Sharpe CI=[-3.45, 1.60], WR CI=[36.4%, 69.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.3%, Median equity=$887, Survival=100.0% |
| Regime | FAIL | bull:25t/+1.1%, bear:5t/-9.6%, chop:3t/-2.2% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: tighter stop 4%** | **PASS** | FAIL | **PASS** | FAIL | **0.23** | **+16.4%** | 14 |
| Alt E: integrated sector rules (3 rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.22 | +15.9% | 14 |
| BS tune: conf=0.4 | **PASS** | FAIL | **PASS** | FAIL | 0.22 | +15.9% | 14 |
| BS tune: conf=0.45 | **PASS** | FAIL | **PASS** | FAIL | 0.22 | +15.9% | 14 |
| Regime tune: tighter stop 4% [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -1.24 | -16.0% | 33 |

---

## 5. Final Recommendation

**EQNR partially validates.** Best config: Regime tune: tighter stop 4% (2/4 gates).

### Regime tune: tighter stop 4%

- **Rules:** `energy_mean_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+16.4%, Trades=14, WR=42.9%, Sharpe=0.23, PF=1.22, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.59, Test Sharpe=0.58, Ratio=98% (need >=50%) |
| Bootstrap | FAIL | p=0.2326, Sharpe CI=[-2.54, 5.91], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.2%, Median equity=$1,207, Survival=100.0% |
| Regime | FAIL | bull:9t/+30.0%, bear:2t/-0.8%, chop:2t/+0.3%, volatile:1t/-6.2% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

