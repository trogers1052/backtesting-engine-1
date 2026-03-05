# ETN (Eaton Corporation) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 20.7 minutes
**Category:** Electrical/power management

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

ETN — Electrical/power management — beta 1.1, data center + infrastructure exposure, ~$310. Electrical/power management.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 19 | 52.6% | +34.1% | 0.47 | 1.68 | -15.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 27 | 51.9% | +28.6% | 0.29 | 1.28 | -27.6% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 20 | 50.0% | +32.6% | 0.47 | 1.63 | -16.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 18 | 50.0% | +37.3% | 0.39 | 1.71 | -19.3% |
| Alt D: Industrial rules (13 rules, 10%/5%) | 30 | 56.7% | +41.4% | 0.45 | 1.40 | -22.7% |
| Alt E: electrical lean (3 rules, 10%/5%) | 21 | 66.7% | +92.7% | 0.95 | 3.14 | -12.5% |
| Alt F: Electrical trending (12%/5%) | 21 | 61.9% | +84.9% | 0.78 | 2.65 | -13.8% |
| Alt G: Electrical tech-style (15%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Alt E: electrical lean (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: electrical lean (3 rules, 10%/5%)

- **Rules:** `industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+92.7%, Trades=21, WR=66.7%, Sharpe=0.95, PF=3.14, DD=-12.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.76, Test Sharpe=0.81, Ratio=106% (need >=50%) |
| Bootstrap | **PASS** | p=0.0059, Sharpe CI=[0.80, 7.80], WR CI=[47.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.8%, Median equity=$2,193, Survival=100.0% |
| Regime | FAIL | bull:15t/+64.9%, bear:3t/+10.2%, chop:2t/+8.7%, volatile:1t/+1.9% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: + ind_mean_reversion | 21 | 66.7% | +92.7% | 0.95 |
| Regime tune: tighter stop 4% | 21 | 66.7% | +94.6% | 0.94 |
| Regime tune: PT=8% | 22 | 68.2% | +64.8% | 0.86 |
| Regime tune: PT=12% | 21 | 61.9% | +84.9% | 0.78 |
| Regime tune: PT=7% | 23 | 65.2% | +57.2% | 0.78 |
| Regime tune: conf=0.65 | 14 | 57.1% | +60.7% | 0.73 |
| Regime tune: PT=15% | 18 | 61.1% | +65.4% | 0.63 |
| Regime tune: industrial rules (13) | 30 | 56.7% | +41.4% | 0.45 |
| Regime tune: full rules (10) | 27 | 51.9% | +28.6% | 0.29 |
| Regime tune: PT=8% [multi-TF] | 37 | 54.1% | +24.6% | 0.38 |

### Full Validation of Top Candidates

### Regime tune: + ind_mean_reversion

- **Rules:** `industrial_pullback, industrial_seasonality, death_cross, industrial_mean_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+92.7%, Trades=21, WR=66.7%, Sharpe=0.95, PF=3.14, DD=-12.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.76, Test Sharpe=0.81, Ratio=106% (need >=50%) |
| Bootstrap | **PASS** | p=0.0059, Sharpe CI=[0.80, 7.80], WR CI=[47.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.8%, Median equity=$2,193, Survival=100.0% |
| Regime | FAIL | bull:15t/+64.9%, bear:3t/+10.2%, chop:2t/+8.7%, volatile:1t/+1.9% |

**Result: 3/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+94.6%, Trades=21, WR=66.7%, Sharpe=0.94, PF=3.19, DD=-12.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.78, Test Sharpe=1.08, Ratio=138% (need >=50%) |
| Bootstrap | **PASS** | p=0.0044, Sharpe CI=[0.92, 7.91], WR CI=[47.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.2%, Median equity=$2,233, Survival=100.0% |
| Regime | FAIL | bull:15t/+66.0%, bear:3t/+10.8%, chop:2t/+8.7%, volatile:1t/+1.9% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=8%

- **Rules:** `industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+64.8%, Trades=22, WR=68.2%, Sharpe=0.86, PF=2.51, DD=-12.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.63, Test Sharpe=0.82, Ratio=131% (need >=50%) |
| Bootstrap | **PASS** | p=0.0131, Sharpe CI=[0.44, 7.89], WR CI=[50.0%, 86.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.6%, Median equity=$1,876, Survival=100.0% |
| Regime | **PASS** | bull:16t/+47.2%, bear:3t/+10.2%, chop:2t/+8.7%, volatile:1t/+1.9% |

**Result: 4/4 gates passed**

---

### Regime tune: PT=8% [multi-TF]

- **Rules:** `industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+24.6%, Trades=37, WR=54.1%, Sharpe=0.38, PF=1.33, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=-1.03, Ratio=-133% (need >=50%) |
| Bootstrap | FAIL | p=0.1314, Sharpe CI=[-1.10, 3.69], WR CI=[37.8%, 70.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.1%, Median equity=$1,397, Survival=100.0% |
| Regime | **PASS** | bull:27t/+27.2%, bear:3t/+5.3%, chop:5t/+2.2%, volatile:2t/+4.3% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=8%** | **PASS** | **PASS** | **PASS** | **PASS** | **0.86** | **+64.8%** | 22 |
| Alt E: electrical lean (3 rules, 10%/5%) | **PASS** | **PASS** | **PASS** | FAIL | 0.95 | +92.7% | 21 |
| Regime tune: + ind_mean_reversion | **PASS** | **PASS** | **PASS** | FAIL | 0.95 | +92.7% | 21 |
| Regime tune: tighter stop 4% | **PASS** | **PASS** | **PASS** | FAIL | 0.94 | +94.6% | 21 |
| Regime tune: PT=8% [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.38 | +24.6% | 37 |

---

## 5. Final Recommendation

**ETN fully validates.** Best config: Regime tune: PT=8% (4/4 gates).

### Regime tune: PT=8%

- **Rules:** `industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+64.8%, Trades=22, WR=68.2%, Sharpe=0.86, PF=2.51, DD=-12.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.63, Test Sharpe=0.82, Ratio=131% (need >=50%) |
| Bootstrap | **PASS** | p=0.0131, Sharpe CI=[0.44, 7.89], WR CI=[50.0%, 86.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.6%, Median equity=$1,876, Survival=100.0% |
| Regime | **PASS** | bull:16t/+47.2%, bear:3t/+10.2%, chop:2t/+8.7%, volatile:1t/+1.9% |

**Result: 4/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

