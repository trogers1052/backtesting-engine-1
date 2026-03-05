# XLV (Health Care Select Sector SPDR) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 69.4 minutes
**Category:** Healthcare sector ETF

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

XLV — Large-cap healthcare ETF — beta 0.75, diversified across pharma/devices/managed care. Healthcare sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 28.6% | +5.7% | 0.01 | 1.14 | -10.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 9 | 11.1% | -7.9% | -0.39 | 0.31 | -17.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 7 | 28.6% | +8.8% | 0.11 | 1.44 | -10.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 6 | 16.7% | +4.5% | -0.03 | 0.65 | -10.7% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 20 | 35.0% | -11.0% | -0.54 | 0.47 | -22.4% |
| Alt E: healthcare_etf lean (4 rules, 10%/5%) | 19 | 42.1% | -3.2% | -0.31 | 0.80 | -18.4% |
| Alt F: ETF tight (6%/3%, conf=0.55) | 21 | 42.9% | -2.1% | -0.32 | 0.87 | -16.7% |
| Alt G: ETF moderate (8%/4%) | 21 | 38.1% | -9.1% | -0.48 | 0.63 | -21.8% |

**Best baseline selected for validation: Alt E: healthcare_etf lean (4 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: healthcare_etf lean (4 rules, 10%/5%)

- **Rules:** `healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-3.2%, Trades=19, WR=42.1%, Sharpe=-0.31, PF=0.80, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.30, Test Sharpe=0.79, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4873, Sharpe CI=[-4.24, 3.19], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.5%, Median equity=$1,001, Survival=100.0% |
| Regime | **PASS** | bull:17t/+5.0%, bear:1t/-5.2%, volatile:1t/+2.4% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 16 | 43.8% | +9.6% | 0.11 |
| BS tune: conf=0.4 | 19 | 47.4% | -3.2% | -0.26 |
| WF tune: ATR stops x2.5 | 19 | 42.1% | -0.4% | -0.28 |
| WF tune: conf=0.45 | 19 | 42.1% | -2.7% | -0.29 |
| WF tune: conf=0.55 | 19 | 42.1% | -3.2% | -0.31 |
| WF tune: conf=0.6 | 19 | 42.1% | -3.2% | -0.31 |
| WF tune: cooldown=7 | 17 | 41.2% | -2.8% | -0.32 |
| WF tune: PT=12% | 18 | 38.9% | -3.7% | -0.34 |
| BS tune: full rules (10) | 9 | 11.1% | -7.9% | -0.39 |
| WF tune: PT=15% | 16 | 43.8% | -5.2% | -0.41 |
| WF tune: cooldown=3 | 19 | 42.1% | -6.5% | -0.45 |
| WF tune: PT=8% | 19 | 42.1% | -7.0% | -0.49 |
| WF tune: PT=7% | 19 | 42.1% | -7.0% | -0.49 |
| BS tune: healthcare rules (13) | 20 | 35.0% | -11.0% | -0.54 |
| WF tune: PT=6% | 21 | 42.9% | -8.6% | -0.57 |
| BS tune: conf=0.4 [multi-TF] | 36 | 36.1% | -6.9% | -0.31 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+9.6%, Trades=16, WR=43.8%, Sharpe=0.11, PF=1.21, DD=-16.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.22, Test Sharpe=0.77, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2401, Sharpe CI=[-2.65, 4.91], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:14t/+16.5%, bear:1t/-5.2%, volatile:1t/+4.5% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-3.2%, Trades=19, WR=47.4%, Sharpe=-0.26, PF=0.81, DD=-18.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.32, Test Sharpe=0.79, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4849, Sharpe CI=[-3.99, 3.24], WR CI=[31.6%, 73.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.0%, Median equity=$1,001, Survival=100.0% |
| Regime | **PASS** | bull:17t/+5.1%, bear:1t/-5.2%, volatile:1t/+2.4% |

**Result: 2/4 gates passed**

---

### WF tune: ATR stops x2.5

- **Rules:** `healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=-0.4%, Trades=19, WR=42.1%, Sharpe=-0.28, PF=0.88, DD=-15.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=0.80, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4043, Sharpe CI=[-3.87, 3.46], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.5%, Median equity=$1,046, Survival=100.0% |
| Regime | FAIL | bull:16t/+12.2%, bear:2t/-8.3%, volatile:1t/+2.4% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-6.9%, Trades=36, WR=36.1%, Sharpe=-0.31, PF=0.71, DD=-21.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.40, Test Sharpe=0.80, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5494, Sharpe CI=[-3.01, 1.98], WR CI=[30.6%, 63.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.1%, Median equity=$976, Survival=100.0% |
| Regime | FAIL | bull:29t/+3.4%, bear:3t/-4.3%, chop:1t/+0.0%, volatile:3t/+0.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | FAIL | FAIL | **PASS** | **PASS** | **-0.26** | **-3.2%** | 19 |
| Alt E: healthcare_etf lean (4 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | -0.31 | -3.2% | 19 |
| WF tune: conf=0.65 | FAIL | FAIL | **PASS** | FAIL | 0.11 | +9.6% | 16 |
| WF tune: ATR stops x2.5 | FAIL | FAIL | **PASS** | FAIL | -0.28 | -0.4% | 19 |
| BS tune: conf=0.4 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.31 | -6.9% | 36 |

---

## 5. Final Recommendation

**XLV partially validates.** Best config: BS tune: conf=0.4 (2/4 gates).

### BS tune: conf=0.4

- **Rules:** `healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-3.2%, Trades=19, WR=47.4%, Sharpe=-0.26, PF=0.81, DD=-18.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.32, Test Sharpe=0.79, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4849, Sharpe CI=[-3.99, 3.24], WR CI=[31.6%, 73.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.0%, Median equity=$1,001, Survival=100.0% |
| Regime | **PASS** | bull:17t/+5.1%, bear:1t/-5.2%, volatile:1t/+2.4% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

