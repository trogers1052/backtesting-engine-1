# AMT (American Tower) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 77.8 minutes
**Category:** Infrastructure REIT

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

AMT — Cell tower REIT — wireless infrastructure, data centers, global. Infrastructure REIT.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 4 | 0.0% | -22.5% | -0.74 | 0.00 | -23.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 8 | 25.0% | -15.7% | -0.83 | 0.45 | -23.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 4 | 0.0% | -15.4% | -0.78 | 0.00 | -16.2% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 6 | 33.3% | -8.7% | -0.36 | 0.57 | -17.7% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 24 | 41.7% | -12.7% | -0.43 | 0.75 | -26.6% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 18 | 61.1% | +22.7% | 0.37 | 1.73 | -19.5% |

**Best baseline selected for validation: Alt E: Financial lean rules (3 sector rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: Financial lean rules (3 sector rules, 10%/5%)

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+22.7%, Trades=18, WR=61.1%, Sharpe=0.37, PF=1.73, DD=-19.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.18, Test Sharpe=-0.09, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1183, Sharpe CI=[-1.39, 6.10], WR CI=[44.4%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.9%, Median equity=$1,309, Survival=100.0% |
| Regime | **PASS** | bull:13t/+14.2%, bear:1t/+6.4%, chop:3t/-1.9%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 16 | 62.5% | +30.3% | 0.48 |
| WF tune: cooldown=7 | 18 | 55.6% | +26.7% | 0.41 |
| WF tune: conf=0.55 | 18 | 61.1% | +22.7% | 0.37 |
| BS tune: + volume_breakout | 18 | 61.1% | +22.7% | 0.37 |
| WF tune: conf=0.45 | 18 | 61.1% | +22.0% | 0.36 |
| WF tune: PT=15% | 18 | 55.6% | +12.6% | 0.34 |
| BS tune: conf=0.4 | 18 | 55.6% | +18.6% | 0.28 |
| WF tune: PT=12% | 18 | 55.6% | +9.7% | 0.19 |
| WF tune: PT=8% | 19 | 57.9% | +10.2% | 0.16 |
| WF tune: conf=0.65 | 10 | 50.0% | +11.5% | 0.14 |
| BS tune: financial rules (12) | 24 | 41.7% | -12.7% | -0.43 |
| WF tune: cooldown=3 | 21 | 42.9% | -10.1% | -0.44 |
| BS tune: full rules (10) | 8 | 25.0% | -15.7% | -0.83 |
| WF tune: conf=0.6 [multi-TF] | 31 | 48.4% | -16.4% | -1.31 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+30.3%, Trades=16, WR=62.5%, Sharpe=0.48, PF=2.19, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.16, Test Sharpe=-0.14, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0761, Sharpe CI=[-1.04, 7.30], WR CI=[43.8%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.4%, Median equity=$1,376, Survival=100.0% |
| Regime | **PASS** | bull:11t/+19.0%, bear:1t/+6.4%, chop:3t/-1.9%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+26.7%, Trades=18, WR=55.6%, Sharpe=0.41, PF=1.74, DD=-21.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.66, Test Sharpe=0.08, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0928, Sharpe CI=[-1.19, 6.29], WR CI=[38.9%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.9%, Median equity=$1,350, Survival=100.0% |
| Regime | **PASS** | bull:13t/+13.3%, bear:1t/+6.4%, chop:3t/+1.4%, crisis:1t/+12.2% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+22.7%, Trades=18, WR=61.1%, Sharpe=0.37, PF=1.73, DD=-19.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.18, Test Sharpe=-0.09, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1183, Sharpe CI=[-1.39, 6.10], WR CI=[44.4%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.9%, Median equity=$1,309, Survival=100.0% |
| Regime | **PASS** | bull:13t/+14.2%, bear:1t/+6.4%, chop:3t/-1.9%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.6 [multi-TF]

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-16.4%, Trades=31, WR=48.4%, Sharpe=-1.31, PF=0.54, DD=-18.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.64, Test Sharpe=-1.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7923, Sharpe CI=[-4.01, 1.40], WR CI=[32.3%, 64.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.9%, Median equity=$867, Survival=100.0% |
| Regime | **PASS** | bull:18t/-15.6%, bear:4t/-5.3%, chop:7t/+4.9%, volatile:1t/+1.7%, crisis:1t/+1.6% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | FAIL | **PASS** | **PASS** | **0.48** | **+30.3%** | 16 |
| WF tune: cooldown=7 | FAIL | FAIL | **PASS** | **PASS** | 0.41 | +26.7% | 18 |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.37 | +22.7% | 18 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | **PASS** | 0.37 | +22.7% | 18 |
| WF tune: conf=0.6 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | -1.31 | -16.4% | 31 |

---

## 5. Final Recommendation

**AMT partially validates.** Best config: WF tune: conf=0.6 (2/4 gates).

### WF tune: conf=0.6

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+30.3%, Trades=16, WR=62.5%, Sharpe=0.48, PF=2.19, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.16, Test Sharpe=-0.14, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0761, Sharpe CI=[-1.04, 7.30], WR CI=[43.8%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.4%, Median equity=$1,376, Survival=100.0% |
| Regime | **PASS** | bull:11t/+19.0%, bear:1t/+6.4%, chop:3t/-1.9%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

