# O (Realty Income) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 79.7 minutes
**Category:** REIT

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

O — Monthly dividend REIT — retail, industrial, gaming triple-net leases. REIT.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 4 | 25.0% | -2.4% | -0.22 | 0.79 | -13.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 10.0% | -14.3% | -0.50 | 0.25 | -22.7% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 4 | 25.0% | +0.8% | -0.16 | 1.10 | -9.9% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 4 | 25.0% | -3.5% | -0.27 | 0.69 | -13.0% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 27 | 48.1% | -11.9% | -0.47 | 0.59 | -23.2% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 16 | 56.2% | -4.1% | -0.29 | 0.86 | -16.2% |

**Best baseline selected for validation: Alt E: Financial lean rules (3 sector rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: Financial lean rules (3 sector rules, 10%/5%)

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-4.1%, Trades=16, WR=56.2%, Sharpe=-0.29, PF=0.86, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.83, Test Sharpe=0.19, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5078, Sharpe CI=[-4.07, 3.74], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.7%, Median equity=$988, Survival=100.0% |
| Regime | **PASS** | bull:14t/+3.2%, bear:1t/+4.7%, volatile:1t/-7.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=3 | 20 | 65.0% | +4.9% | -0.02 |
| WF tune: conf=0.65 | 8 | 37.5% | +0.6% | -0.15 |
| WF tune: conf=0.6 | 14 | 57.1% | -1.3% | -0.17 |
| WF tune: cooldown=7 | 15 | 60.0% | -0.3% | -0.18 |
| WF tune: PT=12% | 16 | 50.0% | -4.0% | -0.28 |
| WF tune: PT=15% | 16 | 50.0% | -4.0% | -0.28 |
| WF tune: conf=0.45 | 16 | 56.2% | -4.1% | -0.29 |
| WF tune: conf=0.55 | 16 | 56.2% | -4.1% | -0.29 |
| BS tune: + volume_breakout | 16 | 56.2% | -4.1% | -0.29 |
| BS tune: conf=0.4 | 16 | 56.2% | -3.8% | -0.32 |
| WF tune: PT=8% | 16 | 56.2% | -5.5% | -0.37 |
| BS tune: financial rules (12) | 27 | 48.1% | -11.9% | -0.47 |
| BS tune: full rules (10) | 10 | 10.0% | -14.3% | -0.50 |
| WF tune: cooldown=3 [multi-TF] | 51 | 54.9% | +4.5% | -0.01 |

### Full Validation of Top Candidates

### WF tune: cooldown=3

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.9%, Trades=20, WR=65.0%, Sharpe=-0.02, PF=1.19, DD=-9.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.40, Test Sharpe=0.24, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2843, Sharpe CI=[-2.34, 4.35], WR CI=[45.0%, 85.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.4%, Median equity=$1,097, Survival=100.0% |
| Regime | **PASS** | bull:17t/+8.3%, bear:1t/+4.7%, chop:2t/-2.2% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+0.6%, Trades=8, WR=37.5%, Sharpe=-0.15, PF=1.03, DD=-13.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.57, Test Sharpe=-0.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4377, Sharpe CI=[-7.65, 5.91], WR CI=[12.5%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.0%, Median equity=$1,023, Survival=100.0% |
| Regime | FAIL | bull:7t/+9.1%, bear:1t/-4.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-1.3%, Trades=14, WR=57.1%, Sharpe=-0.17, PF=0.95, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.29, Test Sharpe=-0.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4514, Sharpe CI=[-4.04, 4.41], WR CI=[28.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.7%, Median equity=$1,014, Survival=100.0% |
| Regime | FAIL | bull:12t/+0.1%, bear:1t/+4.7%, volatile:1t/-2.0% |

**Result: 1/4 gates passed**

---

### WF tune: cooldown=3 [multi-TF]

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.5%, Trades=51, WR=54.9%, Sharpe=-0.01, PF=1.11, DD=-17.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.41, Test Sharpe=0.39, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2332, Sharpe CI=[-1.37, 2.56], WR CI=[47.1%, 74.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.0%, Median equity=$1,157, Survival=100.0% |
| Regime | FAIL | bull:34t/+20.6%, bear:9t/-3.3%, chop:7t/+5.7%, crisis:1t/-6.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=3** | FAIL | FAIL | **PASS** | **PASS** | **-0.02** | **+4.9%** | 20 |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | -0.29 | -4.1% | 16 |
| WF tune: cooldown=3 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.01 | +4.5% | 51 |
| WF tune: conf=0.65 | FAIL | FAIL | **PASS** | FAIL | -0.15 | +0.6% | 8 |
| WF tune: conf=0.6 | FAIL | FAIL | **PASS** | FAIL | -0.17 | -1.3% | 14 |

---

## 5. Final Recommendation

**O partially validates.** Best config: WF tune: cooldown=3 (2/4 gates).

### WF tune: cooldown=3

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.9%, Trades=20, WR=65.0%, Sharpe=-0.02, PF=1.19, DD=-9.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.40, Test Sharpe=0.24, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2843, Sharpe CI=[-2.34, 4.35], WR CI=[45.0%, 85.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.4%, Median equity=$1,097, Survival=100.0% |
| Regime | **PASS** | bull:17t/+8.3%, bear:1t/+4.7%, chop:2t/-2.2% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

