# V (Visa) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 79.0 minutes
**Category:** Payment network

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

V — Global payment network — credit/debit card rails, fintech partnerships. Payment network.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 50.0% | +8.7% | 0.09 | 1.20 | -26.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 47.1% | +14.6% | 0.21 | 1.26 | -19.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 14 | 42.9% | +6.3% | 0.04 | 1.14 | -21.9% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 13 | 46.2% | -2.5% | -0.11 | 0.94 | -26.4% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 20 | 45.0% | +11.4% | 0.13 | 1.18 | -23.5% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 13 | 53.8% | +20.0% | 0.23 | 1.45 | -27.3% |

**Best baseline selected for validation: Alt E: Financial lean rules (3 sector rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: Financial lean rules (3 sector rules, 10%/5%)

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+20.0%, Trades=13, WR=53.8%, Sharpe=0.23, PF=1.45, DD=-27.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.74, Test Sharpe=-0.78, Ratio=-105% (need >=50%) |
| Bootstrap | FAIL | p=0.1679, Sharpe CI=[-2.08, 7.07], WR CI=[30.8%, 76.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.6%, Median equity=$1,284, Survival=100.0% |
| Regime | **PASS** | bull:10t/+22.2%, bear:2t/+19.9%, chop:1t/-12.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 10 | 70.0% | +47.1% | 0.63 |
| WF tune: PT=15% | 11 | 45.5% | +26.3% | 0.40 |
| WF tune: conf=0.65 | 10 | 50.0% | +22.0% | 0.34 |
| WF tune: conf=0.45 | 13 | 53.8% | +20.0% | 0.23 |
| WF tune: conf=0.55 | 13 | 53.8% | +20.0% | 0.23 |
| BS tune: conf=0.4 | 13 | 53.8% | +20.0% | 0.23 |
| BS tune: + volume_breakout | 13 | 53.8% | +20.0% | 0.23 |
| BS tune: full rules (10) | 17 | 47.1% | +14.6% | 0.21 |
| WF tune: PT=12% | 13 | 46.2% | +15.6% | 0.19 |
| BS tune: financial rules (12) | 20 | 45.0% | +11.4% | 0.13 |
| WF tune: PT=8% | 14 | 50.0% | +10.9% | 0.13 |
| WF tune: cooldown=3 | 13 | 46.2% | +10.5% | 0.12 |
| WF tune: cooldown=7 | 13 | 46.2% | +7.0% | 0.06 |
| WF tune: conf=0.6 [multi-TF] | 21 | 52.4% | +9.6% | 0.13 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+47.1%, Trades=10, WR=70.0%, Sharpe=0.63, PF=3.66, DD=-14.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.94, Test Sharpe=-0.83, Ratio=-88% (need >=50%) |
| Bootstrap | **PASS** | p=0.0075, Sharpe CI=[1.10, 15.60], WR CI=[40.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.6%, Median equity=$1,649, Survival=100.0% |
| Regime | **PASS** | bull:8t/+33.5%, bear:2t/+19.9% |

**Result: 3/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.3%, Trades=11, WR=45.5%, Sharpe=0.40, PF=1.68, DD=-23.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-0.78, Ratio=-103% (need >=50%) |
| Bootstrap | FAIL | p=0.1835, Sharpe CI=[-2.81, 6.81], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.3%, Median equity=$1,303, Survival=100.0% |
| Regime | **PASS** | bull:9t/+28.2%, bear:1t/+16.6%, chop:1t/-12.5% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+22.0%, Trades=10, WR=50.0%, Sharpe=0.34, PF=1.83, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.48, Test Sharpe=-1.82, Ratio=-382% (need >=50%) |
| Bootstrap | FAIL | p=0.1469, Sharpe CI=[-2.17, 8.44], WR CI=[20.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.3%, Median equity=$1,289, Survival=100.0% |
| Regime | **PASS** | bull:8t/+8.7%, bear:2t/+19.9% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.6 [multi-TF]

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+9.6%, Trades=21, WR=52.4%, Sharpe=0.13, PF=1.34, DD=-18.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.28, Test Sharpe=-1.12, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2207, Sharpe CI=[-2.19, 4.09], WR CI=[33.3%, 76.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.1%, Median equity=$1,163, Survival=100.0% |
| Regime | **PASS** | bull:15t/+7.8%, bear:4t/+11.4%, chop:2t/-1.8% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | **PASS** | **PASS** | **PASS** | **0.63** | **+47.1%** | 10 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | **PASS** | 0.40 | +26.3% | 11 |
| WF tune: conf=0.65 | FAIL | FAIL | **PASS** | **PASS** | 0.34 | +22.0% | 10 |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.23 | +20.0% | 13 |
| WF tune: conf=0.6 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.13 | +9.6% | 21 |

---

## 5. Final Recommendation

**V partially validates.** Best config: WF tune: conf=0.6 (3/4 gates).

### WF tune: conf=0.6

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+47.1%, Trades=10, WR=70.0%, Sharpe=0.63, PF=3.66, DD=-14.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.94, Test Sharpe=-0.83, Ratio=-88% (need >=50%) |
| Bootstrap | **PASS** | p=0.0075, Sharpe CI=[1.10, 15.60], WR CI=[40.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.6%, Median equity=$1,649, Survival=100.0% |
| Regime | **PASS** | bull:8t/+33.5%, bear:2t/+19.9% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

