# CB (Chubb Limited) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 77.2 minutes
**Category:** Insurance

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

CB — Global P&C insurance — commercial, personal, specialty lines. Insurance.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 50.0% | +23.6% | 0.36 | 1.84 | -16.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 31.2% | -0.1% | -0.06 | 0.94 | -24.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 50.0% | +27.8% | 0.41 | 2.15 | -12.9% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 10 | 50.0% | +19.9% | 0.37 | 1.76 | -17.2% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 26 | 42.3% | +16.3% | 0.20 | 1.23 | -24.8% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 19 | 52.6% | +36.1% | 0.55 | 2.09 | -20.4% |

**Best baseline selected for validation: Alt E: Financial lean rules (3 sector rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: Financial lean rules (3 sector rules, 10%/5%)

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+36.1%, Trades=19, WR=52.6%, Sharpe=0.55, PF=2.09, DD=-20.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.77, Test Sharpe=1.18, Ratio=154% (need >=50%) |
| Bootstrap | FAIL | p=0.0331, Sharpe CI=[-0.21, 6.94], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.9%, Median equity=$1,561, Survival=100.0% |
| Regime | FAIL | bull:13t/+61.1%, bear:3t/+1.9%, chop:2t/-10.1%, volatile:1t/-4.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=15% | 16 | 43.8% | +29.1% | 0.72 |
| BS tune: cooldown=7 | 16 | 56.2% | +36.2% | 0.59 |
| Regime tune: PT=12% | 19 | 47.4% | +34.7% | 0.57 |
| BS tune: conf=0.4 | 18 | 55.6% | +36.6% | 0.56 |
| Regime tune: conf=0.65 | 8 | 62.5% | +36.3% | 0.55 |
| BS tune: conf=0.45 | 19 | 52.6% | +36.1% | 0.55 |
| BS tune: conf=0.55 | 19 | 52.6% | +36.1% | 0.55 |
| BS tune: + volume_breakout | 19 | 52.6% | +36.1% | 0.55 |
| Regime tune: tighter stop 4% | 21 | 47.6% | +32.2% | 0.53 |
| BS tune: cooldown=3 | 20 | 50.0% | +29.4% | 0.37 |
| BS tune: financial rules (12) | 26 | 42.3% | +16.3% | 0.20 |
| BS tune: full rules (10) | 16 | 31.2% | -0.1% | -0.06 |
| Regime tune: PT=15% [multi-TF] | 28 | 42.9% | +25.8% | 0.27 |

### Full Validation of Top Candidates

### Regime tune: PT=15%

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+29.1%, Trades=16, WR=43.8%, Sharpe=0.72, PF=1.61, DD=-18.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.90, Test Sharpe=1.15, Ratio=129% (need >=50%) |
| Bootstrap | FAIL | p=0.1019, Sharpe CI=[-1.55, 5.46], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.8%, Median equity=$1,394, Survival=100.0% |
| Regime | FAIL | bull:10t/+50.3%, bear:3t/+1.9%, chop:2t/-10.1%, volatile:1t/-4.5% |

**Result: 2/4 gates passed**

---

### BS tune: cooldown=7

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+36.2%, Trades=16, WR=56.2%, Sharpe=0.59, PF=2.37, DD=-17.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.63, Test Sharpe=1.14, Ratio=179% (need >=50%) |
| Bootstrap | FAIL | p=0.0404, Sharpe CI=[-0.39, 6.72], WR CI=[43.8%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.2%, Median equity=$1,522, Survival=100.0% |
| Regime | FAIL | bull:11t/+57.6%, bear:3t/-1.3%, chop:1t/-4.7%, volatile:1t/-5.8% |

**Result: 2/4 gates passed**

---

### Regime tune: PT=12%

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+34.7%, Trades=19, WR=47.4%, Sharpe=0.57, PF=1.93, DD=-19.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.78, Test Sharpe=1.14, Ratio=146% (need >=50%) |
| Bootstrap | FAIL | p=0.0600, Sharpe CI=[-0.77, 5.62], WR CI=[36.8%, 78.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.7%, Median equity=$1,517, Survival=100.0% |
| Regime | FAIL | bull:13t/+59.2%, bear:3t/+1.9%, chop:2t/-10.1%, volatile:1t/-4.5% |

**Result: 2/4 gates passed**

---

### Regime tune: PT=15% [multi-TF]

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+25.8%, Trades=28, WR=42.9%, Sharpe=0.27, PF=1.66, DD=-20.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.48, Test Sharpe=0.93, Ratio=195% (need >=50%) |
| Bootstrap | FAIL | p=0.1281, Sharpe CI=[-1.43, 3.62], WR CI=[32.1%, 67.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.7%, Median equity=$1,390, Survival=100.0% |
| Regime | FAIL | bull:19t/+57.7%, bear:4t/-7.1%, chop:2t/-6.0%, volatile:3t/-6.3% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=15%** | **PASS** | FAIL | **PASS** | FAIL | **0.72** | **+29.1%** | 16 |
| BS tune: cooldown=7 | **PASS** | FAIL | **PASS** | FAIL | 0.59 | +36.2% | 16 |
| Regime tune: PT=12% | **PASS** | FAIL | **PASS** | FAIL | 0.57 | +34.7% | 19 |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.55 | +36.1% | 19 |
| Regime tune: PT=15% [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | 0.27 | +25.8% | 28 |

---

## 5. Final Recommendation

**CB partially validates.** Best config: Regime tune: PT=15% (2/4 gates).

### Regime tune: PT=15%

- **Rules:** `financial_mean_reversion, financial_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+29.1%, Trades=16, WR=43.8%, Sharpe=0.72, PF=1.61, DD=-18.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.90, Test Sharpe=1.15, Ratio=129% (need >=50%) |
| Bootstrap | FAIL | p=0.1019, Sharpe CI=[-1.55, 5.46], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.8%, Median equity=$1,394, Survival=100.0% |
| Regime | FAIL | bull:10t/+50.3%, bear:3t/+1.9%, chop:2t/-10.1%, volatile:1t/-4.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

