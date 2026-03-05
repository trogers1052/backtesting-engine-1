# KO (Coca-Cola) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 13.7 minutes
**Category:** Beverages (Dividend King)

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

KO — Beverages bellwether — beta 0.11-0.36, yield 2.6%, 62yr dividend streak. Beverages (Dividend King).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 4 | 50.0% | +9.8% | 0.18 | 1.90 | -14.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 40.0% | +6.3% | 0.07 | 1.04 | -19.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 5 | 40.0% | +5.8% | 0.05 | 1.48 | -12.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 4 | 50.0% | +11.4% | 0.23 | 2.04 | -14.2% |
| Alt D: Staples rules (13 rules, 10%/5%) | 21 | 57.1% | +18.0% | 0.38 | 1.46 | -12.8% |
| Alt E: beverages lean (3 rules, 10%/5%) | 15 | 66.7% | +20.1% | 0.93 | 2.18 | -12.5% |
| Alt F: Beverages tight (6%/3%, conf=0.55, cooldown=7) | 16 | 62.5% | +17.7% | 0.69 | 1.88 | -10.0% |
| Alt G: Beverages moderate (7%/4%) | 16 | 68.8% | +21.8% | 0.81 | 2.28 | -12.2% |

**Best baseline selected for validation: Alt E: beverages lean (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: beverages lean (3 rules, 10%/5%)

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+20.1%, Trades=15, WR=66.7%, Sharpe=0.93, PF=2.18, DD=-12.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.47, Test Sharpe=0.55, Ratio=116% (need >=50%) |
| Bootstrap | FAIL | p=0.0978, Sharpe CI=[-1.40, 6.16], WR CI=[46.7%, 93.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-13.3%, Median equity=$1,247, Survival=100.0% |
| Regime | FAIL | bull:14t/+28.6%, chop:1t/-4.8% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: cooldown=7 | 12 | 75.0% | +27.6% | 1.05 |
| Regime tune: conf=0.65 | 8 | 62.5% | +33.2% | 1.00 |
| Regime tune: tighter stop 4% | 15 | 66.7% | +20.8% | 0.96 |
| BS tune: cooldown=3 | 15 | 66.7% | +20.5% | 0.95 |
| BS tune: conf=0.4 | 15 | 66.7% | +20.1% | 0.93 |
| BS tune: conf=0.45 | 15 | 66.7% | +20.1% | 0.93 |
| BS tune: conf=0.55 | 15 | 66.7% | +20.1% | 0.93 |
| Regime tune: + staples_pullback | 15 | 66.7% | +20.1% | 0.93 |
| Regime tune: PT=12% | 15 | 60.0% | +22.4% | 0.91 |
| Regime tune: PT=8% | 15 | 66.7% | +16.3% | 0.90 |
| Regime tune: PT=15% | 15 | 60.0% | +24.0% | 0.90 |
| Regime tune: PT=7% | 16 | 68.8% | +21.1% | 0.78 |
| BS tune: staples rules (13) | 21 | 57.1% | +18.0% | 0.38 |
| BS tune: full rules (10) | 10 | 40.0% | +6.3% | 0.07 |
| BS tune: cooldown=7 [multi-TF] | 21 | 38.1% | +0.7% | -0.05 |

### Full Validation of Top Candidates

### BS tune: cooldown=7

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+27.6%, Trades=12, WR=75.0%, Sharpe=1.05, PF=3.30, DD=-12.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.77, Test Sharpe=0.74, Ratio=96% (need >=50%) |
| Bootstrap | FAIL | p=0.0339, Sharpe CI=[-0.32, 8.49], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.7%, Median equity=$1,336, Survival=100.0% |
| Regime | FAIL | bull:11t/+35.7%, chop:1t/-4.9% |

**Result: 2/4 gates passed**

---

### Regime tune: conf=0.65

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+33.2%, Trades=8, WR=62.5%, Sharpe=1.00, PF=3.76, DD=-13.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.62, Test Sharpe=0.46, Ratio=74% (need >=50%) |
| Bootstrap | FAIL | p=0.0405, Sharpe CI=[-0.59, 14.05], WR CI=[25.0%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.0%, Median equity=$1,375, Survival=100.0% |
| Regime | FAIL | bull:7t/+39.1%, chop:1t/-4.8% |

**Result: 2/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+20.8%, Trades=15, WR=66.7%, Sharpe=0.96, PF=2.27, DD=-12.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.47, Test Sharpe=0.55, Ratio=116% (need >=50%) |
| Bootstrap | FAIL | p=0.0868, Sharpe CI=[-1.26, 6.21], WR CI=[46.7%, 93.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.7%, Median equity=$1,255, Survival=100.0% |
| Regime | FAIL | bull:14t/+28.6%, chop:1t/-4.1% |

**Result: 2/4 gates passed**

---

### BS tune: cooldown=7 [multi-TF]

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+0.7%, Trades=21, WR=38.1%, Sharpe=-0.05, PF=0.89, DD=-26.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.02, Test Sharpe=0.32, Ratio=1286% (need >=50%) |
| Bootstrap | FAIL | p=0.4001, Sharpe CI=[-3.70, 3.13], WR CI=[28.6%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.2%, Median equity=$1,048, Survival=100.0% |
| Regime | FAIL | bull:19t/+16.5%, volatile:2t/-10.3% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: cooldown=7** | **PASS** | FAIL | **PASS** | FAIL | **1.05** | **+27.6%** | 12 |
| Regime tune: conf=0.65 | **PASS** | FAIL | **PASS** | FAIL | 1.00 | +33.2% | 8 |
| Regime tune: tighter stop 4% | **PASS** | FAIL | **PASS** | FAIL | 0.96 | +20.8% | 15 |
| Alt E: beverages lean (3 rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.93 | +20.1% | 15 |
| BS tune: cooldown=7 [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | -0.05 | +0.7% | 21 |

---

## 5. Final Recommendation

**KO partially validates.** Best config: BS tune: cooldown=7 (2/4 gates).

### BS tune: cooldown=7

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+27.6%, Trades=12, WR=75.0%, Sharpe=1.05, PF=3.30, DD=-12.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.77, Test Sharpe=0.74, Ratio=96% (need >=50%) |
| Bootstrap | FAIL | p=0.0339, Sharpe CI=[-0.32, 8.49], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.7%, Median equity=$1,336, Survival=100.0% |
| Regime | FAIL | bull:11t/+35.7%, chop:1t/-4.9% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

