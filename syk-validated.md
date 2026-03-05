# SYK (Stryker) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 68.5 minutes
**Category:** Medical devices (MOMENTUM)

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

SYK — Medical devices — beta 0.69-0.87, growth compounder, 11% organic sales growth. Medical devices (MOMENTUM).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 41.7% | +5.3% | -0.17 | 1.14 | -18.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 47.1% | +23.9% | 0.64 | 1.46 | -17.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 38.5% | +5.8% | -0.00 | 1.16 | -16.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 10 | 30.0% | -0.3% | -0.29 | 0.99 | -22.5% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 22 | 45.5% | +36.3% | 1.02 | 1.64 | -19.0% |
| Alt E: med_devices lean (3 rules, 10%/5%) | 12 | 41.7% | +5.3% | -0.17 | 1.14 | -18.7% |
| Alt F: Momentum (15%/8%) | 8 | 50.0% | +21.5% | 0.40 | 1.73 | -19.6% |
| Alt G: Tech-style (12%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Alt D: Healthcare rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Healthcare rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.3%, Trades=22, WR=45.5%, Sharpe=1.02, PF=1.64, DD=-19.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.85, Test Sharpe=-1.26, Ratio=-148% (need >=50%) |
| Bootstrap | FAIL | p=0.0918, Sharpe CI=[-1.02, 5.23], WR CI=[22.7%, 68.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.6%, Median equity=$1,510, Survival=100.0% |
| Regime | **PASS** | bull:14t/+18.5%, bear:3t/+26.5%, chop:2t/-6.8%, volatile:3t/+9.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 22 | 50.0% | +38.8% | 1.19 |
| WF tune: PT=6% | 31 | 54.8% | +32.0% | 1.06 |
| WF tune: conf=0.45 | 22 | 45.5% | +36.3% | 1.02 |
| BS tune: healthcare rules (13) | 22 | 45.5% | +36.3% | 1.02 |
| WF tune: ATR stops x2.5 | 23 | 43.5% | +33.0% | 1.01 |
| WF tune: PT=15% | 17 | 41.2% | +33.4% | 0.89 |
| WF tune: PT=12% | 21 | 42.9% | +27.8% | 0.74 |
| WF tune: PT=7% | 28 | 50.0% | +23.1% | 0.68 |
| BS tune: full rules (10) | 17 | 47.1% | +23.9% | 0.64 |
| WF tune: cooldown=7 | 21 | 42.9% | +22.6% | 0.58 |
| WF tune: PT=8% | 27 | 44.4% | +16.6% | 0.48 |
| WF tune: conf=0.65 | 19 | 42.1% | +8.1% | 0.08 |
| WF tune: conf=0.6 | 21 | 38.1% | +7.3% | 0.07 |
| WF tune: conf=0.55 | 23 | 34.8% | +5.9% | 0.01 |
| BS tune: med_devices rules | 12 | 41.7% | +4.7% | -0.35 |
| BS tune: conf=0.4 [multi-TF] | 60 | 53.3% | +24.0% | 0.55 |

### Full Validation of Top Candidates

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+38.8%, Trades=22, WR=50.0%, Sharpe=1.19, PF=1.68, DD=-19.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.85, Test Sharpe=-0.78, Ratio=-91% (need >=50%) |
| Bootstrap | FAIL | p=0.0820, Sharpe CI=[-0.88, 5.39], WR CI=[27.3%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.3%, Median equity=$1,544, Survival=100.0% |
| Regime | **PASS** | bull:14t/+20.7%, bear:3t/+26.5%, chop:2t/-6.8%, volatile:3t/+9.5% |

**Result: 2/4 gates passed**

---

### WF tune: PT=6%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+32.0%, Trades=31, WR=54.8%, Sharpe=1.06, PF=1.46, DD=-19.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.23, Test Sharpe=-1.26, Ratio=-103% (need >=50%) |
| Bootstrap | FAIL | p=0.0834, Sharpe CI=[-0.75, 4.67], WR CI=[38.7%, 71.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.5%, Median equity=$1,484, Survival=100.0% |
| Regime | **PASS** | bull:20t/+7.6%, bear:4t/+15.5%, chop:4t/+16.1%, volatile:3t/+5.2% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.3%, Trades=22, WR=45.5%, Sharpe=1.02, PF=1.64, DD=-19.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.85, Test Sharpe=-1.26, Ratio=-148% (need >=50%) |
| Bootstrap | FAIL | p=0.0918, Sharpe CI=[-1.02, 5.23], WR CI=[22.7%, 68.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.6%, Median equity=$1,510, Survival=100.0% |
| Regime | **PASS** | bull:14t/+18.5%, bear:3t/+26.5%, chop:2t/-6.8%, volatile:3t/+9.5% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+24.0%, Trades=60, WR=53.3%, Sharpe=0.55, PF=1.29, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.55, Test Sharpe=-0.52, Ratio=-95% (need >=50%) |
| Bootstrap | FAIL | p=0.1303, Sharpe CI=[-0.87, 2.70], WR CI=[50.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.8%, Median equity=$1,433, Survival=100.0% |
| Regime | **PASS** | bull:37t/+12.3%, bear:4t/+12.7%, chop:11t/-0.6%, volatile:6t/+13.0%, crisis:2t/+4.8% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | FAIL | FAIL | **PASS** | **PASS** | **1.19** | **+38.8%** | 22 |
| WF tune: PT=6% | FAIL | FAIL | **PASS** | **PASS** | 1.06 | +32.0% | 31 |
| Alt D: Healthcare rules (13 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 1.02 | +36.3% | 22 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | **PASS** | 1.02 | +36.3% | 22 |
| BS tune: conf=0.4 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.55 | +24.0% | 60 |

---

## 5. Final Recommendation

**SYK partially validates.** Best config: BS tune: conf=0.4 (2/4 gates).

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+38.8%, Trades=22, WR=50.0%, Sharpe=1.19, PF=1.68, DD=-19.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.85, Test Sharpe=-0.78, Ratio=-91% (need >=50%) |
| Bootstrap | FAIL | p=0.0820, Sharpe CI=[-0.88, 5.39], WR CI=[27.3%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.3%, Median equity=$1,544, Survival=100.0% |
| Regime | **PASS** | bull:14t/+20.7%, bear:3t/+26.5%, chop:2t/-6.8%, volatile:3t/+9.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

