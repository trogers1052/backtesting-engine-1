# MU (Micron Technology) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 49.4 minutes
**Category:** Memory semiconductors

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

MU — Memory chips (DRAM/NAND/HBM) — most cyclical, extreme mean-reversion. Memory semiconductors.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 21 | 42.9% | +23.6% | 0.23 | 1.29 | -27.3% |
| Alt A: Full general rules (10 rules, 10%/5%) | 37 | 51.4% | +144.2% | 0.85 | 1.87 | -29.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 21 | 38.1% | +18.5% | 0.21 | 1.26 | -26.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 19 | 36.8% | +16.2% | 0.17 | 1.21 | -27.4% |
| Alt D: Tech rules (13 rules, 10%/5%) | 48 | 47.9% | +114.7% | 1.07 | 1.59 | -35.6% |
| Alt E: memory rules (3 rules, 10%/5%) | 28 | 42.9% | +27.9% | 0.29 | 1.36 | -33.4% |
| Alt F: Semi wider PT+stops (15%/7%) | 23 | 56.5% | +137.1% | 1.00 | 2.43 | -26.6% |

**Best baseline selected for validation: Alt D: Tech rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Tech rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+114.7%, Trades=48, WR=47.9%, Sharpe=1.07, PF=1.59, DD=-35.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.69, Test Sharpe=1.16, Ratio=169% (need >=50%) |
| Bootstrap | FAIL | p=0.0365, Sharpe CI=[-0.20, 3.94], WR CI=[37.5%, 66.7%] |
| Monte Carlo | FAIL | Ruin=0.2%, P95 DD=-47.0%, Median equity=$2,612, Survival=99.8% |
| Regime | FAIL | bull:42t/+115.4%, bear:4t/+1.8%, chop:2t/+1.3% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=15% | 34 | 47.1% | +182.8% | 1.29 |
| MC tune: max_loss=4.0% | 52 | 44.2% | +126.7% | 1.09 |
| BS tune: conf=0.4 | 48 | 47.9% | +114.7% | 1.07 |
| BS tune: conf=0.45 | 48 | 47.9% | +114.7% | 1.07 |
| BS tune: tech rules (13) | 48 | 47.9% | +114.7% | 1.07 |
| MC tune: ATR stops x2.0 | 48 | 47.9% | +114.7% | 1.07 |
| Regime tune: PT=12% | 45 | 48.9% | +197.4% | 1.05 |
| BS tune: full rules (10) | 37 | 51.4% | +144.2% | 0.85 |
| BS tune: memory rules | 31 | 48.4% | +59.8% | 0.63 |
| BS tune: cooldown=7 | 40 | 42.5% | +58.6% | 0.48 |
| MC tune: max_loss=3.0% | 62 | 33.9% | +37.6% | 0.47 |
| BS tune: conf=0.55 | 41 | 41.5% | +32.1% | 0.40 |
| Regime tune: conf=0.65 | 37 | 43.2% | +34.9% | 0.36 |
| Regime tune: PT=15% [multi-TF] | 83 | 53.0% | +116.8% | 0.59 |

### Full Validation of Top Candidates

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+182.8%, Trades=34, WR=47.1%, Sharpe=1.29, PF=2.20, DD=-42.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.19, Test Sharpe=1.13, Ratio=95% (need >=50%) |
| Bootstrap | **PASS** | p=0.0167, Sharpe CI=[0.23, 5.18], WR CI=[35.3%, 70.6%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.4%, Median equity=$3,315, Survival=100.0% |
| Regime | FAIL | bull:29t/+131.2%, bear:3t/+10.5%, chop:2t/+1.3% |

**Result: 2/4 gates passed**

---

### MC tune: max_loss=4.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+126.7%, Trades=52, WR=44.2%, Sharpe=1.09, PF=1.63, DD=-40.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.71, Test Sharpe=1.17, Ratio=165% (need >=50%) |
| Bootstrap | FAIL | p=0.0342, Sharpe CI=[-0.13, 3.74], WR CI=[32.7%, 59.6%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-45.1%, Median equity=$2,675, Survival=99.9% |
| Regime | FAIL | bull:46t/+116.6%, bear:4t/+1.0%, chop:2t/+2.1% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+114.7%, Trades=48, WR=47.9%, Sharpe=1.07, PF=1.59, DD=-35.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.69, Test Sharpe=1.16, Ratio=169% (need >=50%) |
| Bootstrap | FAIL | p=0.0365, Sharpe CI=[-0.20, 3.94], WR CI=[37.5%, 66.7%] |
| Monte Carlo | FAIL | Ruin=0.2%, P95 DD=-47.0%, Median equity=$2,612, Survival=99.8% |
| Regime | FAIL | bull:42t/+115.4%, bear:4t/+1.8%, chop:2t/+1.3% |

**Result: 1/4 gates passed**

---

### Regime tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+116.8%, Trades=83, WR=53.0%, Sharpe=0.59, PF=1.52, DD=-45.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.74, Test Sharpe=0.95, Ratio=128% (need >=50%) |
| Bootstrap | FAIL | p=0.0283, Sharpe CI=[-0.05, 2.91], WR CI=[44.6%, 66.3%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-44.7%, Median equity=$2,785, Survival=99.9% |
| Regime | FAIL | bull:57t/+87.2%, bear:10t/+4.3%, chop:10t/+13.7%, volatile:6t/+19.3% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=15%** | **PASS** | **PASS** | FAIL | FAIL | **1.29** | **+182.8%** | 34 |
| MC tune: max_loss=4.0% | **PASS** | FAIL | FAIL | FAIL | 1.09 | +126.7% | 52 |
| Alt D: Tech rules (13 rules, 10%/5%) | **PASS** | FAIL | FAIL | FAIL | 1.07 | +114.7% | 48 |
| BS tune: conf=0.4 | **PASS** | FAIL | FAIL | FAIL | 1.07 | +114.7% | 48 |
| Regime tune: PT=15% [multi-TF] | **PASS** | FAIL | FAIL | FAIL | 0.59 | +116.8% | 83 |

---

## 5. Final Recommendation

**MU partially validates.** Best config: Regime tune: PT=15% (2/4 gates).

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+182.8%, Trades=34, WR=47.1%, Sharpe=1.29, PF=2.20, DD=-42.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.19, Test Sharpe=1.13, Ratio=95% (need >=50%) |
| Bootstrap | **PASS** | p=0.0167, Sharpe CI=[0.23, 5.18], WR CI=[35.3%, 70.6%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.4%, Median equity=$3,315, Survival=100.0% |
| Regime | FAIL | bull:29t/+131.2%, bear:3t/+10.5%, chop:2t/+1.3% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

