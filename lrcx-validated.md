# LRCX (Lam Research) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 42.4 minutes
**Category:** Semi equipment

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

LRCX — Semiconductor equipment (etch/deposition) — cyclical growth, semi capex cycle. Semi equipment.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 15 | 60.0% | +94.5% | 0.79 | 3.22 | -11.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 30 | 56.7% | +171.5% | 0.79 | 2.51 | -24.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 17 | 52.9% | +77.9% | 0.76 | 2.56 | -14.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 14 | 57.1% | +95.2% | 0.66 | 3.17 | -12.3% |
| Alt D: Tech rules (13 rules, 10%/5%) | 38 | 50.0% | +120.7% | 0.53 | 1.85 | -29.7% |
| Alt E: semi_equip rules (4 rules, 10%/5%) | 27 | 48.1% | +58.2% | 0.40 | 1.67 | -23.2% |
| Alt F: Semi wider PT+stops (15%/7%) | 18 | 55.6% | +123.7% | 0.79 | 2.74 | -16.3% |

**Best baseline selected for validation: Alt F: Semi wider PT+stops (15%/7%)**

---

## 2. Full Validation

### Alt F: Semi wider PT+stops (15%/7%)

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 7.0%
- **Cooldown:** 5 bars

**Performance:** Return=+123.7%, Trades=18, WR=55.6%, Sharpe=0.79, PF=2.74, DD=-16.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.50, Test Sharpe=1.30, Ratio=263% (need >=50%) |
| Bootstrap | **PASS** | p=0.0182, Sharpe CI=[0.25, 7.99], WR CI=[38.9%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.8%, Median equity=$2,483, Survival=100.0% |
| Regime | FAIL | bull:16t/+98.8%, bear:1t/-11.2%, chop:1t/+18.1% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: full rules (10) | 21 | 61.9% | +263.7% | 1.14 |
| Regime tune: conf=0.65 | 14 | 64.3% | +151.6% | 0.97 |
| Regime tune: tech rules (13) | 28 | 53.6% | +215.7% | 0.78 |
| Regime tune: tighter stop 4% | 21 | 47.6% | +129.2% | 0.68 |
| Regime tune: PT=12% | 20 | 55.0% | +90.0% | 0.58 |
| Regime tune: full rules (10) [multi-TF] | 60 | 55.0% | +276.0% | 0.95 |

### Full Validation of Top Candidates

### Regime tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 7.0%
- **Cooldown:** 3 bars

**Performance:** Return=+263.7%, Trades=21, WR=61.9%, Sharpe=1.14, PF=4.32, DD=-27.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=1.24, Ratio=148% (need >=50%) |
| Bootstrap | **PASS** | p=0.0010, Sharpe CI=[1.60, 9.52], WR CI=[47.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.3%, Median equity=$4,125, Survival=100.0% |
| Regime | FAIL | bull:18t/+136.7%, chop:3t/+22.8% |

**Result: 3/4 gates passed**

---

### Regime tune: conf=0.65

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.65
- **Max Loss:** 7.0%
- **Cooldown:** 5 bars

**Performance:** Return=+151.6%, Trades=14, WR=64.3%, Sharpe=0.97, PF=3.85, DD=-16.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.67, Test Sharpe=1.07, Ratio=160% (need >=50%) |
| Bootstrap | **PASS** | p=0.0061, Sharpe CI=[1.10, 13.29], WR CI=[50.0%, 92.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.5%, Median equity=$2,809, Survival=100.0% |
| Regime | FAIL | bull:12t/+109.7%, bear:1t/-11.2%, chop:1t/+18.1% |

**Result: 3/4 gates passed**

---

### Regime tune: tech rules (13)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 7.0%
- **Cooldown:** 3 bars

**Performance:** Return=+215.7%, Trades=28, WR=53.6%, Sharpe=0.78, PF=2.89, DD=-31.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.62, Test Sharpe=1.24, Ratio=198% (need >=50%) |
| Bootstrap | **PASS** | p=0.0077, Sharpe CI=[0.68, 6.56], WR CI=[39.3%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.6%, Median equity=$3,627, Survival=100.0% |
| Regime | FAIL | bull:23t/+139.2%, bear:1t/-11.2%, chop:3t/+22.8%, volatile:1t/-1.3% |

**Result: 3/4 gates passed**

---

### Regime tune: full rules (10) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 7.0%
- **Cooldown:** 3 bars

**Performance:** Return=+276.0%, Trades=60, WR=55.0%, Sharpe=0.95, PF=3.21, DD=-28.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.85, Test Sharpe=1.34, Ratio=157% (need >=50%) |
| Bootstrap | **PASS** | p=0.0010, Sharpe CI=[0.95, 4.29], WR CI=[48.3%, 71.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.5%, Median equity=$4,707, Survival=100.0% |
| Regime | FAIL | bull:41t/+145.8%, bear:7t/+2.5%, chop:9t/+26.3%, volatile:3t/-0.6% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: full rules (10)** | **PASS** | **PASS** | **PASS** | FAIL | **1.14** | **+263.7%** | 21 |
| Regime tune: conf=0.65 | **PASS** | **PASS** | **PASS** | FAIL | 0.97 | +151.6% | 14 |
| Regime tune: full rules (10) [multi-TF] | **PASS** | **PASS** | **PASS** | FAIL | 0.95 | +276.0% | 60 |
| Alt F: Semi wider PT+stops (15%/7%) | **PASS** | **PASS** | **PASS** | FAIL | 0.79 | +123.7% | 18 |
| Regime tune: tech rules (13) | **PASS** | **PASS** | **PASS** | FAIL | 0.78 | +215.7% | 28 |

---

## 5. Final Recommendation

**LRCX partially validates.** Best config: Regime tune: full rules (10) (3/4 gates).

### Regime tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 7.0%
- **Cooldown:** 3 bars

**Performance:** Return=+263.7%, Trades=21, WR=61.9%, Sharpe=1.14, PF=4.32, DD=-27.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=1.24, Ratio=148% (need >=50%) |
| Bootstrap | **PASS** | p=0.0010, Sharpe CI=[1.60, 9.52], WR CI=[47.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.3%, Median equity=$4,125, Survival=100.0% |
| Regime | FAIL | bull:18t/+136.7%, chop:3t/+22.8% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

