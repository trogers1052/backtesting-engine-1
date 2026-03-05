# OKLO (Oklo Inc) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2024-05-15 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 9.8 minutes
**Category:** Energy / Nuclear (SPECULATIVE — limited history)

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

OKLO — Nuclear micro-reactors — pre-revenue, speculative, trades with uranium sentiment. Energy / Nuclear (SPECULATIVE — limited history).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 1 | 0.0% | -8.4% | -1.24 | 0.00 | -8.4% |
| Alt A: Full general rules (10 rules, 10%/5%) | 5 | 20.0% | -12.3% | -3.47 | 0.50 | -18.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 1 | 0.0% | -4.2% | -1.48 | 0.00 | -4.2% |
| Alt C: Wider PT (3 rules, 12%/5%) | 1 | 0.0% | -8.4% | -1.24 | 0.00 | -8.4% |
| Alt D: Recommended rules (3 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt E: Mining/uranium rules (10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt F: Wide momentum (20%/10%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt G: Mining volume breakout (15%/8%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-12.3%, Trades=5, WR=20.0%, Sharpe=-3.47, PF=0.50, DD=-18.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7377, Sharpe CI=[-70.83, 3.94], WR CI=[0.0%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.1%, Median equity=$867, Survival=100.0% |
| Regime | **PASS** | bull:5t/-12.4% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=15% | 5 | 20.0% | -7.3% | -0.95 |
| WF tune: PT=6% | 6 | 33.3% | -12.4% | -3.41 |
| WF tune: PT=7% | 6 | 33.3% | -12.4% | -3.41 |
| WF tune: PT=12% | 5 | 20.0% | -12.3% | -3.47 |
| WF tune: conf=0.45 | 5 | 20.0% | -12.3% | -3.47 |
| WF tune: ATR stops x2.5 | 5 | 20.0% | -12.3% | -3.47 |
| BS tune: conf=0.4 | 5 | 20.0% | -12.3% | -3.47 |
| BS tune: full rules (10) | 5 | 20.0% | -12.3% | -3.47 |
| WF tune: PT=8% | 6 | 33.3% | -9.0% | -10.20 |
| WF tune: conf=0.55 | 2 | 0.0% | -10.9% | -1.18 |
| WF tune: conf=0.6 | 2 | 0.0% | -10.9% | -1.18 |
| WF tune: conf=0.65 | 2 | 0.0% | -10.9% | -1.18 |
| WF tune: cooldown=7 | 3 | 33.3% | -2.1% | -0.23 |
| WF tune: recommended rules | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: PT=15% [multi-TF] | 7 | 28.6% | -9.6% | -0.81 |

### Full Validation of Top Candidates

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-7.3%, Trades=5, WR=20.0%, Sharpe=-0.95, PF=0.71, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7339, Sharpe CI=[-70.83, 4.90], WR CI=[0.0%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.1%, Median equity=$916, Survival=100.0% |
| Regime | **PASS** | bull:5t/-5.9% |

**Result: 2/4 gates passed**

---

### WF tune: PT=6%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-12.4%, Trades=6, WR=33.3%, Sharpe=-3.41, PF=0.59, DD=-17.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7075, Sharpe CI=[-15.01, 4.52], WR CI=[0.0%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.0%, Median equity=$867, Survival=100.0% |
| Regime | **PASS** | bull:6t/-11.3% |

**Result: 2/4 gates passed**

---

### WF tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-12.4%, Trades=6, WR=33.3%, Sharpe=-3.41, PF=0.59, DD=-17.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7075, Sharpe CI=[-15.01, 4.52], WR CI=[0.0%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.0%, Median equity=$867, Survival=100.0% |
| Regime | **PASS** | bull:6t/-11.3% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.6%, Trades=7, WR=28.6%, Sharpe=-0.81, PF=0.61, DD=-14.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6370, Sharpe CI=[-149.55, 3.39], WR CI=[0.0%, 57.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.2%, Median equity=$908, Survival=100.0% |
| Regime | **PASS** | bull:7t/-7.7% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=15% [multi-TF]** | FAIL | FAIL | **PASS** | **PASS** | **-0.81** | **-9.6%** | 7 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | **PASS** | -0.95 | -7.3% | 5 |
| WF tune: PT=6% | FAIL | FAIL | **PASS** | **PASS** | -3.41 | -12.4% | 6 |
| WF tune: PT=7% | FAIL | FAIL | **PASS** | **PASS** | -3.41 | -12.4% | 6 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | -3.47 | -12.3% | 5 |

---

## 5. Final Recommendation

**OKLO partially validates.** Best config: WF tune: PT=15% [multi-TF] (2/4 gates).

### WF tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.6%, Trades=7, WR=28.6%, Sharpe=-0.81, PF=0.61, DD=-14.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6370, Sharpe CI=[-149.55, 3.39], WR CI=[0.0%, 57.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.2%, Median equity=$908, Survival=100.0% |
| Regime | **PASS** | bull:7t/-7.7% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

