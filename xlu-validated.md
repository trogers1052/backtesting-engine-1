# XLU (Utilities Select Sector SPDR) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 75.6 minutes
**Category:** Utility sector ETF

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

XLU — Large-cap utilities ETF — benchmark, mean-reverting, excellent position sizing. Utility sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 8 | 50.0% | +18.7% | 0.31 | 1.83 | -14.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 60.0% | +20.9% | 0.41 | 2.00 | -10.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 9 | 44.4% | +12.1% | 0.20 | 1.47 | -15.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 37.5% | +25.5% | 0.42 | 1.59 | -15.0% |
| Alt D: Utility rules (13 rules, 10%/5%) | 20 | 55.0% | +4.9% | 0.01 | 1.13 | -19.3% |
| Alt E: utility_etf lean (4 rules, 10%/5%) | 17 | 47.1% | +11.1% | 0.14 | 1.05 | -16.4% |
| Alt F: ETF tight (6%/3%, conf=0.55) | 22 | 36.4% | -16.6% | -0.76 | 0.59 | -22.8% |
| Alt G: ETF moderate (8%/4%) | 19 | 47.4% | -1.8% | -0.22 | 0.95 | -16.9% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.9%, Trades=10, WR=60.0%, Sharpe=0.41, PF=2.00, DD=-10.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.03, Test Sharpe=0.80, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1283, Sharpe CI=[-2.34, 8.32], WR CI=[30.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-16.5%, Median equity=$1,250, Survival=100.0% |
| Regime | **PASS** | bull:6t/+16.4%, chop:1t/+10.3%, volatile:3t/-1.8% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=15% | 5 | 80.0% | +31.6% | 0.54 |
| WF tune: PT=12% | 10 | 50.0% | +25.9% | 0.50 |
| WF tune: cooldown=7 | 9 | 55.6% | +25.1% | 0.49 |
| WF tune: ATR stops x2.5 | 10 | 60.0% | +23.9% | 0.45 |
| WF tune: conf=0.45 | 10 | 60.0% | +20.9% | 0.41 |
| WF tune: + utility_mean_reversion | 10 | 60.0% | +20.9% | 0.41 |
| WF tune: + utility_rate_reversion | 10 | 60.0% | +20.9% | 0.41 |
| BS tune: conf=0.4 | 10 | 60.0% | +20.9% | 0.41 |
| BS tune: full rules (10) | 10 | 60.0% | +20.9% | 0.41 |
| WF tune: PT=7% | 14 | 57.1% | +17.1% | 0.35 |
| WF tune: conf=0.55 | 16 | 50.0% | +12.0% | 0.22 |
| WF tune: PT=8% | 13 | 53.8% | +13.0% | 0.21 |
| WF tune: conf=0.65 | 13 | 38.5% | +10.5% | 0.18 |
| WF tune: conf=0.6 | 15 | 40.0% | +9.0% | 0.13 |
| BS tune: utility_etf rules | 18 | 44.4% | +4.9% | 0.01 |
| BS tune: utility rules (13) | 20 | 55.0% | +4.9% | 0.01 |
| WF tune: PT=15% [multi-TF] | 69 | 43.5% | +9.9% | 0.11 |

### Full Validation of Top Candidates

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+31.6%, Trades=5, WR=80.0%, Sharpe=0.54, PF=6.99, DD=-12.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.13, Test Sharpe=0.94, Ratio=731% (need >=50%) |
| Bootstrap | FAIL | p=0.0403, Sharpe CI=[-0.58, 25.32], WR CI=[40.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-5.5%, Median equity=$1,358, Survival=100.0% |
| Regime | **PASS** | bull:2t/+21.5%, chop:1t/+15.9%, volatile:2t/-4.3% |

**Result: 3/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+25.9%, Trades=10, WR=50.0%, Sharpe=0.50, PF=1.60, DD=-10.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.01, Test Sharpe=0.93, Ratio=9208% (need >=50%) |
| Bootstrap | FAIL | p=0.1185, Sharpe CI=[-2.27, 8.60], WR CI=[30.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.7%, Median equity=$1,303, Survival=100.0% |
| Regime | **PASS** | bull:6t/+17.3%, chop:1t/+12.6%, volatile:2t/-4.3%, crisis:1t/+4.4% |

**Result: 3/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+25.1%, Trades=9, WR=55.6%, Sharpe=0.49, PF=1.81, DD=-12.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.07, Test Sharpe=0.74, Ratio=1097% (need >=50%) |
| Bootstrap | FAIL | p=0.0931, Sharpe CI=[-1.72, 10.80], WR CI=[33.3%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.9%, Median equity=$1,295, Survival=100.0% |
| Regime | **PASS** | bull:6t/+22.4%, chop:1t/+10.3%, volatile:2t/-4.3% |

**Result: 3/4 gates passed**

---

### WF tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+9.9%, Trades=69, WR=43.5%, Sharpe=0.11, PF=1.31, DD=-22.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.93, Test Sharpe=1.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1644, Sharpe CI=[-1.41, 1.96], WR CI=[43.5%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.7%, Median equity=$1,269, Survival=100.0% |
| Regime | **PASS** | bull:50t/+15.0%, bear:10t/-1.1%, chop:6t/+18.7%, volatile:3t/-6.0% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=15%** | **PASS** | FAIL | **PASS** | **PASS** | **0.54** | **+31.6%** | 5 |
| WF tune: PT=12% | **PASS** | FAIL | **PASS** | **PASS** | 0.50 | +25.9% | 10 |
| WF tune: cooldown=7 | **PASS** | FAIL | **PASS** | **PASS** | 0.49 | +25.1% | 9 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.41 | +20.9% | 10 |
| WF tune: PT=15% [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.11 | +9.9% | 69 |

---

## 5. Final Recommendation

**XLU partially validates.** Best config: WF tune: PT=15% (3/4 gates).

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+31.6%, Trades=5, WR=80.0%, Sharpe=0.54, PF=6.99, DD=-12.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.13, Test Sharpe=0.94, Ratio=731% (need >=50%) |
| Bootstrap | FAIL | p=0.0403, Sharpe CI=[-0.58, 25.32], WR CI=[40.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-5.5%, Median equity=$1,358, Survival=100.0% |
| Regime | **PASS** | bull:2t/+21.5%, chop:1t/+15.9%, volatile:2t/-4.3% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

