# WMT (Walmart) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 32.0 minutes
**Category:** Mass retail

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

WMT — Mass retail — beta 0.26-0.66, e-commerce growth hybrid. Mass retail.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 58.3% | +56.3% | 0.69 | 3.06 | -12.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 20 | 60.0% | +83.2% | 0.60 | 2.30 | -23.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 16 | 50.0% | +53.4% | 0.64 | 2.40 | -13.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 11 | 72.7% | +93.4% | 0.84 | 6.83 | -10.6% |
| Alt D: Staples rules (13 rules, 10%/5%) | 22 | 68.2% | +121.6% | 0.87 | 3.32 | -15.7% |
| Alt E: mass_retail lean (4 rules, 10%/5%) | 17 | 64.7% | +90.7% | 0.80 | 4.23 | -12.3% |
| Alt F: Mass retail balanced (8%/4%) | 21 | 66.7% | +87.8% | 0.75 | 3.23 | -10.7% |
| Alt G: Mass retail moderate (7%/4%) | 23 | 56.5% | +32.5% | 0.40 | 1.73 | -14.1% |

**Best baseline selected for validation: Alt D: Staples rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Staples rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+121.6%, Trades=22, WR=68.2%, Sharpe=0.87, PF=3.32, DD=-15.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.80, Test Sharpe=1.26, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[1.25, 8.95], WR CI=[50.0%, 86.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.3%, Median equity=$2,468, Survival=100.0% |
| Regime | FAIL | bull:16t/+85.2%, bear:1t/+10.5%, chop:1t/-9.9%, volatile:3t/-1.7%, crisis:1t/+13.8% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 24 | 62.5% | +121.2% | 0.89 |
| Regime tune: staples rules (13) | 22 | 68.2% | +121.6% | 0.87 |
| Regime tune: PT=15% | 17 | 58.8% | +91.1% | 0.66 |
| Regime tune: full rules (10) | 20 | 60.0% | +83.2% | 0.60 |
| Regime tune: PT=8% | 28 | 60.7% | +80.8% | 0.58 |
| Regime tune: PT=12% | 21 | 57.1% | +69.9% | 0.56 |
| Regime tune: PT=7% | 31 | 61.3% | +65.5% | 0.54 |
| Regime tune: conf=0.65 | 24 | 50.0% | +31.7% | 0.29 |
| Regime tune: tighter stop 4% [multi-TF] | 37 | 51.4% | +86.5% | 0.59 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+121.2%, Trades=24, WR=62.5%, Sharpe=0.89, PF=3.20, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.80, Test Sharpe=1.21, Ratio=152% (need >=50%) |
| Bootstrap | **PASS** | p=0.0019, Sharpe CI=[1.43, 7.84], WR CI=[45.8%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.0%, Median equity=$2,462, Survival=100.0% |
| Regime | **PASS** | bull:17t/+70.9%, bear:2t/+21.4%, chop:1t/-4.2%, volatile:3t/-1.7%, crisis:1t/+10.6% |

**Result: 4/4 gates passed**

---

### Regime tune: staples rules (13)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+121.6%, Trades=22, WR=68.2%, Sharpe=0.87, PF=3.32, DD=-15.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.80, Test Sharpe=1.26, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[1.25, 8.95], WR CI=[50.0%, 86.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.3%, Median equity=$2,468, Survival=100.0% |
| Regime | FAIL | bull:16t/+85.2%, bear:1t/+10.5%, chop:1t/-9.9%, volatile:3t/-1.7%, crisis:1t/+13.8% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+91.1%, Trades=17, WR=58.8%, Sharpe=0.66, PF=3.02, DD=-24.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.54, Test Sharpe=1.20, Ratio=221% (need >=50%) |
| Bootstrap | **PASS** | p=0.0138, Sharpe CI=[0.44, 7.95], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.1%, Median equity=$2,071, Survival=100.0% |
| Regime | **PASS** | bull:11t/+61.7%, bear:1t/+15.3%, chop:1t/-9.9%, volatile:3t/-1.7%, crisis:1t/+15.7% |

**Result: 4/4 gates passed**

---

### Regime tune: tighter stop 4% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+86.5%, Trades=37, WR=51.4%, Sharpe=0.59, PF=2.16, DD=-19.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.59, Test Sharpe=1.07, Ratio=182% (need >=50%) |
| Bootstrap | **PASS** | p=0.0123, Sharpe CI=[0.38, 4.97], WR CI=[40.5%, 73.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.4%, Median equity=$2,104, Survival=100.0% |
| Regime | FAIL | bull:19t/+68.9%, bear:5t/-12.9%, chop:3t/+15.9%, volatile:8t/+3.8%, crisis:2t/+5.7% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: tighter stop 4%** | **PASS** | **PASS** | **PASS** | **PASS** | **0.89** | **+121.2%** | 24 |
| Regime tune: PT=15% | **PASS** | **PASS** | **PASS** | **PASS** | 0.66 | +91.1% | 17 |
| Alt D: Staples rules (13 rules, 10%/5%) | **PASS** | **PASS** | **PASS** | FAIL | 0.87 | +121.6% | 22 |
| Regime tune: staples rules (13) | **PASS** | **PASS** | **PASS** | FAIL | 0.87 | +121.6% | 22 |
| Regime tune: tighter stop 4% [multi-TF] | **PASS** | **PASS** | **PASS** | FAIL | 0.59 | +86.5% | 37 |

---

## 5. Final Recommendation

**WMT fully validates.** Best config: Regime tune: tighter stop 4% (4/4 gates).

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+121.2%, Trades=24, WR=62.5%, Sharpe=0.89, PF=3.20, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.80, Test Sharpe=1.21, Ratio=152% (need >=50%) |
| Bootstrap | **PASS** | p=0.0019, Sharpe CI=[1.43, 7.84], WR CI=[45.8%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.0%, Median equity=$2,462, Survival=100.0% |
| Regime | **PASS** | bull:17t/+70.9%, bear:2t/+21.4%, chop:1t/-4.2%, volatile:3t/-1.7%, crisis:1t/+10.6% |

**Result: 4/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

