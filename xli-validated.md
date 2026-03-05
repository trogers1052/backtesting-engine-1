# XLI (Industrial Select Sector SPDR) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 20.4 minutes
**Category:** Industrial sector ETF

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

XLI — Large-cap industrials ETF — diversified across sub-sectors, benchmark. Industrial sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 71.4% | +38.3% | 0.93 | 4.67 | -8.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 12 | 58.3% | +31.3% | 0.75 | 2.34 | -15.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 62.5% | +35.4% | 0.91 | 3.66 | -12.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 6 | 66.7% | +40.0% | 0.74 | 5.98 | -11.5% |
| Alt D: Industrial rules (13 rules, 10%/5%) | 13 | 76.9% | +33.9% | 0.86 | 3.68 | -15.3% |
| Alt E: industrial_etf lean (4 rules, 10%/5%) | 12 | 66.7% | +37.9% | 0.78 | 2.92 | -10.1% |
| Alt F: ETF tight (8%/4%, conf=0.55) | 11 | 72.7% | +32.5% | 0.73 | 3.52 | -11.8% |
| Alt G: ETF moderate (10%/4%) | 13 | 61.5% | +33.2% | 0.61 | 2.23 | -13.6% |

**Best baseline selected for validation: Alt D: Industrial rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Industrial rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, industrial_mean_reversion, industrial_pullback, industrial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+33.9%, Trades=13, WR=76.9%, Sharpe=0.86, PF=3.68, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.69, Test Sharpe=1.17, Ratio=170% (need >=50%) |
| Bootstrap | FAIL | p=0.0303, Sharpe CI=[-0.19, 8.07], WR CI=[61.5%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.9%, Median equity=$1,425, Survival=100.0% |
| Regime | FAIL | bull:6t/+27.5%, bear:3t/+3.4%, chop:3t/+6.4%, crisis:1t/+0.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: conf=0.65 | 16 | 56.2% | +37.4% | 1.01 |
| Regime tune: PT=7% | 16 | 81.2% | +42.3% | 0.98 |
| Regime tune: tighter stop 4% | 16 | 68.8% | +36.1% | 0.95 |
| Regime tune: PT=12% | 12 | 75.0% | +38.5% | 0.91 |
| BS tune: cooldown=7 | 14 | 64.3% | +32.8% | 0.90 |
| BS tune: conf=0.55 | 34 | 64.7% | +40.4% | 0.89 |
| BS tune: conf=0.4 | 13 | 76.9% | +33.9% | 0.86 |
| BS tune: conf=0.45 | 13 | 76.9% | +33.9% | 0.86 |
| BS tune: industrial rules (13) | 13 | 76.9% | +33.9% | 0.86 |
| Regime tune: PT=8% | 17 | 76.5% | +43.7% | 0.85 |
| BS tune: full rules (10) | 12 | 58.3% | +31.3% | 0.75 |
| BS tune: industrial_etf rules | 11 | 72.7% | +32.6% | 0.65 |
| Regime tune: PT=15% | 9 | 77.8% | +29.0% | 0.54 |
| Regime tune: conf=0.65 [multi-TF] | 38 | 52.6% | +32.7% | 0.64 |

### Full Validation of Top Candidates

### Regime tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, industrial_mean_reversion, industrial_pullback, industrial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+37.4%, Trades=16, WR=56.2%, Sharpe=1.01, PF=3.25, DD=-10.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.96, Test Sharpe=0.76, Ratio=80% (need >=50%) |
| Bootstrap | **PASS** | p=0.0184, Sharpe CI=[0.27, 7.27], WR CI=[37.5%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.7%, Median equity=$1,462, Survival=100.0% |
| Regime | **PASS** | bull:11t/+23.8%, bear:1t/+7.1%, chop:3t/+8.9%, crisis:1t/+0.5% |

**Result: 4/4 gates passed**

---

### Regime tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, industrial_mean_reversion, industrial_pullback, industrial_seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+42.3%, Trades=16, WR=81.2%, Sharpe=0.98, PF=3.60, DD=-11.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.65, Test Sharpe=1.14, Ratio=175% (need >=50%) |
| Bootstrap | **PASS** | p=0.0100, Sharpe CI=[0.59, 10.52], WR CI=[62.5%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.2%, Median equity=$1,551, Survival=100.0% |
| Regime | **PASS** | bull:10t/+25.5%, bear:3t/+10.8%, chop:2t/+9.5%, crisis:1t/+0.5% |

**Result: 4/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, industrial_mean_reversion, industrial_pullback, industrial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.1%, Trades=16, WR=68.8%, Sharpe=0.95, PF=2.76, DD=-13.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.69, Test Sharpe=1.17, Ratio=171% (need >=50%) |
| Bootstrap | FAIL | p=0.0397, Sharpe CI=[-0.42, 7.04], WR CI=[50.0%, 93.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.8%, Median equity=$1,462, Survival=100.0% |
| Regime | FAIL | bull:9t/+36.9%, bear:3t/-3.1%, chop:3t/+6.8%, crisis:1t/+0.5% |

**Result: 2/4 gates passed**

---

### Regime tune: conf=0.65 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, industrial_mean_reversion, industrial_pullback, industrial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+32.7%, Trades=38, WR=52.6%, Sharpe=0.64, PF=1.69, DD=-15.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.24, Test Sharpe=0.88, Ratio=364% (need >=50%) |
| Bootstrap | FAIL | p=0.0679, Sharpe CI=[-0.59, 3.87], WR CI=[47.4%, 78.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.5%, Median equity=$1,475, Survival=100.0% |
| Regime | **PASS** | bull:24t/+25.2%, bear:3t/+0.9%, chop:6t/+6.7%, volatile:5t/+10.2% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: conf=0.65** | **PASS** | **PASS** | **PASS** | **PASS** | **1.01** | **+37.4%** | 16 |
| Regime tune: PT=7% | **PASS** | **PASS** | **PASS** | **PASS** | 0.98 | +42.3% | 16 |
| Regime tune: conf=0.65 [multi-TF] | **PASS** | FAIL | **PASS** | **PASS** | 0.64 | +32.7% | 38 |
| Regime tune: tighter stop 4% | **PASS** | FAIL | **PASS** | FAIL | 0.95 | +36.1% | 16 |
| Alt D: Industrial rules (13 rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.86 | +33.9% | 13 |

---

## 5. Final Recommendation

**XLI fully validates.** Best config: Regime tune: conf=0.65 (4/4 gates).

### Regime tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, industrial_mean_reversion, industrial_pullback, industrial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+37.4%, Trades=16, WR=56.2%, Sharpe=1.01, PF=3.25, DD=-10.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.96, Test Sharpe=0.76, Ratio=80% (need >=50%) |
| Bootstrap | **PASS** | p=0.0184, Sharpe CI=[0.27, 7.27], WR CI=[37.5%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.7%, Median equity=$1,462, Survival=100.0% |
| Regime | **PASS** | bull:11t/+23.8%, bear:1t/+7.1%, chop:3t/+8.9%, crisis:1t/+0.5% |

**Result: 4/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

