# EWY (iShares MSCI South Korea ETF) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 13.0 minutes
**Category:** International ETF (tech-heavy)

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

EWY — South Korea country ETF — ~40% Samsung/SK Hynix, tech-heavy, beta ~1.1. International ETF (tech-heavy).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 55.6% | +32.4% | 0.39 | 2.51 | -13.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 52.9% | +56.4% | 0.49 | 2.35 | -18.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 33.3% | +9.1% | 0.10 | 1.30 | -27.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 50.0% | +25.3% | 0.29 | 2.28 | -18.9% |
| Alt D: Recommended rules (4 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt E: Tech full (13 rules, 10%/5%) | 17 | 52.9% | +56.4% | 0.49 | 2.35 | -18.2% |
| Alt F: Tech momentum (15%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt G: Tech reversion (8%/4%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+56.4%, Trades=17, WR=52.9%, Sharpe=0.49, PF=2.35, DD=-18.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.19, Test Sharpe=1.34, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0422, Sharpe CI=[-0.44, 6.90], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.3%, Median equity=$1,673, Survival=100.0% |
| Regime | FAIL | bull:15t/+51.9%, chop:2t/+6.0% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=12% | 17 | 52.9% | +56.6% | 0.56 |
| WF tune: PT=7% | 19 | 63.2% | +42.2% | 0.53 |
| WF tune: PT=15% | 12 | 58.3% | +62.5% | 0.52 |
| WF tune: PT=8% | 18 | 55.6% | +39.9% | 0.50 |
| WF tune: PT=6% | 23 | 65.2% | +52.1% | 0.50 |
| WF tune: conf=0.45 | 17 | 52.9% | +56.4% | 0.49 |
| BS tune: conf=0.4 | 17 | 52.9% | +56.4% | 0.49 |
| BS tune: full rules (10) | 17 | 52.9% | +56.4% | 0.49 |
| WF tune: cooldown=7 | 15 | 60.0% | +52.8% | 0.49 |
| WF tune: ATR stops x2.5 | 18 | 50.0% | +49.4% | 0.44 |
| WF tune: conf=0.65 | 17 | 52.9% | +39.8% | 0.37 |
| WF tune: conf=0.6 | 19 | 52.6% | +37.9% | 0.36 |
| WF tune: conf=0.55 | 20 | 50.0% | +34.4% | 0.33 |
| Regime tune: tighter stop 4% | 22 | 40.9% | +30.7% | 0.29 |
| WF tune: recommended rules | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: PT=12% [multi-TF] | 68 | 55.9% | +20.2% | 0.23 |

### Full Validation of Top Candidates

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+56.6%, Trades=17, WR=52.9%, Sharpe=0.56, PF=2.29, DD=-17.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.02, Test Sharpe=1.30, Ratio=6298% (need >=50%) |
| Bootstrap | FAIL | p=0.0435, Sharpe CI=[-0.47, 6.94], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.6%, Median equity=$1,680, Survival=100.0% |
| Regime | FAIL | bull:15t/+52.5%, chop:2t/+5.9% |

**Result: 2/4 gates passed**

---

### WF tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+42.2%, Trades=19, WR=63.2%, Sharpe=0.53, PF=1.88, DD=-20.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.07, Test Sharpe=1.30, Ratio=1835% (need >=50%) |
| Bootstrap | FAIL | p=0.0716, Sharpe CI=[-0.86, 6.94], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.6%, Median equity=$1,521, Survival=100.0% |
| Regime | FAIL | bull:16t/+33.7%, chop:3t/+13.5% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+62.5%, Trades=12, WR=58.3%, Sharpe=0.52, PF=3.25, DD=-18.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.14, Test Sharpe=1.32, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0278, Sharpe CI=[-0.07, 8.88], WR CI=[33.3%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.7%, Median equity=$1,774, Survival=100.0% |
| Regime | FAIL | bull:10t/+53.3%, chop:2t/+11.1% |

**Result: 1/4 gates passed**

---

### WF tune: PT=12% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.2%, Trades=68, WR=55.9%, Sharpe=0.23, PF=1.22, DD=-29.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.87, Test Sharpe=1.26, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1412, Sharpe CI=[-0.84, 2.41], WR CI=[45.6%, 69.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.7%, Median equity=$1,425, Survival=100.0% |
| Regime | **PASS** | bull:42t/+28.9%, bear:9t/+4.9%, chop:11t/+6.2%, volatile:6t/+1.7% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=12%** | **PASS** | FAIL | **PASS** | FAIL | **0.56** | **+56.6%** | 17 |
| WF tune: PT=7% | **PASS** | FAIL | **PASS** | FAIL | 0.53 | +42.2% | 19 |
| WF tune: PT=12% [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.23 | +20.2% | 68 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.52 | +62.5% | 12 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.49 | +56.4% | 17 |

---

## 5. Final Recommendation

**EWY partially validates.** Best config: WF tune: PT=12% (2/4 gates).

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+56.6%, Trades=17, WR=52.9%, Sharpe=0.56, PF=2.29, DD=-17.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.02, Test Sharpe=1.30, Ratio=6298% (need >=50%) |
| Bootstrap | FAIL | p=0.0435, Sharpe CI=[-0.47, 6.94], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.6%, Median equity=$1,680, Survival=100.0% |
| Regime | FAIL | bull:15t/+52.5%, chop:2t/+5.9% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

