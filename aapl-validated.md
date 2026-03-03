# AAPL (Apple) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 83.5 minutes
**Category:** Mega-cap tech

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

AAPL — Mega-cap consumer tech — iPhone, Mac, Services. Mega-cap tech.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 30.0% | -2.5% | -0.43 | 0.92 | -18.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 35.3% | +2.2% | -0.01 | 1.00 | -19.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 30.0% | -0.6% | -0.37 | 0.99 | -18.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 10 | 30.0% | +0.9% | -0.23 | 1.04 | -18.7% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 10 | 30.0% | -6.8% | -0.61 | 0.77 | -19.3% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+2.2%, Trades=17, WR=35.3%, Sharpe=-0.01, PF=1.00, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.17, Test Sharpe=0.20, Ratio=116% (need >=50%) |
| Bootstrap | FAIL | p=0.3666, Sharpe CI=[-3.39, 4.05], WR CI=[17.6%, 64.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.1%, Median equity=$1,073, Survival=100.0% |
| Regime | FAIL | bull:13t/+16.4%, chop:1t/-4.4%, volatile:3t/+0.1% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=15% | 12 | 25.0% | +3.4% | -0.01 |
| BS tune: conf=0.4 | 17 | 35.3% | +2.2% | -0.01 |
| BS tune: conf=0.45 | 17 | 35.3% | +2.2% | -0.01 |
| BS tune: full rules (10) | 17 | 35.3% | +2.2% | -0.01 |
| BS tune: + volume_breakout | 17 | 35.3% | +2.2% | -0.01 |
| Regime tune: PT=12% | 16 | 31.2% | +0.7% | -0.04 |
| BS tune: cooldown=7 | 17 | 35.3% | +2.0% | -0.06 |
| Regime tune: tighter stop 4% | 19 | 31.6% | -4.2% | -0.14 |
| Regime tune: conf=0.65 | 17 | 35.3% | -4.3% | -0.19 |
| BS tune: conf=0.55 | 19 | 31.6% | -8.8% | -0.26 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | 79 | 39.2% | -7.1% | -0.18 |

### Full Validation of Top Candidates

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+3.4%, Trades=12, WR=25.0%, Sharpe=-0.01, PF=1.03, DD=-18.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.05, Test Sharpe=0.39, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3622, Sharpe CI=[-6.64, 4.52], WR CI=[8.3%, 58.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.1%, Median equity=$1,085, Survival=100.0% |
| Regime | FAIL | bull:8t/+12.5%, chop:1t/-4.4%, volatile:3t/+4.9% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+2.2%, Trades=17, WR=35.3%, Sharpe=-0.01, PF=1.00, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.17, Test Sharpe=0.20, Ratio=116% (need >=50%) |
| Bootstrap | FAIL | p=0.3666, Sharpe CI=[-3.39, 4.05], WR CI=[17.6%, 64.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.1%, Median equity=$1,073, Survival=100.0% |
| Regime | FAIL | bull:13t/+16.4%, chop:1t/-4.4%, volatile:3t/+0.1% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+2.2%, Trades=17, WR=35.3%, Sharpe=-0.01, PF=1.00, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.17, Test Sharpe=0.20, Ratio=116% (need >=50%) |
| Bootstrap | FAIL | p=0.3666, Sharpe CI=[-3.39, 4.05], WR CI=[17.6%, 64.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.1%, Median equity=$1,073, Survival=100.0% |
| Regime | FAIL | bull:13t/+16.4%, chop:1t/-4.4%, volatile:3t/+0.1% |

**Result: 2/4 gates passed**

---

### Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-7.1%, Trades=79, WR=39.2%, Sharpe=-0.18, PF=0.93, DD=-27.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.03, Test Sharpe=0.14, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3857, Sharpe CI=[-1.50, 1.72], WR CI=[32.9%, 54.4%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.2%, Median equity=$1,081, Survival=100.0% |
| Regime | FAIL | bull:50t/+36.4%, bear:8t/+0.0%, chop:10t/-6.2%, volatile:11t/-15.3% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%)** | **PASS** | FAIL | **PASS** | FAIL | **-0.01** | **+2.2%** | 17 |
| BS tune: conf=0.4 | **PASS** | FAIL | **PASS** | FAIL | -0.01 | +2.2% | 17 |
| BS tune: conf=0.45 | **PASS** | FAIL | **PASS** | FAIL | -0.01 | +2.2% | 17 |
| Regime tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | -0.01 | +3.4% | 12 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | FAIL | FAIL | FAIL | FAIL | -0.18 | -7.1% | 79 |

---

## 5. Final Recommendation

**AAPL partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) (2/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+2.2%, Trades=17, WR=35.3%, Sharpe=-0.01, PF=1.00, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.17, Test Sharpe=0.20, Ratio=116% (need >=50%) |
| Bootstrap | FAIL | p=0.3666, Sharpe CI=[-3.39, 4.05], WR CI=[17.6%, 64.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.1%, Median equity=$1,073, Survival=100.0% |
| Regime | FAIL | bull:13t/+16.4%, chop:1t/-4.4%, volatile:3t/+0.1% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

