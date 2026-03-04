# XOM (ExxonMobil) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 31.8 minutes
**Category:** Large-cap integrated oil

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

XOM — Largest US integrated oil major — upstream, downstream, chemicals. Large-cap integrated oil.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 10.0% | -28.2% | -0.56 | 0.18 | -40.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 19 | 42.1% | +20.7% | 0.24 | 1.34 | -29.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 15.4% | -15.0% | -0.42 | 0.49 | -29.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 10 | 10.0% | -27.7% | -0.54 | 0.19 | -40.1% |
| Alt D: Energy rules (14 rules, 10%/5%) | 21 | 47.6% | +29.6% | 0.40 | 1.56 | -24.4% |
| Alt E: integrated sector rules (3 rules, 10%/5%) | 14 | 28.6% | -18.6% | -0.37 | 0.46 | -34.4% |

**Best baseline selected for validation: Alt D: Energy rules (14 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Energy rules (14 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+29.6%, Trades=21, WR=47.6%, Sharpe=0.40, PF=1.56, DD=-24.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.11, Test Sharpe=1.22, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1320, Sharpe CI=[-1.49, 4.99], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.8%, Median equity=$1,388, Survival=100.0% |
| Regime | **PASS** | bull:16t/+13.0%, chop:2t/+6.7%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 21 | 47.6% | +40.5% | 0.50 |
| WF tune: ATR stops x2.5 | 21 | 47.6% | +31.7% | 0.43 |
| WF tune: conf=0.45 | 21 | 47.6% | +29.6% | 0.40 |
| BS tune: energy rules (14) | 21 | 47.6% | +29.6% | 0.40 |
| WF tune: PT=12% | 18 | 44.4% | +27.9% | 0.38 |
| BS tune: full rules (10) | 19 | 42.1% | +20.7% | 0.24 |
| WF tune: conf=0.65 | 17 | 41.2% | +15.8% | 0.19 |
| WF tune: PT=8% | 26 | 46.2% | +13.9% | 0.17 |
| WF tune: PT=15% | 16 | 37.5% | +11.4% | 0.13 |
| WF tune: cooldown=7 | 19 | 42.1% | +2.7% | -0.04 |
| WF tune: conf=0.55 | 19 | 36.8% | -0.4% | -0.05 |
| WF tune: conf=0.6 | 19 | 36.8% | -0.4% | -0.05 |
| BS tune: sector-specific rules | 16 | 31.2% | -16.7% | -0.33 |
| BS tune: conf=0.4 [multi-TF] | 80 | 45.0% | +6.2% | 0.07 |

### Full Validation of Top Candidates

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.5%, Trades=21, WR=47.6%, Sharpe=0.50, PF=1.76, DD=-22.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.00, Test Sharpe=1.22, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0868, Sharpe CI=[-1.01, 5.56], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.0%, Median equity=$1,515, Survival=100.0% |
| Regime | **PASS** | bull:16t/+22.2%, chop:2t/+6.7%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### WF tune: ATR stops x2.5

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+31.7%, Trades=21, WR=47.6%, Sharpe=0.43, PF=1.61, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.08, Test Sharpe=1.22, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1267, Sharpe CI=[-1.41, 5.06], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.3%, Median equity=$1,400, Survival=100.0% |
| Regime | **PASS** | bull:16t/+13.7%, chop:2t/+6.7%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+29.6%, Trades=21, WR=47.6%, Sharpe=0.40, PF=1.56, DD=-24.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.11, Test Sharpe=1.22, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1320, Sharpe CI=[-1.49, 4.99], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.8%, Median equity=$1,388, Survival=100.0% |
| Regime | **PASS** | bull:16t/+13.0%, chop:2t/+6.7%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+6.2%, Trades=80, WR=45.0%, Sharpe=0.07, PF=1.07, DD=-35.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.45, Test Sharpe=1.08, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2449, Sharpe CI=[-1.15, 2.01], WR CI=[45.0%, 67.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.8%, Median equity=$1,255, Survival=100.0% |
| Regime | **PASS** | bull:60t/+23.6%, bear:5t/-0.3%, chop:8t/+3.4%, volatile:6t/+8.2%, crisis:1t/-5.1% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | FAIL | FAIL | **PASS** | **PASS** | **0.50** | **+40.5%** | 21 |
| WF tune: ATR stops x2.5 | FAIL | FAIL | **PASS** | **PASS** | 0.43 | +31.7% | 21 |
| Alt D: Energy rules (14 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.40 | +29.6% | 21 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | **PASS** | 0.40 | +29.6% | 21 |
| BS tune: conf=0.4 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.07 | +6.2% | 80 |

---

## 5. Final Recommendation

**XOM partially validates.** Best config: BS tune: conf=0.4 (2/4 gates).

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.5%, Trades=21, WR=47.6%, Sharpe=0.50, PF=1.76, DD=-22.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.00, Test Sharpe=1.22, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0868, Sharpe CI=[-1.01, 5.56], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.0%, Median equity=$1,515, Survival=100.0% |
| Regime | **PASS** | bull:16t/+22.2%, chop:2t/+6.7%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

