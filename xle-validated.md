# XLE (Energy Select Sector SPDR ETF) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 32.1 minutes
**Category:** Energy sector ETF

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

XLE — Broad energy sector basket — XOM, CVX top holdings. Energy sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 11.1% | -33.0% | -0.69 | 0.16 | -43.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 15 | 40.0% | +3.4% | 0.01 | 1.05 | -27.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 10.0% | -26.6% | -0.64 | 0.21 | -37.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 11.1% | -32.3% | -0.65 | 0.18 | -43.2% |
| Alt D: Energy rules (14 rules, 10%/5%) | 23 | 47.8% | +15.6% | 0.20 | 1.28 | -19.9% |
| Alt E: energy_etf sector rules (3 rules, 10%/5%) | 20 | 35.0% | -16.6% | -0.28 | 0.63 | -36.2% |

**Best baseline selected for validation: Alt D: Energy rules (14 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Energy rules (14 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+15.6%, Trades=23, WR=47.8%, Sharpe=0.20, PF=1.28, DD=-19.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.52, Test Sharpe=0.79, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2208, Sharpe CI=[-2.02, 4.10], WR CI=[30.4%, 73.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.3%, Median equity=$1,214, Survival=100.0% |
| Regime | FAIL | bull:18t/+27.4%, bear:1t/-6.4%, chop:1t/+3.1%, volatile:3t/-0.1% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 24 | 50.0% | +23.8% | 0.34 |
| WF tune: PT=8% | 27 | 48.1% | +17.3% | 0.27 |
| WF tune: PT=15% | 22 | 45.5% | +16.9% | 0.24 |
| WF tune: PT=12% | 22 | 50.0% | +16.2% | 0.21 |
| WF tune: conf=0.45 | 23 | 47.8% | +15.6% | 0.20 |
| BS tune: energy rules (14) | 23 | 47.8% | +15.6% | 0.20 |
| BS tune: conf=0.4 | 23 | 47.8% | +12.4% | 0.15 |
| WF tune: conf=0.65 | 22 | 45.5% | +11.0% | 0.14 |
| WF tune: ATR stops x2.5 | 25 | 48.0% | +8.8% | 0.09 |
| WF tune: conf=0.6 | 25 | 52.0% | +6.7% | 0.04 |
| BS tune: full rules (10) | 15 | 40.0% | +3.4% | 0.01 |
| WF tune: cooldown=7 | 22 | 45.5% | +4.4% | 0.01 |
| WF tune: conf=0.55 | 27 | 48.1% | +3.9% | -0.02 |
| BS tune: sector-specific rules | 20 | 35.0% | -15.8% | -0.26 |
| Regime tune: tighter stop 4% [multi-TF] | 89 | 44.9% | -28.9% | -0.44 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+23.8%, Trades=24, WR=50.0%, Sharpe=0.34, PF=1.47, DD=-19.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.25, Test Sharpe=0.79, Ratio=314% (need >=50%) |
| Bootstrap | FAIL | p=0.1419, Sharpe CI=[-1.49, 4.39], WR CI=[33.3%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.0%, Median equity=$1,312, Survival=100.0% |
| Regime | FAIL | bull:19t/+28.5%, bear:1t/-4.4%, chop:1t/+3.1%, volatile:3t/+4.1% |

**Result: 2/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+17.3%, Trades=27, WR=48.1%, Sharpe=0.27, PF=1.28, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.74, Test Sharpe=1.04, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2018, Sharpe CI=[-1.70, 3.97], WR CI=[33.3%, 70.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.3%, Median equity=$1,250, Survival=100.0% |
| Regime | **PASS** | bull:19t/+23.9%, bear:2t/-12.1%, chop:3t/+2.5%, volatile:3t/+12.6% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+16.9%, Trades=22, WR=45.5%, Sharpe=0.24, PF=1.22, DD=-19.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.23, Test Sharpe=0.80, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2154, Sharpe CI=[-2.13, 4.05], WR CI=[31.8%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$1,230, Survival=100.0% |
| Regime | FAIL | bull:17t/+24.9%, bear:1t/-6.8%, chop:1t/+3.1%, volatile:3t/+4.7% |

**Result: 1/4 gates passed**

---

### Regime tune: tighter stop 4% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=-28.9%, Trades=89, WR=44.9%, Sharpe=-0.44, PF=0.65, DD=-49.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.73, Test Sharpe=0.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7180, Sharpe CI=[-2.23, 0.99], WR CI=[42.7%, 62.9%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.2%, Median equity=$835, Survival=100.0% |
| Regime | FAIL | bull:67t/+2.3%, bear:5t/-13.3%, chop:10t/-9.9%, volatile:7t/+7.2% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: tighter stop 4%** | **PASS** | FAIL | **PASS** | FAIL | **0.34** | **+23.8%** | 24 |
| WF tune: PT=8% | FAIL | FAIL | **PASS** | **PASS** | 0.27 | +17.3% | 27 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.24 | +16.9% | 22 |
| Alt D: Energy rules (14 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.20 | +15.6% | 23 |
| Regime tune: tighter stop 4% [multi-TF] | FAIL | FAIL | FAIL | FAIL | -0.44 | -28.9% | 89 |

---

## 5. Final Recommendation

**XLE partially validates.** Best config: Regime tune: tighter stop 4% (2/4 gates).

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+23.8%, Trades=24, WR=50.0%, Sharpe=0.34, PF=1.47, DD=-19.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.25, Test Sharpe=0.79, Ratio=314% (need >=50%) |
| Bootstrap | FAIL | p=0.1419, Sharpe CI=[-1.49, 4.39], WR CI=[33.3%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.0%, Median equity=$1,312, Survival=100.0% |
| Regime | FAIL | bull:19t/+28.5%, bear:1t/-4.4%, chop:1t/+3.1%, volatile:3t/+4.1% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

