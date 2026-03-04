# FET (Forum Energy Technologies) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 45.5 minutes
**Category:** Small-cap oilfield services

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

FET — Oilfield services equipment — capex cycle play, NOT mining. Small-cap oilfield services.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 54.5% | +36.8% | 0.32 | 2.14 | -15.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 24 | 50.0% | +53.6% | 0.39 | 1.53 | -36.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 53.8% | +36.0% | 0.33 | 2.09 | -15.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 55.6% | +45.2% | 0.34 | 2.83 | -18.1% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 24 | 50.0% | +53.6% | 0.39 | 1.53 | -36.0% |
| Alt E: oilfield_services rules (3 rules, 10%/5%) | 22 | 45.5% | +40.6% | 0.32 | 1.57 | -34.2% |
| Alt F: OFS momentum (12%/6%) | 21 | 47.6% | +42.2% | 0.31 | 1.65 | -43.3% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+53.6%, Trades=24, WR=50.0%, Sharpe=0.39, PF=1.53, DD=-36.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.03, Test Sharpe=0.84, Ratio=2804% (need >=50%) |
| Bootstrap | FAIL | p=0.1226, Sharpe CI=[-1.26, 4.98], WR CI=[29.2%, 70.8%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.5%, Median equity=$1,646, Survival=99.9% |
| Regime | FAIL | bull:18t/+60.9%, bear:2t/+2.7%, chop:2t/-21.2%, volatile:2t/+22.4% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.55 | 20 | 60.0% | +92.0% | 0.63 |
| Regime tune: conf=0.65 | 20 | 60.0% | +92.0% | 0.63 |
| MC tune: max_loss=3.0% | 29 | 44.8% | +92.2% | 0.61 |
| Regime tune: PT=15% | 22 | 40.9% | +66.9% | 0.52 |
| BS tune: oilfield_services rules | 21 | 52.4% | +68.1% | 0.44 |
| MC tune: max_loss=4.0% | 30 | 43.3% | +54.5% | 0.42 |
| FET tune: energy momentum + wider PT (12%/6%) | 19 | 47.4% | +66.1% | 0.39 |
| BS tune: conf=0.4 | 24 | 50.0% | +53.6% | 0.39 |
| BS tune: conf=0.45 | 24 | 50.0% | +53.6% | 0.39 |
| BS tune: full rules (10) | 24 | 50.0% | +53.6% | 0.39 |
| BS tune: full mining rules (14) | 24 | 50.0% | +53.6% | 0.39 |
| BS tune: + volume_breakout | 24 | 50.0% | +53.6% | 0.39 |
| MC tune: ATR stops x2.0 | 24 | 50.0% | +53.6% | 0.39 |
| Regime tune: + commodity_breakout | 24 | 50.0% | +53.6% | 0.39 |
| Regime tune: + miner_metal_ratio | 24 | 50.0% | +53.6% | 0.39 |
| BS tune: cooldown=7 | 19 | 47.4% | +22.8% | 0.24 |
| Regime tune: PT=12% | 24 | 41.7% | +26.1% | 0.22 |
| FET tune: energy rules (14) | 33 | 42.4% | +6.3% | 0.12 |
| BS tune: conf=0.55 [multi-TF] | 55 | 50.9% | +44.0% | 0.32 |

### Full Validation of Top Candidates

### BS tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+92.0%, Trades=20, WR=60.0%, Sharpe=0.63, PF=2.11, DD=-19.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.01, Test Sharpe=0.84, Ratio=8945% (need >=50%) |
| Bootstrap | FAIL | p=0.0481, Sharpe CI=[-0.44, 7.20], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.4%, Median equity=$2,067, Survival=100.0% |
| Regime | FAIL | bull:16t/+72.7%, bear:3t/+30.7%, chop:1t/-16.7% |

**Result: 2/4 gates passed**

---

### Regime tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+92.0%, Trades=20, WR=60.0%, Sharpe=0.63, PF=2.11, DD=-19.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.01, Test Sharpe=0.84, Ratio=8945% (need >=50%) |
| Bootstrap | FAIL | p=0.0481, Sharpe CI=[-0.44, 7.20], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.4%, Median equity=$2,067, Survival=100.0% |
| Regime | FAIL | bull:16t/+72.7%, bear:3t/+30.7%, chop:1t/-16.7% |

**Result: 2/4 gates passed**

---

### MC tune: max_loss=3.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 3 bars

**Performance:** Return=+92.2%, Trades=29, WR=44.8%, Sharpe=0.61, PF=1.96, DD=-22.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.02, Test Sharpe=0.65, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0366, Sharpe CI=[-0.26, 5.12], WR CI=[27.6%, 62.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.9%, Median equity=$2,128, Survival=100.0% |
| Regime | **PASS** | bull:19t/+48.1%, bear:4t/+18.3%, chop:3t/+3.0%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.55 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+44.0%, Trades=55, WR=50.9%, Sharpe=0.32, PF=1.45, DD=-36.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.47, Test Sharpe=1.29, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0932, Sharpe CI=[-0.62, 3.12], WR CI=[40.0%, 65.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.1%, Median equity=$1,636, Survival=100.0% |
| Regime | **PASS** | bull:42t/+33.0%, bear:4t/+12.0%, chop:8t/+3.0%, volatile:1t/+10.9% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.55** | **PASS** | FAIL | **PASS** | FAIL | **0.63** | **+92.0%** | 20 |
| Regime tune: conf=0.65 | **PASS** | FAIL | **PASS** | FAIL | 0.63 | +92.0% | 20 |
| MC tune: max_loss=3.0% | FAIL | FAIL | **PASS** | **PASS** | 0.61 | +92.2% | 29 |
| BS tune: conf=0.55 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.32 | +44.0% | 55 |
| Alt A: Full general rules (10 rules, 10%/5%) | **PASS** | FAIL | FAIL | FAIL | 0.39 | +53.6% | 24 |

---

## 5. Final Recommendation

**FET partially validates.** Best config: BS tune: conf=0.55 (2/4 gates).

### BS tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+92.0%, Trades=20, WR=60.0%, Sharpe=0.63, PF=2.11, DD=-19.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.01, Test Sharpe=0.84, Ratio=8945% (need >=50%) |
| Bootstrap | FAIL | p=0.0481, Sharpe CI=[-0.44, 7.20], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.4%, Median equity=$2,067, Survival=100.0% |
| Regime | FAIL | bull:16t/+72.7%, bear:3t/+30.7%, chop:1t/-16.7% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

