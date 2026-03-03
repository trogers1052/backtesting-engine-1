# XLE (Energy Select Sector SPDR ETF) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 16.5 minutes
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
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 15 | 40.0% | +3.4% | 0.01 | 1.05 | -27.3% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+3.4%, Trades=15, WR=40.0%, Sharpe=0.01, PF=1.05, DD=-27.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.59, Test Sharpe=0.70, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3740, Sharpe CI=[-3.78, 4.12], WR CI=[20.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.1%, Median equity=$1,062, Survival=100.0% |
| Regime | FAIL | bull:11t/+16.0%, bear:1t/-6.4%, volatile:3t/-0.1% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 15 | 40.0% | +12.7% | 0.15 |
| WF tune: PT=15% | 14 | 35.7% | +5.8% | 0.04 |
| WF tune: conf=0.45 | 15 | 40.0% | +3.4% | 0.01 |
| BS tune: conf=0.4 | 15 | 40.0% | +3.4% | 0.01 |
| BS tune: full rules (10) | 15 | 40.0% | +3.4% | 0.01 |
| BS tune: energy rules (12) | 15 | 40.0% | +3.4% | 0.01 |
| BS tune: + volume_breakout | 15 | 40.0% | +3.4% | 0.01 |
| BS tune: + commodity_breakout | 15 | 40.0% | +3.4% | 0.01 |
| Regime tune: + dollar_weakness | 15 | 40.0% | +3.4% | 0.01 |
| WF tune: PT=12% | 14 | 35.7% | +3.0% | 0.00 |
| WF tune: PT=8% | 18 | 44.4% | +1.5% | -0.04 |
| WF tune: cooldown=7 | 14 | 35.7% | -3.4% | -0.11 |
| WF tune: conf=0.6 | 17 | 47.1% | -4.5% | -0.20 |
| WF tune: conf=0.65 | 14 | 42.9% | -5.4% | -0.20 |
| WF tune: conf=0.55 | 19 | 42.1% | -8.1% | -0.26 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+12.7%, Trades=15, WR=40.0%, Sharpe=0.15, PF=1.36, DD=-22.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.29, Test Sharpe=0.70, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2393, Sharpe CI=[-2.89, 4.71], WR CI=[20.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.3%, Median equity=$1,163, Survival=100.0% |
| Regime | FAIL | bull:11t/+18.4%, bear:1t/-4.4%, volatile:3t/+4.1% |

**Result: 1/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+5.8%, Trades=14, WR=35.7%, Sharpe=0.04, PF=0.95, DD=-25.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.47, Test Sharpe=0.69, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3524, Sharpe CI=[-4.33, 4.29], WR CI=[21.4%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.6%, Median equity=$1,089, Survival=100.0% |
| Regime | FAIL | bull:10t/+14.7%, bear:1t/-6.8%, volatile:3t/+4.7% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+3.4%, Trades=15, WR=40.0%, Sharpe=0.01, PF=1.05, DD=-27.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.59, Test Sharpe=0.70, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3740, Sharpe CI=[-3.78, 4.12], WR CI=[20.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.1%, Median equity=$1,062, Survival=100.0% |
| Regime | FAIL | bull:11t/+16.0%, bear:1t/-6.4%, volatile:3t/-0.1% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: tighter stop 4%** | FAIL | FAIL | **PASS** | FAIL | **0.15** | **+12.7%** | 15 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.04 | +5.8% | 14 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.01 | +3.4% | 15 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.01 | +3.4% | 15 |

---

## 5. Final Recommendation

**XLE partially validates.** Best config: Regime tune: tighter stop 4% (1/4 gates).

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+12.7%, Trades=15, WR=40.0%, Sharpe=0.15, PF=1.36, DD=-22.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.29, Test Sharpe=0.70, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2393, Sharpe CI=[-2.89, 4.71], WR CI=[20.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.3%, Median equity=$1,163, Survival=100.0% |
| Regime | FAIL | bull:11t/+18.4%, bear:1t/-4.4%, volatile:3t/+4.1% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

