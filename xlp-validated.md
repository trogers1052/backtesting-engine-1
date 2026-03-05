# XLP (Consumer Staples Select Sector SPDR) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 24.8 minutes
**Category:** Consumer staples sector ETF

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

XLP — Large-cap consumer staples ETF — benchmark, mean-reverting, diversified. Consumer staples sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 11.1% | -18.5% | -0.72 | 0.32 | -18.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 12 | 33.3% | -6.1% | -0.45 | 0.62 | -11.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 10.0% | -18.3% | -0.92 | 0.32 | -18.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 11.1% | -17.9% | -0.68 | 0.35 | -18.2% |
| Alt D: Staples rules (13 rules, 10%/5%) | 20 | 60.0% | +6.2% | 0.03 | 1.13 | -13.5% |
| Alt E: staples_etf lean (4 rules, 10%/5%) | 18 | 61.1% | +5.8% | 0.02 | 1.13 | -12.3% |
| Alt F: ETF tight (6%/3%, conf=0.55) | 24 | 50.0% | -5.0% | -0.22 | 0.81 | -20.5% |
| Alt G: ETF moderate (8%/4%) | 20 | 50.0% | +4.1% | -0.01 | 1.07 | -16.9% |

**Best baseline selected for validation: Alt D: Staples rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Staples rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+6.2%, Trades=20, WR=60.0%, Sharpe=0.03, PF=1.13, DD=-13.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.02, Test Sharpe=-0.01, Ratio=-51% (need >=50%) |
| Bootstrap | FAIL | p=0.2515, Sharpe CI=[-2.24, 4.60], WR CI=[45.0%, 85.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.0%, Median equity=$1,118, Survival=100.0% |
| Regime | FAIL | bull:14t/+21.0%, bear:2t/-0.8%, chop:3t/-4.0%, volatile:1t/-3.4% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 33 | 60.6% | +8.3% | 0.10 |
| WF tune: PT=7% | 19 | 57.9% | +7.2% | 0.06 |
| BS tune: staples_etf rules | 19 | 57.9% | +6.7% | 0.04 |
| WF tune: PT=15% | 18 | 61.1% | +6.3% | 0.04 |
| WF tune: conf=0.45 | 20 | 60.0% | +6.2% | 0.03 |
| BS tune: conf=0.4 | 20 | 60.0% | +6.2% | 0.03 |
| BS tune: staples rules (13) | 20 | 60.0% | +6.2% | 0.03 |
| WF tune: PT=6% | 21 | 57.1% | +6.1% | 0.03 |
| WF tune: conf=0.55 | 34 | 55.9% | +6.3% | 0.03 |
| WF tune: PT=12% | 20 | 55.0% | +5.7% | 0.02 |
| WF tune: PT=8% | 20 | 60.0% | +5.5% | 0.01 |
| WF tune: conf=0.65 | 18 | 50.0% | +3.1% | -0.06 |
| WF tune: cooldown=7 | 20 | 55.0% | +1.5% | -0.08 |
| WF tune: ATR stops x2.5 | 22 | 50.0% | +0.8% | -0.08 |
| Regime tune: tighter stop 4% | 23 | 47.8% | +0.2% | -0.09 |
| BS tune: full rules (10) | 12 | 33.3% | -6.1% | -0.45 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+8.3%, Trades=33, WR=60.6%, Sharpe=0.10, PF=1.23, DD=-14.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.25, Test Sharpe=0.03, Ratio=10% (need >=50%) |
| Bootstrap | FAIL | p=0.1439, Sharpe CI=[-1.14, 4.68], WR CI=[57.6%, 87.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.3%, Median equity=$1,158, Survival=100.0% |
| Regime | FAIL | bull:26t/+22.3%, bear:2t/+1.1%, chop:4t/-8.0%, volatile:1t/+0.3% |

**Result: 1/4 gates passed**

---

### WF tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+7.2%, Trades=19, WR=57.9%, Sharpe=0.06, PF=1.21, DD=-13.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.02, Test Sharpe=-0.01, Ratio=-51% (need >=50%) |
| Bootstrap | FAIL | p=0.2120, Sharpe CI=[-1.92, 5.30], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.9%, Median equity=$1,124, Survival=100.0% |
| Regime | FAIL | bull:13t/+21.1%, bear:2t/-0.8%, chop:3t/-4.0%, volatile:1t/-3.4% |

**Result: 1/4 gates passed**

---

### BS tune: staples_etf rules

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+6.7%, Trades=19, WR=57.9%, Sharpe=0.04, PF=1.16, DD=-13.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.13, Test Sharpe=-0.02, Ratio=-18% (need >=50%) |
| Bootstrap | FAIL | p=0.2699, Sharpe CI=[-2.32, 4.62], WR CI=[47.4%, 89.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.9%, Median equity=$1,108, Survival=100.0% |
| Regime | FAIL | bull:13t/+21.2%, bear:3t/-5.3%, chop:3t/-4.0% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | FAIL | **PASS** | FAIL | **0.10** | **+8.3%** | 33 |
| WF tune: PT=7% | FAIL | FAIL | **PASS** | FAIL | 0.06 | +7.2% | 19 |
| BS tune: staples_etf rules | FAIL | FAIL | **PASS** | FAIL | 0.04 | +6.7% | 19 |
| Alt D: Staples rules (13 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.03 | +6.2% | 20 |

---

## 5. Final Recommendation

**XLP partially validates.** Best config: WF tune: conf=0.6 (1/4 gates).

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, consumer_staples_mean_reversion, consumer_staples_pullback, consumer_staples_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+8.3%, Trades=33, WR=60.6%, Sharpe=0.10, PF=1.23, DD=-14.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.25, Test Sharpe=0.03, Ratio=10% (need >=50%) |
| Bootstrap | FAIL | p=0.1439, Sharpe CI=[-1.14, 4.68], WR CI=[57.6%, 87.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.3%, Median equity=$1,158, Survival=100.0% |
| Regime | FAIL | bull:26t/+22.3%, bear:2t/+1.1%, chop:4t/-8.0%, volatile:1t/+0.3% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

