# SOFI (SoFi Technologies) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 40.3 minutes
**Category:** Mid-cap fintech

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

SOFI — Fintech — digital banking, lending, investing platform. Mid-cap fintech.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 36.4% | -3.4% | -0.20 | 0.94 | -24.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 36 | 41.7% | +12.5% | 0.18 | 1.07 | -35.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 36.4% | -3.4% | -0.20 | 0.94 | -24.2% |
| Alt C: Wider PT (3 rules, 12%/5%) | 10 | 30.0% | -9.3% | -0.35 | 0.82 | -24.2% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 11 | 36.4% | -3.4% | -0.20 | 0.94 | -24.2% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+12.5%, Trades=36, WR=41.7%, Sharpe=0.18, PF=1.07, DD=-35.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.50, Test Sharpe=-1.26, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2819, Sharpe CI=[-1.76, 3.04], WR CI=[25.0%, 58.3%] |
| Monte Carlo | FAIL | Ruin=3.0%, P95 DD=-56.8%, Median equity=$1,212, Survival=97.0% |
| Regime | **PASS** | bull:34t/+31.5%, chop:1t/-6.2%, volatile:1t/+14.6% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 28 | 50.0% | +78.7% | 0.78 |
| MC tune: max_loss=3.0% | 42 | 35.7% | +35.5% | 0.44 |
| MC tune: max_loss=4.0% | 39 | 38.5% | +20.4% | 0.30 |
| WF tune: conf=0.55 | 25 | 40.0% | +18.6% | 0.29 |
| WF tune: conf=0.6 | 25 | 40.0% | +18.6% | 0.29 |
| WF tune: conf=0.65 | 25 | 40.0% | +18.6% | 0.29 |
| WF tune: conf=0.45 | 36 | 41.7% | +12.5% | 0.18 |
| BS tune: conf=0.4 | 36 | 41.7% | +12.5% | 0.18 |
| BS tune: full rules (10) | 36 | 41.7% | +12.5% | 0.18 |
| BS tune: + volume_breakout | 36 | 41.7% | +12.5% | 0.18 |
| WF tune: PT=15% | 33 | 33.3% | -4.5% | 0.01 |
| WF tune: PT=8% | 36 | 41.7% | -0.4% | -0.01 |
| WF tune: PT=12% | 34 | 35.3% | -13.8% | -0.17 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+78.7%, Trades=28, WR=50.0%, Sharpe=0.78, PF=1.54, DD=-30.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.22, Test Sharpe=0.49, Ratio=40% (need >=50%) |
| Bootstrap | FAIL | p=0.0786, Sharpe CI=[-0.78, 4.76], WR CI=[32.1%, 67.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-44.4%, Median equity=$1,948, Survival=99.9% |
| Regime | FAIL | bull:26t/+76.5%, chop:1t/-6.2%, volatile:1t/+14.6% |

**Result: 0/4 gates passed**

---

### MC tune: max_loss=3.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 3 bars

**Performance:** Return=+35.5%, Trades=42, WR=35.7%, Sharpe=0.44, PF=1.19, DD=-33.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.10, Test Sharpe=-1.49, Ratio=-136% (need >=50%) |
| Bootstrap | FAIL | p=0.1896, Sharpe CI=[-1.35, 3.03], WR CI=[21.4%, 50.0%] |
| Monte Carlo | FAIL | Ruin=0.4%, P95 DD=-49.1%, Median equity=$1,482, Survival=99.6% |
| Regime | FAIL | bull:40t/+48.8%, chop:1t/-6.2%, volatile:1t/+14.6% |

**Result: 0/4 gates passed**

---

### MC tune: max_loss=4.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.4%, Trades=39, WR=38.5%, Sharpe=0.30, PF=1.11, DD=-33.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.60, Test Sharpe=-1.49, Ratio=-248% (need >=50%) |
| Bootstrap | FAIL | p=0.2537, Sharpe CI=[-1.60, 2.93], WR CI=[23.1%, 53.8%] |
| Monte Carlo | FAIL | Ruin=1.8%, P95 DD=-54.4%, Median equity=$1,299, Survival=98.2% |
| Regime | FAIL | bull:37t/+37.3%, chop:1t/-6.2%, volatile:1t/+14.6% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%)** | FAIL | FAIL | FAIL | **PASS** | **0.18** | **+12.5%** | 36 |
| WF tune: cooldown=7 | FAIL | FAIL | FAIL | FAIL | 0.78 | +78.7% | 28 |
| MC tune: max_loss=3.0% | FAIL | FAIL | FAIL | FAIL | 0.44 | +35.5% | 42 |
| MC tune: max_loss=4.0% | FAIL | FAIL | FAIL | FAIL | 0.30 | +20.4% | 39 |

---

## 5. Final Recommendation

**SOFI partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) (1/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+12.5%, Trades=36, WR=41.7%, Sharpe=0.18, PF=1.07, DD=-35.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.50, Test Sharpe=-1.26, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2819, Sharpe CI=[-1.76, 3.04], WR CI=[25.0%, 58.3%] |
| Monte Carlo | FAIL | Ruin=3.0%, P95 DD=-56.8%, Median equity=$1,212, Survival=97.0% |
| Regime | **PASS** | bull:34t/+31.5%, chop:1t/-6.2%, volatile:1t/+14.6% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

