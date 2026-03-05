# CEG (Constellation Energy) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 64.8 minutes
**Category:** Nuclear / AI power (MOMENTUM)

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

CEG — Largest US nuclear fleet (21 reactors) — AI data center play, beta 1.1-1.6. Nuclear / AI power (MOMENTUM).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 15 | 33.3% | -12.0% | -0.43 | 0.80 | -28.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 31 | 35.5% | +25.6% | 0.33 | 1.16 | -29.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 15 | 33.3% | -7.0% | -0.25 | 0.87 | -25.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 13 | 23.1% | -17.0% | -0.72 | 0.70 | -28.9% |
| Alt D: Utility rules (13 rules, 10%/5%) | 31 | 35.5% | +25.6% | 0.33 | 1.16 | -29.0% |
| Alt E: nuclear_power lean (3 rules, 10%/5%) | 16 | 31.2% | -16.4% | -0.52 | 0.75 | -32.4% |
| Alt F: Nuclear momentum (15%/8%) | 12 | 33.3% | -3.3% | -0.13 | 0.96 | -40.1% |
| Alt G: Nuclear energy rules (12%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt H: Nuclear wide (20%/10%) | 9 | 33.3% | -14.7% | -0.46 | 0.78 | -41.0% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+25.6%, Trades=31, WR=35.5%, Sharpe=0.33, PF=1.16, DD=-29.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.62, Test Sharpe=-0.25, Ratio=-40% (need >=50%) |
| Bootstrap | FAIL | p=0.2560, Sharpe CI=[-2.01, 3.14], WR CI=[19.4%, 51.6%] |
| Monte Carlo | FAIL | Ruin=1.4%, P95 DD=-53.2%, Median equity=$1,289, Survival=98.6% |
| Regime | FAIL | bull:27t/+49.7%, chop:3t/-1.9%, crisis:1t/-2.4% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 24 | 37.5% | +40.3% | 0.56 |
| WF tune: conf=0.55 | 25 | 36.0% | +33.6% | 0.47 |
| WF tune: conf=0.6 | 25 | 36.0% | +33.6% | 0.47 |
| WF tune: PT=8% | 34 | 41.2% | +26.1% | 0.38 |
| WF tune: PT=12% | 29 | 34.5% | +38.6% | 0.36 |
| WF tune: conf=0.45 | 31 | 35.5% | +25.6% | 0.33 |
| WF tune: ATR stops x2.5 | 31 | 35.5% | +25.6% | 0.33 |
| WF tune: + utility_mean_reversion | 31 | 35.5% | +25.6% | 0.33 |
| WF tune: + utility_rate_reversion | 31 | 35.5% | +25.6% | 0.33 |
| BS tune: conf=0.4 | 31 | 35.5% | +25.6% | 0.33 |
| BS tune: full rules (10) | 31 | 35.5% | +25.6% | 0.33 |
| BS tune: utility rules (13) | 31 | 35.5% | +25.6% | 0.33 |
| MC tune: ATR stops x2.0 | 31 | 35.5% | +25.6% | 0.33 |
| WF tune: PT=7% | 36 | 47.2% | +12.3% | 0.31 |
| MC tune: max_loss=3.0% | 30 | 30.0% | +11.4% | 0.25 |
| MC tune: max_loss=4.0% | 30 | 33.3% | +3.9% | -0.02 |
| WF tune: cooldown=7 | 27 | 33.3% | +3.9% | -0.02 |
| BS tune: nuclear_power rules | 16 | 31.2% | -17.2% | -0.49 |
| WF tune: PT=15% | 28 | 25.0% | -22.0% | -0.71 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.3%, Trades=24, WR=37.5%, Sharpe=0.56, PF=1.40, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.58, Test Sharpe=0.32, Ratio=55% (need >=50%) |
| Bootstrap | FAIL | p=0.1589, Sharpe CI=[-1.57, 4.11], WR CI=[20.8%, 58.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.4%, Median equity=$1,510, Survival=100.0% |
| Regime | FAIL | bull:22t/+45.4%, chop:1t/+11.7%, volatile:1t/-2.4% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+33.6%, Trades=25, WR=36.0%, Sharpe=0.47, PF=1.33, DD=-22.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.58, Test Sharpe=0.32, Ratio=55% (need >=50%) |
| Bootstrap | FAIL | p=0.1926, Sharpe CI=[-1.82, 3.85], WR CI=[20.0%, 56.0%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-42.2%, Median equity=$1,411, Survival=99.9% |
| Regime | FAIL | bull:22t/+45.4%, chop:2t/+5.1%, volatile:1t/-2.4% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+33.6%, Trades=25, WR=36.0%, Sharpe=0.47, PF=1.33, DD=-22.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.58, Test Sharpe=0.32, Ratio=55% (need >=50%) |
| Bootstrap | FAIL | p=0.1926, Sharpe CI=[-1.82, 3.85], WR CI=[20.0%, 56.0%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-42.2%, Median equity=$1,411, Survival=99.9% |
| Regime | FAIL | bull:22t/+45.4%, chop:2t/+5.1%, volatile:1t/-2.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | **PASS** | FAIL | FAIL | FAIL | **0.56** | **+40.3%** | 24 |
| WF tune: conf=0.55 | **PASS** | FAIL | FAIL | FAIL | 0.47 | +33.6% | 25 |
| WF tune: conf=0.6 | **PASS** | FAIL | FAIL | FAIL | 0.47 | +33.6% | 25 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | 0.33 | +25.6% | 31 |

---

## 5. Final Recommendation

**CEG partially validates.** Best config: WF tune: conf=0.65 (1/4 gates).

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.3%, Trades=24, WR=37.5%, Sharpe=0.56, PF=1.40, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.58, Test Sharpe=0.32, Ratio=55% (need >=50%) |
| Bootstrap | FAIL | p=0.1589, Sharpe CI=[-1.57, 4.11], WR CI=[20.8%, 58.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.4%, Median equity=$1,510, Survival=100.0% |
| Regime | FAIL | bull:22t/+45.4%, chop:1t/+11.7%, volatile:1t/-2.4% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

