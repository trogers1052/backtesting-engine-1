# GLD (SPDR Gold Trust) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 60.1 minutes
**Category:** Gold commodity ETF (WARNING: ~$250/share, too expensive for $888 account)

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

GLD — Physical gold commodity ETF — tracks gold spot price directly. Gold commodity ETF (WARNING: ~$250/share, too expensive for $888 account).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 14 | 71.4% | +75.1% | 0.78 | 8.47 | -9.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 87.5% | +134.0% | 0.82 | 151.33 | -7.6% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 16 | 75.0% | +86.9% | 0.83 | 8.34 | -8.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 14 | 71.4% | +107.3% | 0.84 | 11.36 | -9.0% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 19 | 68.4% | +162.6% | 0.91 | 8.81 | -7.8% |
| Alt E: gold_commodity rules (3 rules, 10%/5%) | 14 | 71.4% | +75.1% | 0.78 | 8.47 | -9.0% |
| Alt F: Gold commodity tight (6%/3%) | 25 | 64.0% | +61.6% | 0.77 | 2.49 | -9.7% |
| Alt G: Gold commodity (8%/4%) | 21 | 76.2% | +124.1% | 0.98 | 7.45 | -7.3% |

**Best baseline selected for validation: Alt G: Gold commodity (8%/4%)**

---

## 2. Full Validation

### Alt G: Gold commodity (8%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+124.1%, Trades=21, WR=76.2%, Sharpe=0.98, PF=7.45, DD=-7.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.79, Test Sharpe=1.18, Ratio=150% (need >=50%) |
| Bootstrap | **PASS** | p=0.0000, Sharpe CI=[3.09, 12.62], WR CI=[57.1%, 95.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.6%, Median equity=$2,628, Survival=100.0% |
| Regime | FAIL | bull:16t/+74.7%, bear:3t/+16.2%, volatile:1t/-0.8%, crisis:1t/+11.6% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: full mining rules (14) | 19 | 68.4% | +87.6% | 1.18 |
| Regime tune: full rules (10) | 22 | 72.7% | +87.4% | 1.11 |
| Regime tune: + volume_breakout | 21 | 76.2% | +127.5% | 1.04 |
| Regime tune: + commodity_breakout | 18 | 66.7% | +94.6% | 1.01 |
| Regime tune: + miner_metal_ratio | 21 | 76.2% | +124.1% | 0.98 |
| Regime tune: PT=15% | 14 | 71.4% | +108.2% | 0.97 |
| Regime tune: PT=12% | 16 | 75.0% | +135.6% | 0.88 |
| Regime tune: conf=0.65 | 19 | 68.4% | +88.5% | 0.71 |
| Regime tune: full mining rules (14) [multi-TF] | 47 | 51.1% | +105.1% | 0.77 |

### Full Validation of Top Candidates

### Regime tune: full mining rules (14)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+87.6%, Trades=19, WR=68.4%, Sharpe=1.18, PF=4.12, DD=-10.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.91, Test Sharpe=1.23, Ratio=135% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[1.78, 11.23], WR CI=[47.4%, 89.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.5%, Median equity=$2,219, Survival=100.0% |
| Regime | FAIL | bull:14t/+77.5%, bear:1t/+8.8%, chop:2t/+4.9%, volatile:2t/-6.3% |

**Result: 3/4 gates passed**

---

### Regime tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+87.4%, Trades=22, WR=72.7%, Sharpe=1.11, PF=4.05, DD=-10.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.78, Test Sharpe=1.23, Ratio=157% (need >=50%) |
| Bootstrap | **PASS** | p=0.0020, Sharpe CI=[1.70, 9.90], WR CI=[54.5%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.3%, Median equity=$2,227, Survival=100.0% |
| Regime | FAIL | bull:16t/+67.2%, bear:3t/+15.2%, chop:1t/+8.9%, volatile:2t/-6.3% |

**Result: 3/4 gates passed**

---

### Regime tune: + volume_breakout

- **Rules:** `trend_continuation, seasonality, death_cross, volume_breakout`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+127.5%, Trades=21, WR=76.2%, Sharpe=1.04, PF=7.78, DD=-7.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.90, Test Sharpe=1.18, Ratio=132% (need >=50%) |
| Bootstrap | **PASS** | p=0.0000, Sharpe CI=[3.27, 13.09], WR CI=[57.1%, 95.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.4%, Median equity=$2,727, Survival=100.0% |
| Regime | FAIL | bull:16t/+75.1%, bear:3t/+19.7%, volatile:1t/-0.8%, crisis:1t/+11.6% |

**Result: 3/4 gates passed**

---

### Regime tune: full mining rules (14) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+105.1%, Trades=47, WR=51.1%, Sharpe=0.77, PF=3.26, DD=-9.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.47, Test Sharpe=1.23, Ratio=264% (need >=50%) |
| Bootstrap | **PASS** | p=0.0004, Sharpe CI=[1.51, 5.34], WR CI=[44.7%, 74.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-14.0%, Median equity=$2,540, Survival=100.0% |
| Regime | FAIL | bull:24t/+76.8%, bear:10t/+9.2%, chop:2t/+8.2%, volatile:10t/-3.8%, crisis:1t/+8.0% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: full mining rules (14)** | **PASS** | **PASS** | **PASS** | FAIL | **1.18** | **+87.6%** | 19 |
| Regime tune: full rules (10) | **PASS** | **PASS** | **PASS** | FAIL | 1.11 | +87.4% | 22 |
| Regime tune: + volume_breakout | **PASS** | **PASS** | **PASS** | FAIL | 1.04 | +127.5% | 21 |
| Alt G: Gold commodity (8%/4%) | **PASS** | **PASS** | **PASS** | FAIL | 0.98 | +124.1% | 21 |
| Regime tune: full mining rules (14) [multi-TF] | **PASS** | **PASS** | **PASS** | FAIL | 0.77 | +105.1% | 47 |

---

## 5. Final Recommendation

**GLD partially validates.** Best config: Regime tune: full mining rules (14) (3/4 gates).

### Regime tune: full mining rules (14)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+87.6%, Trades=19, WR=68.4%, Sharpe=1.18, PF=4.12, DD=-10.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.91, Test Sharpe=1.23, Ratio=135% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[1.78, 11.23], WR CI=[47.4%, 89.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.5%, Median equity=$2,219, Survival=100.0% |
| Regime | FAIL | bull:14t/+77.5%, bear:1t/+8.8%, chop:2t/+4.9%, volatile:2t/-6.3% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

