# FCX (Freeport-McMoRan) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 58.7 minutes
**Category:** Large-cap copper miner

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

FCX — Largest publicly traded copper miner — Grasberg mine, gold/moly byproducts. Large-cap copper miner.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 24 | 45.8% | -12.3% | -0.14 | 0.87 | -51.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 38 | 44.7% | +11.5% | 0.15 | 1.08 | -56.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 26 | 42.3% | -5.8% | -0.11 | 0.93 | -40.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 23 | 39.1% | -13.0% | -0.09 | 0.79 | -57.0% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 45 | 37.8% | -5.2% | -0.00 | 0.96 | -50.8% |
| Alt E: copper rules (4 rules, 10%/5%) | 31 | 38.7% | -16.6% | -0.36 | 0.83 | -48.1% |
| Alt F: Copper momentum (12%/6%) | 27 | 37.0% | -16.3% | -0.23 | 0.78 | -49.6% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+11.5%, Trades=38, WR=44.7%, Sharpe=0.15, PF=1.08, DD=-56.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.44, Test Sharpe=0.52, Ratio=118% (need >=50%) |
| Bootstrap | FAIL | p=0.2854, Sharpe CI=[-1.73, 2.94], WR CI=[28.9%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.9%, P95 DD=-51.9%, Median equity=$1,190, Survival=99.1% |
| Regime | **PASS** | bull:31t/+12.7%, bear:2t/+20.6%, chop:4t/-11.1%, volatile:1t/+10.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 38 | 44.7% | +17.3% | 0.19 |
| BS tune: conf=0.45 | 38 | 44.7% | +17.3% | 0.19 |
| BS tune: conf=0.55 | 33 | 48.5% | +14.7% | 0.16 |
| BS tune: full rules (10) | 38 | 44.7% | +11.5% | 0.15 |
| BS tune: + volume_breakout | 40 | 42.5% | +9.8% | 0.14 |
| MC tune: ATR stops x2.0 | 39 | 43.6% | +6.9% | 0.12 |
| MC tune: max_loss=4.0% | 42 | 38.1% | +1.2% | 0.06 |
| MC tune: max_loss=3.0% | 51 | 35.3% | -4.0% | 0.00 |
| BS tune: full mining rules (14) | 45 | 37.8% | -5.2% | -0.00 |
| BS tune: cooldown=7 | 34 | 44.1% | -4.3% | -0.03 |
| BS tune: copper rules | 36 | 33.3% | -26.6% | -0.36 |
| BS tune: conf=0.4 [multi-TF] | 94 | 44.7% | -15.1% | -0.03 |

### Full Validation of Top Candidates

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+17.3%, Trades=38, WR=44.7%, Sharpe=0.19, PF=1.11, DD=-54.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.44, Test Sharpe=0.65, Ratio=147% (need >=50%) |
| Bootstrap | FAIL | p=0.2529, Sharpe CI=[-1.63, 3.08], WR CI=[28.9%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.8%, P95 DD=-51.2%, Median equity=$1,256, Survival=99.2% |
| Regime | **PASS** | bull:31t/+12.7%, bear:2t/+20.6%, chop:4t/-5.4%, volatile:1t/+10.5% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+17.3%, Trades=38, WR=44.7%, Sharpe=0.19, PF=1.11, DD=-54.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.44, Test Sharpe=0.65, Ratio=147% (need >=50%) |
| Bootstrap | FAIL | p=0.2529, Sharpe CI=[-1.63, 3.08], WR CI=[28.9%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.8%, P95 DD=-51.2%, Median equity=$1,256, Survival=99.2% |
| Regime | **PASS** | bull:31t/+12.7%, bear:2t/+20.6%, chop:4t/-5.4%, volatile:1t/+10.5% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+14.7%, Trades=33, WR=48.5%, Sharpe=0.16, PF=1.05, DD=-52.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.94, Test Sharpe=0.13, Ratio=14% (need >=50%) |
| Bootstrap | FAIL | p=0.2712, Sharpe CI=[-1.85, 3.40], WR CI=[33.3%, 69.7%] |
| Monte Carlo | FAIL | Ruin=0.2%, P95 DD=-46.1%, Median equity=$1,228, Survival=99.8% |
| Regime | **PASS** | bull:26t/+12.7%, bear:2t/+20.6%, chop:4t/-11.1%, volatile:1t/+10.5% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-15.1%, Trades=94, WR=44.7%, Sharpe=-0.03, PF=0.89, DD=-55.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.20, Test Sharpe=0.81, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4339, Sharpe CI=[-1.44, 1.52], WR CI=[36.2%, 56.4%] |
| Monte Carlo | FAIL | Ruin=0.9%, P95 DD=-52.0%, Median equity=$1,009, Survival=99.1% |
| Regime | **PASS** | bull:63t/-21.9%, bear:8t/+4.8%, chop:14t/+19.8%, volatile:9t/+10.0% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | **PASS** | FAIL | FAIL | **PASS** | **0.19** | **+17.3%** | 38 |
| BS tune: conf=0.45 | **PASS** | FAIL | FAIL | **PASS** | 0.19 | +17.3% | 38 |
| Alt A: Full general rules (10 rules, 10%/5%) | **PASS** | FAIL | FAIL | **PASS** | 0.15 | +11.5% | 38 |
| BS tune: conf=0.55 | FAIL | FAIL | FAIL | **PASS** | 0.16 | +14.7% | 33 |
| BS tune: conf=0.4 [multi-TF] | FAIL | FAIL | FAIL | **PASS** | -0.03 | -15.1% | 94 |

---

## 5. Final Recommendation

**FCX partially validates.** Best config: BS tune: conf=0.4 (2/4 gates).

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+17.3%, Trades=38, WR=44.7%, Sharpe=0.19, PF=1.11, DD=-54.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.44, Test Sharpe=0.65, Ratio=147% (need >=50%) |
| Bootstrap | FAIL | p=0.2529, Sharpe CI=[-1.63, 3.08], WR CI=[28.9%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.8%, P95 DD=-51.2%, Median equity=$1,256, Survival=99.2% |
| Regime | **PASS** | bull:31t/+12.7%, bear:2t/+20.6%, chop:4t/-5.4%, volatile:1t/+10.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

