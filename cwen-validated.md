# CWEN (Clearway Energy) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 74.8 minutes
**Category:** Clean energy yieldco

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

CWEN — Clean energy yieldco (12+ GW wind/solar/storage) — bond-like cash flows, 5%+ yield. Clean energy yieldco.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 27.3% | -11.8% | -0.34 | 0.64 | -29.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 44.4% | +21.6% | 0.57 | 1.42 | -27.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 23.1% | -13.6% | -0.37 | 0.60 | -33.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 11 | 27.3% | -12.0% | -0.37 | 0.70 | -28.8% |
| Alt D: Utility rules (13 rules, 10%/5%) | 26 | 46.2% | +18.1% | 0.32 | 1.27 | -25.9% |
| Alt E: yieldco lean (3 rules, 10%/5%) | 18 | 33.3% | -18.8% | -0.40 | 0.67 | -32.7% |
| Alt F: Yieldco reversion (8%/5%, conf=0.50) | 18 | 33.3% | -22.2% | -0.49 | 0.61 | -33.4% |
| Alt G: Yieldco + midstream rule (10%/5%) | 18 | 33.3% | -18.8% | -0.40 | 0.67 | -32.7% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.6%, Trades=18, WR=44.4%, Sharpe=0.57, PF=1.42, DD=-27.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.04, Test Sharpe=1.04, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1881, Sharpe CI=[-2.06, 4.97], WR CI=[22.2%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,293, Survival=100.0% |
| Regime | **PASS** | bull:13t/+18.0%, chop:2t/+3.1%, volatile:3t/+10.6% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 18 | 44.4% | +21.6% | 0.57 |
| WF tune: ATR stops x2.5 | 18 | 44.4% | +21.6% | 0.57 |
| WF tune: + utility_mean_reversion | 18 | 44.4% | +21.6% | 0.57 |
| WF tune: + utility_rate_reversion | 18 | 44.4% | +21.6% | 0.57 |
| BS tune: conf=0.4 | 18 | 44.4% | +21.6% | 0.57 |
| BS tune: full rules (10) | 18 | 44.4% | +21.6% | 0.57 |
| WF tune: PT=7% | 22 | 54.5% | +27.9% | 0.54 |
| BS tune: utility rules (13) | 26 | 46.2% | +18.1% | 0.32 |
| WF tune: PT=8% | 21 | 42.9% | +3.8% | -0.03 |
| BS tune: yieldco rules | 19 | 42.1% | -8.8% | -0.23 |
| WF tune: cooldown=7 | 18 | 33.3% | -8.4% | -0.31 |
| WF tune: PT=15% | 16 | 31.2% | -7.0% | -0.35 |
| WF tune: PT=12% | 18 | 33.3% | -9.7% | -0.45 |
| WF tune: conf=0.6 | 15 | 26.7% | -24.8% | -0.67 |
| WF tune: conf=0.55 | 16 | 25.0% | -28.2% | -0.69 |
| WF tune: conf=0.65 | 13 | 15.4% | -31.5% | -0.69 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | 72 | 41.7% | -19.4% | -0.36 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.6%, Trades=18, WR=44.4%, Sharpe=0.57, PF=1.42, DD=-27.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.04, Test Sharpe=1.04, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1881, Sharpe CI=[-2.06, 4.97], WR CI=[22.2%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,293, Survival=100.0% |
| Regime | **PASS** | bull:13t/+18.0%, chop:2t/+3.1%, volatile:3t/+10.6% |

**Result: 2/4 gates passed**

---

### WF tune: ATR stops x2.5

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+21.6%, Trades=18, WR=44.4%, Sharpe=0.57, PF=1.42, DD=-27.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.04, Test Sharpe=1.04, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1881, Sharpe CI=[-2.06, 4.97], WR CI=[22.2%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,293, Survival=100.0% |
| Regime | **PASS** | bull:13t/+18.0%, chop:2t/+3.1%, volatile:3t/+10.6% |

**Result: 2/4 gates passed**

---

### WF tune: + utility_mean_reversion

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, utility_mean_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.6%, Trades=18, WR=44.4%, Sharpe=0.57, PF=1.42, DD=-27.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.04, Test Sharpe=1.04, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1881, Sharpe CI=[-2.06, 4.97], WR CI=[22.2%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,293, Survival=100.0% |
| Regime | **PASS** | bull:13t/+18.0%, chop:2t/+3.1%, volatile:3t/+10.6% |

**Result: 2/4 gates passed**

---

### Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-19.4%, Trades=72, WR=41.7%, Sharpe=-0.36, PF=0.75, DD=-40.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.32, Test Sharpe=0.77, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5864, Sharpe CI=[-2.11, 1.42], WR CI=[33.3%, 55.6%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.1%, Median equity=$916, Survival=100.0% |
| Regime | FAIL | bull:57t/-5.7%, bear:6t/+0.4%, chop:5t/-9.6%, volatile:4t/+11.4% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%)** | FAIL | FAIL | **PASS** | **PASS** | **0.57** | **+21.6%** | 18 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | **PASS** | 0.57 | +21.6% | 18 |
| WF tune: ATR stops x2.5 | FAIL | FAIL | **PASS** | **PASS** | 0.57 | +21.6% | 18 |
| WF tune: + utility_mean_reversion | FAIL | FAIL | **PASS** | **PASS** | 0.57 | +21.6% | 18 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | FAIL | FAIL | FAIL | FAIL | -0.36 | -19.4% | 72 |

---

## 5. Final Recommendation

**CWEN partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) (2/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.6%, Trades=18, WR=44.4%, Sharpe=0.57, PF=1.42, DD=-27.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.04, Test Sharpe=1.04, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1881, Sharpe CI=[-2.06, 4.97], WR CI=[22.2%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,293, Survival=100.0% |
| Regime | **PASS** | bull:13t/+18.0%, chop:2t/+3.1%, volatile:3t/+10.6% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

