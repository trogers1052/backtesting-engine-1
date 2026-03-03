# EPD (Enterprise Products Partners) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 16.3 minutes
**Category:** Large-cap midstream

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

EPD — Midstream MLP — pipelines, storage, NGL processing. Large-cap midstream.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 8 | 50.0% | +7.0% | 0.05 | 1.30 | -15.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 50.0% | +11.4% | 0.16 | 1.47 | -15.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 36.4% | -1.0% | -0.15 | 0.95 | -15.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 42.9% | +3.9% | -0.01 | 1.17 | -15.4% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 10 | 50.0% | +11.4% | 0.16 | 1.47 | -15.2% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+11.4%, Trades=10, WR=50.0%, Sharpe=0.16, PF=1.47, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.34, Test Sharpe=0.54, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2213, Sharpe CI=[-3.11, 7.31], WR CI=[30.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.3%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:7t/+21.3%, volatile:3t/-5.8% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=15% | 8 | 50.0% | +16.1% | 0.24 |
| WF tune: PT=12% | 10 | 50.0% | +17.1% | 0.22 |
| WF tune: conf=0.45 | 10 | 50.0% | +11.4% | 0.16 |
| BS tune: conf=0.4 | 10 | 50.0% | +11.4% | 0.16 |
| BS tune: full rules (10) | 10 | 50.0% | +11.4% | 0.16 |
| BS tune: energy rules (12) | 10 | 50.0% | +11.4% | 0.16 |
| BS tune: + volume_breakout | 10 | 50.0% | +11.4% | 0.16 |
| BS tune: + commodity_breakout | 10 | 50.0% | +11.4% | 0.16 |
| Regime tune: + dollar_weakness | 10 | 50.0% | +11.4% | 0.16 |
| Regime tune: tighter stop 4% | 13 | 38.5% | +8.5% | 0.10 |
| WF tune: cooldown=7 | 10 | 50.0% | +5.3% | 0.01 |
| WF tune: PT=8% | 12 | 50.0% | +3.3% | -0.06 |
| WF tune: conf=0.55 | 23 | 30.4% | -17.8% | -0.49 |
| WF tune: conf=0.6 | 20 | 35.0% | -21.9% | -0.64 |
| WF tune: conf=0.65 | 14 | 14.3% | -29.3% | -1.23 |

### Full Validation of Top Candidates

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+16.1%, Trades=8, WR=50.0%, Sharpe=0.24, PF=2.28, DD=-15.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.22, Test Sharpe=0.57, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1972, Sharpe CI=[-5.05, 7.09], WR CI=[12.5%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.7%, Median equity=$1,189, Survival=100.0% |
| Regime | FAIL | bull:5t/+25.7%, volatile:3t/-5.8% |

**Result: 1/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+17.1%, Trades=10, WR=50.0%, Sharpe=0.22, PF=1.80, DD=-17.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.22, Test Sharpe=0.57, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2117, Sharpe CI=[-3.61, 6.60], WR CI=[20.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.3%, Median equity=$1,205, Survival=100.0% |
| Regime | FAIL | bull:7t/+27.9%, volatile:3t/-5.8% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+11.4%, Trades=10, WR=50.0%, Sharpe=0.16, PF=1.47, DD=-15.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.34, Test Sharpe=0.54, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2213, Sharpe CI=[-3.11, 7.31], WR CI=[30.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.3%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:7t/+21.3%, volatile:3t/-5.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=15%** | FAIL | FAIL | **PASS** | FAIL | **0.24** | **+16.1%** | 8 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | FAIL | 0.22 | +17.1% | 10 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.16 | +11.4% | 10 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.16 | +11.4% | 10 |

---

## 5. Final Recommendation

**EPD partially validates.** Best config: WF tune: PT=15% (1/4 gates).

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+16.1%, Trades=8, WR=50.0%, Sharpe=0.24, PF=2.28, DD=-15.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.22, Test Sharpe=0.57, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1972, Sharpe CI=[-5.05, 7.09], WR CI=[12.5%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.7%, Median equity=$1,189, Survival=100.0% |
| Regime | FAIL | bull:5t/+25.7%, volatile:3t/-5.8% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

