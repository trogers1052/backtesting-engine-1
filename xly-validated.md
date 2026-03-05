# XLY (Consumer Discretionary Select Sector SPDR) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 23.1 minutes
**Category:** Consumer Discretionary ETF

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

XLY — Sector ETF — AMZN ~22%, TSLA ~14%, growth-weighted consumer discretionary. Consumer Discretionary ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 44.4% | +17.6% | 0.26 | 1.92 | -15.4% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 43.8% | +19.7% | 0.30 | 1.41 | -14.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 30.0% | +1.9% | -0.22 | 1.07 | -16.8% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 57.1% | +34.7% | 0.53 | 4.19 | -12.0% |
| Alt D: Recommended rules (3 rules, 10%/5%) | 9 | 44.4% | +17.6% | 0.26 | 1.92 | -15.4% |
| Alt E: Tech rules (ETF is growth-weighted) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt F: ETF tight (6%/3%, conf=0.55) | 15 | 40.0% | -0.2% | -0.22 | 0.99 | -14.8% |
| Alt G: ETF moderate (8%/4%) | 11 | 36.4% | -1.7% | -0.55 | 0.95 | -13.9% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.7%, Trades=16, WR=43.8%, Sharpe=0.30, PF=1.41, DD=-14.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.51, Test Sharpe=-0.11, Ratio=-22% (need >=50%) |
| Bootstrap | FAIL | p=0.2019, Sharpe CI=[-2.33, 5.38], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.2%, Median equity=$1,246, Survival=100.0% |
| Regime | FAIL | bull:14t/+33.6%, chop:1t/-4.4%, crisis:1t/-2.3% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: recommended rules | 8 | 50.0% | +29.7% | 0.49 |
| WF tune: PT=12% | 15 | 40.0% | +27.4% | 0.36 |
| WF tune: conf=0.55 | 16 | 50.0% | +18.4% | 0.33 |
| WF tune: ATR stops x2.5 | 17 | 41.2% | +21.0% | 0.30 |
| WF tune: conf=0.45 | 16 | 43.8% | +19.7% | 0.30 |
| BS tune: conf=0.4 | 16 | 43.8% | +19.7% | 0.30 |
| BS tune: full rules (10) | 16 | 43.8% | +19.7% | 0.30 |
| WF tune: PT=15% | 14 | 28.6% | +16.3% | 0.26 |
| WF tune: conf=0.6 | 15 | 46.7% | +14.4% | 0.25 |
| Regime tune: tighter stop 4% | 18 | 38.9% | +14.9% | 0.21 |
| WF tune: PT=8% | 21 | 42.9% | +11.2% | 0.15 |
| WF tune: cooldown=7 | 16 | 37.5% | +9.4% | 0.10 |
| WF tune: PT=7% | 22 | 40.9% | -3.1% | -0.24 |
| WF tune: conf=0.65 | 13 | 38.5% | -6.6% | -0.26 |
| WF tune: PT=6% | 24 | 41.7% | -11.1% | -0.65 |
| WF tune: recommended rules [multi-TF] | 11 | 36.4% | +0.4% | -0.16 |

### Full Validation of Top Candidates

### WF tune: recommended rules

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+29.7%, Trades=8, WR=50.0%, Sharpe=0.49, PF=2.76, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.51, Test Sharpe=0.30, Ratio=59% (need >=50%) |
| Bootstrap | FAIL | p=0.0749, Sharpe CI=[-1.41, 11.75], WR CI=[25.0%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-13.5%, Median equity=$1,337, Survival=100.0% |
| Regime | FAIL | bull:8t/+31.7% |

**Result: 2/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+27.4%, Trades=15, WR=40.0%, Sharpe=0.36, PF=1.58, DD=-18.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.80, Test Sharpe=0.06, Ratio=8% (need >=50%) |
| Bootstrap | FAIL | p=0.1714, Sharpe CI=[-1.87, 5.62], WR CI=[20.0%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.3%, Median equity=$1,324, Survival=100.0% |
| Regime | FAIL | bull:13t/+40.4%, chop:1t/-4.4%, crisis:1t/-2.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+18.4%, Trades=16, WR=50.0%, Sharpe=0.33, PF=1.58, DD=-12.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.33, Test Sharpe=-1.15, Ratio=-346% (need >=50%) |
| Bootstrap | FAIL | p=0.1617, Sharpe CI=[-2.16, 5.30], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.1%, Median equity=$1,236, Survival=100.0% |
| Regime | FAIL | bull:14t/+23.1%, chop:1t/+2.9%, volatile:1t/-2.0% |

**Result: 1/4 gates passed**

---

### WF tune: recommended rules [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.4%, Trades=11, WR=36.4%, Sharpe=-0.16, PF=1.01, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.57, Test Sharpe=-1.98, Ratio=-347% (need >=50%) |
| Bootstrap | FAIL | p=0.3966, Sharpe CI=[-6.11, 4.40], WR CI=[9.1%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.1%, Median equity=$1,019, Survival=100.0% |
| Regime | FAIL | bull:11t/+5.1% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: recommended rules** | **PASS** | FAIL | **PASS** | FAIL | **0.49** | **+29.7%** | 8 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | FAIL | 0.36 | +27.4% | 15 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.33 | +18.4% | 16 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.30 | +19.7% | 16 |
| WF tune: recommended rules [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.16 | +0.4% | 11 |

---

## 5. Final Recommendation

**XLY partially validates.** Best config: WF tune: recommended rules (2/4 gates).

### WF tune: recommended rules

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+29.7%, Trades=8, WR=50.0%, Sharpe=0.49, PF=2.76, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.51, Test Sharpe=0.30, Ratio=59% (need >=50%) |
| Bootstrap | FAIL | p=0.0749, Sharpe CI=[-1.41, 11.75], WR CI=[25.0%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-13.5%, Median equity=$1,337, Survival=100.0% |
| Regime | FAIL | bull:8t/+31.7% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

