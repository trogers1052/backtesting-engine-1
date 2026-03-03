# XOM (ExxonMobil) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 26.9 minutes
**Category:** Large-cap integrated oil

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

XOM — Largest US integrated oil major — upstream, downstream, chemicals. Large-cap integrated oil.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 10.0% | -28.2% | -0.56 | 0.18 | -40.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 19 | 42.1% | +20.7% | 0.24 | 1.34 | -29.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 15.4% | -15.0% | -0.42 | 0.49 | -29.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 10 | 10.0% | -27.7% | -0.54 | 0.19 | -40.1% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 19 | 42.1% | +20.7% | 0.24 | 1.34 | -29.5% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.7%, Trades=19, WR=42.1%, Sharpe=0.24, PF=1.34, DD=-29.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.27, Test Sharpe=1.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2006, Sharpe CI=[-2.02, 4.88], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,276, Survival=100.0% |
| Regime | **PASS** | bull:15t/+1.3%, chop:1t/+10.1%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 19 | 42.1% | +20.7% | 0.24 |
| BS tune: conf=0.4 | 19 | 42.1% | +20.7% | 0.24 |
| BS tune: full rules (10) | 19 | 42.1% | +20.7% | 0.24 |
| BS tune: energy rules (12) | 19 | 42.1% | +20.7% | 0.24 |
| BS tune: + volume_breakout | 19 | 42.1% | +20.7% | 0.24 |
| BS tune: + commodity_breakout | 19 | 42.1% | +20.7% | 0.24 |
| WF tune: PT=8% | 23 | 43.5% | +9.0% | 0.10 |
| WF tune: PT=12% | 17 | 35.3% | +9.2% | 0.10 |
| WF tune: conf=0.65 | 15 | 40.0% | +3.3% | 0.02 |
| WF tune: cooldown=7 | 17 | 41.2% | +3.0% | 0.01 |
| WF tune: PT=15% | 15 | 26.7% | -2.9% | -0.07 |
| WF tune: conf=0.55 | 17 | 29.4% | -6.4% | -0.12 |
| WF tune: conf=0.6 | 17 | 29.4% | -8.4% | -0.14 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | 80 | 33.8% | +0.5% | 0.00 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.7%, Trades=19, WR=42.1%, Sharpe=0.24, PF=1.34, DD=-29.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.27, Test Sharpe=1.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2006, Sharpe CI=[-2.02, 4.88], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,276, Survival=100.0% |
| Regime | **PASS** | bull:15t/+1.3%, chop:1t/+10.1%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.7%, Trades=19, WR=42.1%, Sharpe=0.24, PF=1.34, DD=-29.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.27, Test Sharpe=1.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2006, Sharpe CI=[-2.02, 4.88], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,276, Survival=100.0% |
| Regime | **PASS** | bull:15t/+1.3%, chop:1t/+10.1%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.7%, Trades=19, WR=42.1%, Sharpe=0.24, PF=1.34, DD=-29.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.27, Test Sharpe=1.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2006, Sharpe CI=[-2.02, 4.88], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,276, Survival=100.0% |
| Regime | **PASS** | bull:15t/+1.3%, chop:1t/+10.1%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.5%, Trades=80, WR=33.8%, Sharpe=0.00, PF=1.00, DD=-41.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.51, Test Sharpe=1.14, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3264, Sharpe CI=[-1.39, 1.79], WR CI=[40.0%, 62.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.8%, Median equity=$1,148, Survival=100.0% |
| Regime | FAIL | bull:60t/+16.2%, bear:5t/-0.3%, chop:8t/+1.6%, volatile:7t/+3.1% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%)** | FAIL | FAIL | **PASS** | **PASS** | **0.24** | **+20.7%** | 19 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | **PASS** | 0.24 | +20.7% | 19 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | **PASS** | 0.24 | +20.7% | 19 |
| BS tune: full rules (10) | FAIL | FAIL | **PASS** | **PASS** | 0.24 | +20.7% | 19 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | FAIL | FAIL | **PASS** | FAIL | 0.00 | +0.5% | 80 |

---

## 5. Final Recommendation

**XOM partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) (2/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+20.7%, Trades=19, WR=42.1%, Sharpe=0.24, PF=1.34, DD=-29.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.27, Test Sharpe=1.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2006, Sharpe CI=[-2.02, 4.88], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.4%, Median equity=$1,276, Survival=100.0% |
| Regime | **PASS** | bull:15t/+1.3%, chop:1t/+10.1%, volatile:3t/+19.0% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

