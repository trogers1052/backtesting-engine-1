# D (Dominion Energy) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 74.7 minutes
**Category:** Regulated utility

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

D — Large regulated utility (electric + gas) — beta 0.70, frozen dividend risk. Regulated utility.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 4 | 50.0% | +13.5% | 0.48 | 3.97 | -13.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 7 | 57.1% | +19.4% | 0.52 | 2.88 | -11.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 5 | 20.0% | -5.3% | -0.33 | 0.69 | -12.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 4 | 50.0% | +15.1% | 0.51 | 4.27 | -13.1% |
| Alt D: Utility rules (13 rules, 10%/5%) | 21 | 47.6% | +8.1% | 0.08 | 1.24 | -16.8% |
| Alt E: regulated lean (3 rules, 10%/5%) | 17 | 41.2% | -12.2% | -1.05 | 0.76 | -19.3% |
| Alt F: Regulated tight (7%/4%, conf=0.55, cooldown=7) | 20 | 35.0% | -17.3% | -0.93 | 0.63 | -23.5% |
| Alt G: Regulated moderate (8%/4%) | 20 | 40.0% | -4.5% | -0.33 | 0.90 | -21.2% |

**Best baseline selected for validation: Alt D: Utility rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Utility rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, utility_mean_reversion, utility_rate_reversion, utility_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+8.1%, Trades=21, WR=47.6%, Sharpe=0.08, PF=1.24, DD=-16.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.07, Test Sharpe=0.28, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3119, Sharpe CI=[-2.62, 3.82], WR CI=[28.6%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.5%, Median equity=$1,129, Survival=100.0% |
| Regime | **PASS** | bull:16t/+5.4%, bear:1t/+2.3%, chop:4t/+9.4% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: full rules (10) | 7 | 57.1% | +19.4% | 0.52 |
| WF tune: cooldown=7 | 18 | 55.6% | +19.2% | 0.30 |
| WF tune: PT=12% | 20 | 50.0% | +19.7% | 0.28 |
| WF tune: ATR stops x2.5 | 21 | 47.6% | +8.7% | 0.10 |
| BS tune: utility rules (13) | 21 | 47.6% | +8.1% | 0.08 |
| WF tune: PT=15% | 14 | 42.9% | +6.2% | 0.03 |
| WF tune: conf=0.45 | 21 | 42.9% | -0.3% | -0.10 |
| BS tune: conf=0.4 | 21 | 42.9% | -0.3% | -0.10 |
| WF tune: PT=8% | 21 | 47.6% | -0.0% | -0.16 |
| WF tune: conf=0.6 | 22 | 36.4% | -8.3% | -0.19 |
| WF tune: conf=0.55 | 28 | 42.9% | -7.5% | -0.27 |
| BS tune: regulated rules | 20 | 40.0% | -9.6% | -0.37 |
| WF tune: conf=0.65 | 16 | 37.5% | -11.2% | -0.38 |
| WF tune: PT=7% | 21 | 47.6% | -6.6% | -0.54 |
| BS tune: full rules (10) [multi-TF] | 73 | 49.3% | -4.0% | -0.42 |

### Full Validation of Top Candidates

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.4%, Trades=7, WR=57.1%, Sharpe=0.52, PF=2.88, DD=-11.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.28, Test Sharpe=0.66, Ratio=238% (need >=50%) |
| Bootstrap | FAIL | p=0.1241, Sharpe CI=[-2.63, 10.31], WR CI=[28.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.0%, Median equity=$1,221, Survival=100.0% |
| Regime | FAIL | bull:7t/+22.0% |

**Result: 2/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, utility_mean_reversion, utility_rate_reversion, utility_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+19.2%, Trades=18, WR=55.6%, Sharpe=0.30, PF=1.47, DD=-16.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.28, Test Sharpe=0.92, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1941, Sharpe CI=[-2.09, 5.00], WR CI=[33.3%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.6%, Median equity=$1,250, Survival=100.0% |
| Regime | **PASS** | bull:13t/+14.6%, bear:1t/+2.3%, chop:4t/+9.8% |

**Result: 2/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, utility_mean_reversion, utility_rate_reversion, utility_seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.7%, Trades=20, WR=50.0%, Sharpe=0.28, PF=1.50, DD=-13.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.11, Test Sharpe=0.53, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2005, Sharpe CI=[-2.19, 4.37], WR CI=[30.0%, 70.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.0%, Median equity=$1,258, Survival=100.0% |
| Regime | **PASS** | bull:15t/+15.1%, bear:1t/+2.3%, chop:4t/+11.0% |

**Result: 2/4 gates passed**

---

### BS tune: full rules (10) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-4.0%, Trades=73, WR=49.3%, Sharpe=-0.42, PF=0.94, DD=-20.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.20, Test Sharpe=0.89, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3439, Sharpe CI=[-1.60, 1.75], WR CI=[41.1%, 63.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.1%, Median equity=$1,109, Survival=100.0% |
| Regime | FAIL | bull:48t/+13.9%, bear:7t/+0.3%, chop:12t/+4.1%, volatile:5t/-0.4%, crisis:1t/-5.1% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: full rules (10)** | **PASS** | FAIL | **PASS** | FAIL | **0.52** | **+19.4%** | 7 |
| WF tune: cooldown=7 | FAIL | FAIL | **PASS** | **PASS** | 0.30 | +19.2% | 18 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | **PASS** | 0.28 | +19.7% | 20 |
| Alt D: Utility rules (13 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.08 | +8.1% | 21 |
| BS tune: full rules (10) [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.42 | -4.0% | 73 |

---

## 5. Final Recommendation

**D partially validates.** Best config: BS tune: full rules (10) (2/4 gates).

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.4%, Trades=7, WR=57.1%, Sharpe=0.52, PF=2.88, DD=-11.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.28, Test Sharpe=0.66, Ratio=238% (need >=50%) |
| Bootstrap | FAIL | p=0.1241, Sharpe CI=[-2.63, 10.31], WR CI=[28.6%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.0%, Median equity=$1,221, Survival=100.0% |
| Regime | FAIL | bull:7t/+22.0% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

