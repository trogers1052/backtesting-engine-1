# VRTX (Vertex Pharmaceuticals) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 66.9 minutes
**Category:** Large-cap biotech (hybrid)

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

VRTX — Large-cap biotech — beta 0.23-0.49, $12B CF revenue, unusually low-beta for biotech. Large-cap biotech (hybrid).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 33.3% | -2.4% | -0.00 | 0.88 | -34.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 44.4% | +42.3% | 0.40 | 1.74 | -21.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 27.3% | +6.7% | 0.08 | 1.04 | -30.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 33.3% | +1.5% | 0.04 | 0.97 | -33.6% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 24 | 58.3% | +84.0% | 0.80 | 2.29 | -18.4% |
| Alt E: large_biotech lean (4 rules, 10%/5%) | 18 | 55.6% | +33.0% | 0.39 | 1.71 | -22.8% |
| Alt F: Biotech hybrid (10%/5%) | 18 | 50.0% | +18.1% | 0.28 | 1.46 | -22.7% |
| Alt G: Biotech moderate (8%/5%) | 21 | 57.1% | +19.3% | 0.29 | 1.32 | -25.9% |

**Best baseline selected for validation: Alt D: Healthcare rules (13 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Healthcare rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+84.0%, Trades=24, WR=58.3%, Sharpe=0.80, PF=2.29, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.78, Test Sharpe=1.23, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0149, Sharpe CI=[0.31, 7.09], WR CI=[41.7%, 79.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.7%, Median equity=$2,084, Survival=100.0% |
| Regime | FAIL | bull:17t/+80.5%, bear:2t/+2.3%, chop:3t/-7.7%, volatile:2t/+5.9% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: healthcare rules (13) | 24 | 58.3% | +84.0% | 0.80 |
| Regime tune: PT=8% | 26 | 57.7% | +62.6% | 0.66 |
| Regime tune: PT=7% | 32 | 59.4% | +46.1% | 0.53 |
| Regime tune: tighter stop 4% | 29 | 48.3% | +46.5% | 0.51 |
| Regime tune: PT=12% | 19 | 57.9% | +48.9% | 0.45 |
| Regime tune: full rules (10) | 18 | 44.4% | +42.3% | 0.40 |
| Regime tune: PT=15% | 18 | 50.0% | +44.0% | 0.39 |
| Regime tune: conf=0.65 | 19 | 42.1% | +15.8% | 0.30 |
| Alt D: Healthcare rules (13 rules, 10%/5%) [multi-TF] | 53 | 54.7% | +69.5% | 0.89 |

### Full Validation of Top Candidates

### Regime tune: healthcare rules (13)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+84.0%, Trades=24, WR=58.3%, Sharpe=0.80, PF=2.29, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.78, Test Sharpe=1.23, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0149, Sharpe CI=[0.31, 7.09], WR CI=[41.7%, 79.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.7%, Median equity=$2,084, Survival=100.0% |
| Regime | FAIL | bull:17t/+80.5%, bear:2t/+2.3%, chop:3t/-7.7%, volatile:2t/+5.9% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+62.6%, Trades=26, WR=57.7%, Sharpe=0.66, PF=1.67, DD=-25.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.89, Test Sharpe=1.23, Ratio=139% (need >=50%) |
| Bootstrap | FAIL | p=0.0543, Sharpe CI=[-0.50, 5.98], WR CI=[42.3%, 80.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.9%, Median equity=$1,801, Survival=100.0% |
| Regime | FAIL | bull:19t/+70.1%, bear:1t/+9.1%, chop:3t/-7.7%, volatile:3t/-3.8% |

**Result: 2/4 gates passed**

---

### Regime tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+46.1%, Trades=32, WR=59.4%, Sharpe=0.53, PF=1.55, DD=-22.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.69, Test Sharpe=1.20, Ratio=175% (need >=50%) |
| Bootstrap | FAIL | p=0.0557, Sharpe CI=[-0.48, 5.39], WR CI=[43.8%, 78.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.0%, Median equity=$1,751, Survival=100.0% |
| Regime | FAIL | bull:23t/+60.8%, bear:1t/+7.5%, chop:5t/-0.2%, volatile:3t/-3.8% |

**Result: 2/4 gates passed**

---

### Alt D: Healthcare rules (13 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+69.5%, Trades=53, WR=54.7%, Sharpe=0.89, PF=1.79, DD=-16.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.05, Test Sharpe=1.09, Ratio=103% (need >=50%) |
| Bootstrap | FAIL | p=0.0294, Sharpe CI=[-0.05, 3.76], WR CI=[45.3%, 71.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.1%, Median equity=$1,917, Survival=100.0% |
| Regime | FAIL | bull:42t/+69.8%, bear:1t/+10.0%, chop:5t/-2.8%, volatile:3t/+4.5%, crisis:2t/-9.6% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt D: Healthcare rules (13 rules, 10%/5%)** | **PASS** | **PASS** | **PASS** | FAIL | **0.80** | **+84.0%** | 24 |
| Regime tune: healthcare rules (13) | **PASS** | **PASS** | **PASS** | FAIL | 0.80 | +84.0% | 24 |
| Alt D: Healthcare rules (13 rules, 10%/5%) [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | 0.89 | +69.5% | 53 |
| Regime tune: PT=8% | **PASS** | FAIL | **PASS** | FAIL | 0.66 | +62.6% | 26 |
| Regime tune: PT=7% | **PASS** | FAIL | **PASS** | FAIL | 0.53 | +46.1% | 32 |

---

## 5. Final Recommendation

**VRTX partially validates.** Best config: Alt D: Healthcare rules (13 rules, 10%/5%) (3/4 gates).

### Alt D: Healthcare rules (13 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, healthcare_mean_reversion, healthcare_pullback, healthcare_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+84.0%, Trades=24, WR=58.3%, Sharpe=0.80, PF=2.29, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.78, Test Sharpe=1.23, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0149, Sharpe CI=[0.31, 7.09], WR CI=[41.7%, 79.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.7%, Median equity=$2,084, Survival=100.0% |
| Regime | FAIL | bull:17t/+80.5%, bear:2t/+2.3%, chop:3t/-7.7%, volatile:2t/+5.9% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

