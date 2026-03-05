# VPU (Vanguard Utilities ETF) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 73.8 minutes
**Category:** Utility sector ETF

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

VPU — Broad utilities ETF — broader than XLU, includes small/mid-cap utilities. Utility sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 42.9% | +20.2% | 0.44 | 1.70 | -14.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 60.0% | +33.7% | 0.86 | 2.22 | -13.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 37.5% | +14.0% | 0.30 | 1.24 | -16.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 3 | 66.7% | +35.5% | 0.81 | 999.99 | -10.9% |
| Alt D: Utility rules (13 rules, 10%/5%) | 18 | 50.0% | +22.1% | 0.40 | 1.38 | -15.3% |
| Alt E: utility_etf lean (4 rules, 10%/5%) | 15 | 46.7% | +7.9% | 0.07 | 1.21 | -16.9% |
| Alt F: ETF tight (6%/3%, conf=0.55) | 20 | 30.0% | -16.0% | -0.92 | 0.61 | -22.1% |
| Alt G: ETF moderate (8%/4%) | 18 | 38.9% | -5.5% | -0.36 | 0.87 | -19.5% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+33.7%, Trades=10, WR=60.0%, Sharpe=0.86, PF=2.22, DD=-13.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.41, Test Sharpe=0.74, Ratio=183% (need >=50%) |
| Bootstrap | FAIL | p=0.0547, Sharpe CI=[-0.91, 10.93], WR CI=[40.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-16.3%, Median equity=$1,411, Survival=100.0% |
| Regime | FAIL | bull:9t/+35.5%, volatile:1t/+2.1% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=15% | 5 | 80.0% | +39.4% | 0.90 |
| Regime tune: tighter stop 4% | 11 | 54.5% | +32.7% | 0.87 |
| Regime tune: PT=12% | 8 | 75.0% | +36.5% | 0.86 |
| BS tune: conf=0.4 | 10 | 60.0% | +33.7% | 0.86 |
| BS tune: conf=0.45 | 10 | 60.0% | +33.7% | 0.86 |
| BS tune: full rules (10) | 10 | 60.0% | +33.7% | 0.86 |
| Regime tune: + utility_mean_reversion | 10 | 60.0% | +33.7% | 0.86 |
| Regime tune: + utility_rate_reversion | 10 | 60.0% | +33.7% | 0.86 |
| Regime tune: PT=8% | 12 | 66.7% | +27.5% | 0.82 |
| BS tune: cooldown=7 | 8 | 75.0% | +28.6% | 0.79 |
| BS tune: utility rules (13) | 18 | 50.0% | +22.1% | 0.40 |
| BS tune: conf=0.55 | 16 | 43.8% | +11.5% | 0.22 |
| BS tune: utility_etf rules | 15 | 46.7% | +10.6% | 0.14 |
| Regime tune: conf=0.65 | 12 | 41.7% | +1.6% | -0.27 |
| Regime tune: PT=15% [multi-TF] | 62 | 45.2% | +28.7% | 0.42 |

### Full Validation of Top Candidates

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+39.4%, Trades=5, WR=80.0%, Sharpe=0.90, PF=999.99, DD=-10.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.47, Test Sharpe=0.97, Ratio=205% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[4.48, 32.27], WR CI=[100.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-0.0%, Median equity=$1,487, Survival=100.0% |
| Regime | FAIL | bull:4t/+40.1%, volatile:1t/+2.1% |

**Result: 3/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+32.7%, Trades=11, WR=54.5%, Sharpe=0.87, PF=2.23, DD=-10.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.45, Test Sharpe=0.81, Ratio=179% (need >=50%) |
| Bootstrap | FAIL | p=0.0469, Sharpe CI=[-0.66, 9.51], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.7%, Median equity=$1,409, Survival=100.0% |
| Regime | **PASS** | bull:9t/+24.7%, volatile:2t/+12.3% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.5%, Trades=8, WR=75.0%, Sharpe=0.86, PF=3.92, DD=-15.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.49, Test Sharpe=0.96, Ratio=195% (need >=50%) |
| Bootstrap | FAIL | p=0.0381, Sharpe CI=[-0.46, 13.33], WR CI=[37.5%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.3%, Median equity=$1,456, Survival=100.0% |
| Regime | FAIL | bull:7t/+38.8%, volatile:1t/+2.1% |

**Result: 2/4 gates passed**

---

### Regime tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+28.7%, Trades=62, WR=45.2%, Sharpe=0.42, PF=1.69, DD=-14.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.10, Test Sharpe=0.87, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0336, Sharpe CI=[-0.19, 2.52], WR CI=[38.7%, 64.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-8.6%, Median equity=$1,472, Survival=100.0% |
| Regime | FAIL | bull:48t/+40.5%, bear:8t/+0.6%, chop:4t/+1.0%, volatile:2t/-0.7% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=15%** | **PASS** | **PASS** | **PASS** | FAIL | **0.90** | **+39.4%** | 5 |
| Regime tune: tighter stop 4% | **PASS** | FAIL | **PASS** | **PASS** | 0.87 | +32.7% | 11 |
| Regime tune: PT=12% | **PASS** | FAIL | **PASS** | FAIL | 0.86 | +36.5% | 8 |
| Alt A: Full general rules (10 rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.86 | +33.7% | 10 |
| Regime tune: PT=15% [multi-TF] | FAIL | FAIL | **PASS** | FAIL | 0.42 | +28.7% | 62 |

---

## 5. Final Recommendation

**VPU partially validates.** Best config: Regime tune: PT=15% (3/4 gates).

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+39.4%, Trades=5, WR=80.0%, Sharpe=0.90, PF=999.99, DD=-10.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.47, Test Sharpe=0.97, Ratio=205% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[4.48, 32.27], WR CI=[100.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-0.0%, Median equity=$1,487, Survival=100.0% |
| Regime | FAIL | bull:4t/+40.1%, volatile:1t/+2.1% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

