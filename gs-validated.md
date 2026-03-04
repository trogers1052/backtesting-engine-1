# GS (Goldman Sachs) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 78.4 minutes
**Category:** Investment bank

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

GS — Premier investment bank — M&A, trading, wealth management, consumer. Investment bank.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 17 | 41.2% | +11.3% | 0.15 | 1.43 | -9.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 55.6% | +45.6% | 0.44 | 2.15 | -20.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 17 | 41.2% | +15.6% | 0.24 | 1.61 | -9.6% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 17 | 41.2% | +1.3% | -0.16 | 1.19 | -10.8% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 21 | 57.1% | +47.1% | 0.45 | 2.16 | -19.8% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 16 | 56.2% | +44.2% | 0.44 | 2.92 | -9.8% |

**Best baseline selected for validation: Alt D: Financial-specific rules (12 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Financial-specific rules (12 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+47.1%, Trades=21, WR=57.1%, Sharpe=0.45, PF=2.16, DD=-19.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.26, Test Sharpe=0.55, Ratio=207% (need >=50%) |
| Bootstrap | FAIL | p=0.0397, Sharpe CI=[-0.36, 5.94], WR CI=[38.1%, 76.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.6%, Median equity=$1,787, Survival=100.0% |
| Regime | FAIL | bull:15t/+74.1%, bear:3t/-15.7%, chop:2t/-1.0%, volatile:1t/+8.4% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=15% | 18 | 55.6% | +78.4% | 0.55 |
| BS tune: cooldown=7 | 17 | 58.8% | +46.6% | 0.49 |
| Regime tune: conf=0.65 | 22 | 63.6% | +40.4% | 0.47 |
| BS tune: conf=0.4 | 21 | 57.1% | +47.1% | 0.45 |
| BS tune: conf=0.45 | 21 | 57.1% | +47.1% | 0.45 |
| BS tune: financial rules (12) | 21 | 57.1% | +47.1% | 0.45 |
| BS tune: + volume_breakout | 21 | 57.1% | +47.1% | 0.45 |
| BS tune: full rules (10) | 18 | 55.6% | +45.6% | 0.44 |
| Regime tune: PT=12% | 22 | 50.0% | +37.6% | 0.41 |
| BS tune: conf=0.55 | 23 | 65.2% | +39.9% | 0.41 |
| Regime tune: tighter stop 4% | 23 | 52.2% | +31.5% | 0.38 |
| Regime tune: PT=15% [multi-TF] | 55 | 56.4% | +47.1% | 0.55 |

### Full Validation of Top Candidates

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+78.4%, Trades=18, WR=55.6%, Sharpe=0.55, PF=3.22, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.28, Test Sharpe=0.53, Ratio=191% (need >=50%) |
| Bootstrap | FAIL | p=0.0349, Sharpe CI=[-0.26, 6.63], WR CI=[33.3%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.3%, Median equity=$1,904, Survival=100.0% |
| Regime | FAIL | bull:12t/+72.5%, bear:3t/-15.7%, chop:2t/+8.1%, volatile:1t/+8.4% |

**Result: 2/4 gates passed**

---

### BS tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+46.6%, Trades=17, WR=58.8%, Sharpe=0.49, PF=2.59, DD=-13.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.26, Test Sharpe=0.50, Ratio=191% (need >=50%) |
| Bootstrap | FAIL | p=0.0471, Sharpe CI=[-0.55, 6.86], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.4%, Median equity=$1,703, Survival=100.0% |
| Regime | FAIL | bull:12t/+63.4%, bear:2t/-10.5%, chop:2t/-1.0%, volatile:1t/+8.4% |

**Result: 2/4 gates passed**

---

### Regime tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.4%, Trades=22, WR=63.6%, Sharpe=0.47, PF=2.10, DD=-13.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.28, Test Sharpe=0.65, Ratio=233% (need >=50%) |
| Bootstrap | FAIL | p=0.0734, Sharpe CI=[-0.92, 5.38], WR CI=[40.9%, 81.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.9%, Median equity=$1,555, Survival=100.0% |
| Regime | FAIL | bull:18t/+52.5%, bear:3t/-10.6%, chop:1t/+8.5% |

**Result: 2/4 gates passed**

---

### Regime tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+47.1%, Trades=55, WR=56.4%, Sharpe=0.55, PF=1.91, DD=-18.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.60, Ratio=130% (need >=50%) |
| Bootstrap | FAIL | p=0.0339, Sharpe CI=[-0.14, 3.28], WR CI=[45.5%, 70.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.9%, Median equity=$2,021, Survival=100.0% |
| Regime | FAIL | bull:37t/+79.0%, bear:3t/-11.8%, chop:7t/+12.4%, volatile:7t/+5.5%, crisis:1t/-5.3% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=15% [multi-TF]** | **PASS** | FAIL | **PASS** | FAIL | **0.55** | **+47.1%** | 55 |
| Regime tune: PT=15% | **PASS** | FAIL | **PASS** | FAIL | 0.55 | +78.4% | 18 |
| BS tune: cooldown=7 | **PASS** | FAIL | **PASS** | FAIL | 0.49 | +46.6% | 17 |
| Regime tune: conf=0.65 | **PASS** | FAIL | **PASS** | FAIL | 0.47 | +40.4% | 22 |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.45 | +47.1% | 21 |

---

## 5. Final Recommendation

**GS partially validates.** Best config: Regime tune: PT=15% [multi-TF] (2/4 gates).

### Regime tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+47.1%, Trades=55, WR=56.4%, Sharpe=0.55, PF=1.91, DD=-18.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.60, Ratio=130% (need >=50%) |
| Bootstrap | FAIL | p=0.0339, Sharpe CI=[-0.14, 3.28], WR CI=[45.5%, 70.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.9%, Median equity=$2,021, Survival=100.0% |
| Regime | FAIL | bull:37t/+79.0%, bear:3t/-11.8%, chop:7t/+12.4%, volatile:7t/+5.5%, crisis:1t/-5.3% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

