# JPM (JPMorgan Chase) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 80.4 minutes
**Category:** Large-cap diversified bank

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

JPM — Largest US bank — investment banking, retail, commercial, asset mgmt. Large-cap diversified bank.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 14 | 50.0% | +33.5% | 0.34 | 1.82 | -14.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 20 | 60.0% | +65.6% | 0.70 | 2.07 | -25.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 14 | 50.0% | +38.2% | 0.41 | 2.05 | -10.7% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 15 | 53.3% | +26.8% | 0.29 | 1.94 | -16.8% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 22 | 63.6% | +80.4% | 0.80 | 2.23 | -24.3% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 13 | 61.5% | +35.5% | 0.61 | 2.47 | -12.0% |

**Best baseline selected for validation: Alt D: Financial-specific rules (12 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Financial-specific rules (12 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+80.4%, Trades=22, WR=63.6%, Sharpe=0.80, PF=2.23, DD=-24.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.99, Test Sharpe=0.42, Ratio=43% (need >=50%) |
| Bootstrap | FAIL | p=0.0313, Sharpe CI=[-0.11, 7.15], WR CI=[40.9%, 81.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.3%, Median equity=$2,001, Survival=100.0% |
| Regime | FAIL | bull:16t/+76.6%, bear:3t/-2.4%, chop:2t/-5.5%, volatile:1t/+10.3% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 25 | 60.0% | +109.9% | 0.83 |
| WF tune: PT=15% | 19 | 52.6% | +64.0% | 0.81 |
| WF tune: conf=0.45 | 22 | 63.6% | +80.4% | 0.80 |
| BS tune: conf=0.4 | 22 | 63.6% | +80.4% | 0.80 |
| BS tune: financial rules (12) | 22 | 63.6% | +80.4% | 0.80 |
| BS tune: + volume_breakout | 22 | 63.6% | +80.4% | 0.80 |
| WF tune: PT=8% | 27 | 63.0% | +74.8% | 0.71 |
| BS tune: full rules (10) | 20 | 60.0% | +65.6% | 0.70 |
| WF tune: PT=12% | 18 | 61.1% | +74.4% | 0.67 |
| WF tune: cooldown=7 | 22 | 54.5% | +39.1% | 0.65 |
| WF tune: conf=0.65 | 24 | 50.0% | +26.4% | 0.50 |
| WF tune: conf=0.55 | 26 | 57.7% | +18.8% | 0.26 |
| WF tune: conf=0.6 | 25 | 56.0% | +12.7% | 0.15 |
| Regime tune: tighter stop 4% [multi-TF] | 41 | 56.1% | +128.6% | 0.98 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+109.9%, Trades=25, WR=60.0%, Sharpe=0.83, PF=2.77, DD=-14.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.98, Test Sharpe=0.43, Ratio=44% (need >=50%) |
| Bootstrap | **PASS** | p=0.0080, Sharpe CI=[0.65, 6.91], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.7%, Median equity=$2,326, Survival=100.0% |
| Regime | FAIL | bull:19t/+81.3%, bear:3t/-2.4%, chop:2t/+4.1%, crisis:1t/+10.1% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+64.0%, Trades=19, WR=52.6%, Sharpe=0.81, PF=1.99, DD=-25.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.08, Test Sharpe=0.13, Ratio=12% (need >=50%) |
| Bootstrap | FAIL | p=0.0792, Sharpe CI=[-0.96, 6.11], WR CI=[31.6%, 73.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.6%, Median equity=$1,790, Survival=100.0% |
| Regime | FAIL | bull:13t/+57.2%, bear:3t/-2.4%, chop:2t/-0.0%, volatile:1t/+15.5% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+80.4%, Trades=22, WR=63.6%, Sharpe=0.80, PF=2.23, DD=-24.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.99, Test Sharpe=0.42, Ratio=43% (need >=50%) |
| Bootstrap | FAIL | p=0.0313, Sharpe CI=[-0.11, 7.15], WR CI=[40.9%, 81.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.3%, Median equity=$2,001, Survival=100.0% |
| Regime | FAIL | bull:16t/+76.6%, bear:3t/-2.4%, chop:2t/-5.5%, volatile:1t/+10.3% |

**Result: 1/4 gates passed**

---

### Regime tune: tighter stop 4% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+128.6%, Trades=41, WR=56.1%, Sharpe=0.98, PF=2.29, DD=-19.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.36, Test Sharpe=0.44, Ratio=33% (need >=50%) |
| Bootstrap | **PASS** | p=0.0033, Sharpe CI=[0.93, 5.28], WR CI=[41.5%, 70.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.0%, Median equity=$2,723, Survival=100.0% |
| Regime | **PASS** | bull:22t/+69.2%, bear:8t/+22.3%, chop:3t/+1.8%, volatile:8t/+16.0% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: tighter stop 4% [multi-TF]** | FAIL | **PASS** | **PASS** | **PASS** | **0.98** | **+128.6%** | 41 |
| Regime tune: tighter stop 4% | FAIL | **PASS** | **PASS** | FAIL | 0.83 | +109.9% | 25 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.81 | +64.0% | 19 |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.80 | +80.4% | 22 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.80 | +80.4% | 22 |

---

## 5. Final Recommendation

**JPM partially validates.** Best config: Regime tune: tighter stop 4% [multi-TF] (3/4 gates).

### Regime tune: tighter stop 4% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+128.6%, Trades=41, WR=56.1%, Sharpe=0.98, PF=2.29, DD=-19.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.36, Test Sharpe=0.44, Ratio=33% (need >=50%) |
| Bootstrap | **PASS** | p=0.0033, Sharpe CI=[0.93, 5.28], WR CI=[41.5%, 70.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.0%, Median equity=$2,723, Survival=100.0% |
| Regime | **PASS** | bull:22t/+69.2%, bear:8t/+22.3%, chop:3t/+1.8%, volatile:8t/+16.0% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

