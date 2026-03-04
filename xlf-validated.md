# XLF (Financial Select Sector SPDR ETF) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 81.2 minutes
**Category:** Financials sector ETF

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

XLF — Broad financials sector basket — BRK.B, JPM, V, MA top holdings. Financials sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 50.0% | +12.2% | 0.13 | 1.42 | -14.4% |
| Alt A: Full general rules (10 rules, 10%/5%) | 15 | 53.3% | +12.6% | 0.14 | 1.36 | -21.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 45.5% | -2.4% | -0.08 | 0.94 | -18.0% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 10 | 50.0% | -1.8% | -0.10 | 0.95 | -18.5% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 20 | 60.0% | +22.0% | 0.26 | 1.54 | -21.4% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 11 | 54.5% | +3.9% | 0.02 | 1.22 | -20.9% |

**Best baseline selected for validation: Alt D: Financial-specific rules (12 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Financial-specific rules (12 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+22.0%, Trades=20, WR=60.0%, Sharpe=0.26, PF=1.54, DD=-21.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.49, Test Sharpe=-0.05, Ratio=-10% (need >=50%) |
| Bootstrap | FAIL | p=0.1714, Sharpe CI=[-1.69, 5.45], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.4%, Median equity=$1,284, Survival=100.0% |
| Regime | **PASS** | bull:12t/+23.3%, bear:3t/-2.1%, chop:3t/-3.0%, volatile:2t/+11.4% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 20 | 70.0% | +31.2% | 0.55 |
| WF tune: conf=0.55 | 22 | 68.2% | +30.4% | 0.50 |
| WF tune: conf=0.45 | 18 | 66.7% | +26.2% | 0.32 |
| BS tune: conf=0.4 | 18 | 66.7% | +26.2% | 0.32 |
| WF tune: conf=0.65 | 15 | 66.7% | +19.4% | 0.31 |
| WF tune: PT=12% | 16 | 62.5% | +26.3% | 0.31 |
| WF tune: PT=15% | 15 | 60.0% | +24.3% | 0.29 |
| BS tune: financial rules (12) | 20 | 60.0% | +22.0% | 0.26 |
| BS tune: + volume_breakout | 20 | 60.0% | +22.0% | 0.26 |
| WF tune: PT=8% | 19 | 63.2% | +20.9% | 0.26 |
| WF tune: cooldown=7 | 19 | 57.9% | +22.5% | 0.24 |
| BS tune: full rules (10) | 15 | 53.3% | +12.6% | 0.14 |
| WF tune: conf=0.6 [multi-TF] | 75 | 50.7% | +13.2% | 0.21 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+31.2%, Trades=20, WR=70.0%, Sharpe=0.55, PF=2.58, DD=-10.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.17, Test Sharpe=-0.14, Ratio=-12% (need >=50%) |
| Bootstrap | FAIL | p=0.0297, Sharpe CI=[-0.12, 6.43], WR CI=[50.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.6%, Median equity=$1,391, Survival=100.0% |
| Regime | **PASS** | bull:12t/+20.6%, bear:4t/+5.1%, chop:3t/+5.3%, volatile:1t/+4.0% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.4%, Trades=22, WR=68.2%, Sharpe=0.50, PF=2.36, DD=-11.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.15, Test Sharpe=-0.27, Ratio=-24% (need >=50%) |
| Bootstrap | FAIL | p=0.0352, Sharpe CI=[-0.25, 5.88], WR CI=[50.0%, 86.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-12.0%, Median equity=$1,381, Survival=100.0% |
| Regime | **PASS** | bull:14t/+19.9%, bear:4t/+5.1%, chop:3t/+5.3%, volatile:1t/+4.0% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+26.2%, Trades=18, WR=66.7%, Sharpe=0.32, PF=1.66, DD=-21.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.59, Test Sharpe=-0.02, Ratio=-3% (need >=50%) |
| Bootstrap | FAIL | p=0.1428, Sharpe CI=[-1.50, 6.09], WR CI=[44.4%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.9%, Median equity=$1,321, Survival=100.0% |
| Regime | FAIL | bull:11t/+27.5%, bear:3t/-2.1%, chop:2t/-4.4%, volatile:2t/+11.4% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+13.2%, Trades=75, WR=50.7%, Sharpe=0.21, PF=1.29, DD=-11.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.82, Test Sharpe=-0.44, Ratio=-54% (need >=50%) |
| Bootstrap | FAIL | p=0.0803, Sharpe CI=[-0.53, 2.48], WR CI=[45.3%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.1%, Median equity=$1,322, Survival=100.0% |
| Regime | **PASS** | bull:48t/+17.7%, bear:10t/+6.8%, chop:8t/+7.6%, volatile:9t/-2.1% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | FAIL | **PASS** | **PASS** | **0.55** | **+31.2%** | 20 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | **PASS** | 0.50 | +30.4% | 22 |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.26 | +22.0% | 20 |
| WF tune: conf=0.6 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.21 | +13.2% | 75 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.32 | +26.2% | 18 |

---

## 5. Final Recommendation

**XLF partially validates.** Best config: WF tune: conf=0.6 (2/4 gates).

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+31.2%, Trades=20, WR=70.0%, Sharpe=0.55, PF=2.58, DD=-10.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.17, Test Sharpe=-0.14, Ratio=-12% (need >=50%) |
| Bootstrap | FAIL | p=0.0297, Sharpe CI=[-0.12, 6.43], WR CI=[50.0%, 90.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-11.6%, Median equity=$1,391, Survival=100.0% |
| Regime | **PASS** | bull:12t/+20.6%, bear:4t/+5.1%, chop:3t/+5.3%, volatile:1t/+4.0% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

