# SPGI (S&P Global) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 63.0 minutes
**Category:** Financial data/ratings

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

SPGI — Credit ratings, market intelligence, indices — owns S&P 500 brand. Financial data/ratings.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 8 | 37.5% | -0.6% | -0.22 | 0.98 | -16.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 38.9% | -4.3% | -0.13 | 0.92 | -26.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 9 | 33.3% | -2.7% | -0.37 | 0.91 | -16.7% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 11 | 45.5% | +0.7% | -0.13 | 1.02 | -20.1% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 24 | 50.0% | -0.0% | -0.04 | 1.00 | -24.8% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 14 | 50.0% | -11.8% | -1.80 | 0.62 | -16.0% |

**Best baseline selected for validation: Alt D: Financial-specific rules (12 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Financial-specific rules (12 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-0.0%, Trades=24, WR=50.0%, Sharpe=-0.04, PF=1.00, DD=-24.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.48, Test Sharpe=-1.51, Ratio=-312% (need >=50%) |
| Bootstrap | FAIL | p=0.4040, Sharpe CI=[-2.71, 3.37], WR CI=[29.2%, 70.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.0%, Median equity=$1,033, Survival=100.0% |
| Regime | FAIL | bull:17t/+26.4%, bear:2t/+5.5%, chop:3t/-22.2%, volatile:2t/-0.1% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 21 | 42.9% | +14.2% | 0.14 |
| Regime tune: tighter stop 4% | 24 | 50.0% | +9.9% | 0.10 |
| WF tune: conf=0.55 | 29 | 41.4% | +4.3% | 0.06 |
| WF tune: conf=0.6 | 27 | 44.4% | +2.4% | 0.03 |
| WF tune: conf=0.45 | 24 | 50.0% | -0.0% | -0.04 |
| BS tune: conf=0.4 | 24 | 50.0% | -0.0% | -0.04 |
| BS tune: financial rules (12) | 24 | 50.0% | -0.0% | -0.04 |
| BS tune: + volume_breakout | 24 | 50.0% | -0.0% | -0.04 |
| WF tune: PT=8% | 28 | 50.0% | -1.0% | -0.09 |
| BS tune: full rules (10) | 18 | 38.9% | -4.3% | -0.13 |
| WF tune: PT=15% | 21 | 42.9% | -3.2% | -0.14 |
| WF tune: cooldown=7 | 21 | 47.6% | -3.4% | -0.24 |
| WF tune: PT=12% | 21 | 42.9% | -11.1% | -0.41 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+14.2%, Trades=21, WR=42.9%, Sharpe=0.14, PF=1.28, DD=-23.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.50, Test Sharpe=-1.89, Ratio=-381% (need >=50%) |
| Bootstrap | FAIL | p=0.1950, Sharpe CI=[-1.85, 4.60], WR CI=[23.8%, 61.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.9%, Median equity=$1,269, Survival=100.0% |
| Regime | FAIL | bull:17t/+33.8%, bear:2t/+4.8%, chop:1t/-6.0%, volatile:1t/-3.3% |

**Result: 1/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+9.9%, Trades=24, WR=50.0%, Sharpe=0.10, PF=1.17, DD=-21.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.55, Test Sharpe=-1.16, Ratio=-209% (need >=50%) |
| Bootstrap | FAIL | p=0.2428, Sharpe CI=[-1.96, 4.09], WR CI=[29.2%, 70.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.1%, Median equity=$1,201, Survival=100.0% |
| Regime | FAIL | bull:18t/+43.7%, bear:2t/+5.5%, chop:3t/-21.8%, volatile:1t/-3.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.3%, Trades=29, WR=41.4%, Sharpe=0.06, PF=1.08, DD=-28.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.49, Test Sharpe=-1.45, Ratio=-293% (need >=50%) |
| Bootstrap | FAIL | p=0.2981, Sharpe CI=[-2.06, 3.32], WR CI=[27.6%, 62.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.9%, Median equity=$1,146, Survival=100.0% |
| Regime | FAIL | bull:23t/+35.5%, bear:2t/+5.5%, chop:2t/-10.9%, volatile:1t/-3.3%, crisis:1t/-8.3% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | FAIL | FAIL | **PASS** | FAIL | **0.14** | **+14.2%** | 21 |
| Regime tune: tighter stop 4% | FAIL | FAIL | **PASS** | FAIL | 0.10 | +9.9% | 24 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.06 | +4.3% | 29 |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | -0.04 | -0.0% | 24 |

---

## 5. Final Recommendation

**SPGI partially validates.** Best config: WF tune: conf=0.65 (1/4 gates).

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, financial_mean_reversion, financial_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+14.2%, Trades=21, WR=42.9%, Sharpe=0.14, PF=1.28, DD=-23.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.50, Test Sharpe=-1.89, Ratio=-381% (need >=50%) |
| Bootstrap | FAIL | p=0.1950, Sharpe CI=[-1.85, 4.60], WR CI=[23.8%, 61.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.9%, Median equity=$1,269, Survival=100.0% |
| Regime | FAIL | bull:17t/+33.8%, bear:2t/+4.8%, chop:1t/-6.0%, volatile:1t/-3.3% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

