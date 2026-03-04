# NU (Nu Holdings) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 80.1 minutes
**Category:** Fintech (emerging market)

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

NU — Brazilian digital bank — fintech, credit cards, lending, crypto. Fintech (emerging market).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 15 | 53.3% | +44.8% | 0.60 | 1.78 | -20.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 26 | 50.0% | +73.9% | 0.77 | 1.67 | -21.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 18 | 44.4% | +38.1% | 0.46 | 1.59 | -18.1% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 15 | 60.0% | +50.8% | 0.57 | 2.12 | -14.1% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 27 | 48.1% | +55.6% | 0.60 | 1.46 | -29.3% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 18 | 44.4% | +3.8% | 0.01 | 1.04 | -40.7% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+73.9%, Trades=26, WR=50.0%, Sharpe=0.77, PF=1.67, DD=-21.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.07, Test Sharpe=0.42, Ratio=39% (need >=50%) |
| Bootstrap | FAIL | p=0.0559, Sharpe CI=[-0.60, 5.40], WR CI=[30.8%, 69.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.7%, Median equity=$1,871, Survival=100.0% |
| Regime | FAIL | bull:24t/+67.0%, chop:2t/+5.9% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 21 | 52.4% | +52.6% | 1.08 |
| WF tune: PT=15% | 19 | 47.4% | +76.9% | 0.86 |
| WF tune: PT=12% | 25 | 48.0% | +76.3% | 0.80 |
| WF tune: conf=0.45 | 26 | 50.0% | +73.9% | 0.77 |
| BS tune: conf=0.4 | 26 | 50.0% | +73.9% | 0.77 |
| BS tune: full rules (10) | 26 | 50.0% | +73.9% | 0.77 |
| BS tune: + volume_breakout | 26 | 50.0% | +73.9% | 0.77 |
| BS tune: + financial_mean_reversion | 26 | 50.0% | +73.9% | 0.77 |
| WF tune: PT=8% | 36 | 50.0% | +50.6% | 0.60 |
| BS tune: financial rules (12) | 27 | 48.1% | +55.6% | 0.60 |
| Regime tune: + financial_seasonality | 27 | 48.1% | +55.6% | 0.60 |
| Regime tune: tighter stop 4% | 30 | 43.3% | +62.1% | 0.59 |
| WF tune: conf=0.6 | 25 | 48.0% | +27.6% | 0.42 |
| WF tune: conf=0.65 | 25 | 48.0% | +27.6% | 0.42 |
| WF tune: conf=0.55 | 26 | 46.2% | +25.3% | 0.40 |
| WF tune: cooldown=7 [multi-TF] | 38 | 55.3% | +106.6% | 0.65 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+52.6%, Trades=21, WR=52.4%, Sharpe=1.08, PF=1.60, DD=-19.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.15, Test Sharpe=0.76, Ratio=66% (need >=50%) |
| Bootstrap | FAIL | p=0.0847, Sharpe CI=[-0.98, 5.70], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.5%, Median equity=$1,622, Survival=100.0% |
| Regime | FAIL | bull:19t/+51.7%, chop:2t/+5.9% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+76.9%, Trades=19, WR=47.4%, Sharpe=0.86, PF=1.81, DD=-21.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.29, Test Sharpe=0.35, Ratio=27% (need >=50%) |
| Bootstrap | FAIL | p=0.0593, Sharpe CI=[-0.68, 6.10], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.3%, Median equity=$1,882, Survival=100.0% |
| Regime | FAIL | bull:17t/+65.2%, chop:2t/+10.6% |

**Result: 1/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+76.3%, Trades=25, WR=48.0%, Sharpe=0.80, PF=1.54, DD=-33.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.32, Test Sharpe=0.34, Ratio=26% (need >=50%) |
| Bootstrap | FAIL | p=0.0617, Sharpe CI=[-0.57, 5.36], WR CI=[28.0%, 68.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.6%, Median equity=$1,900, Survival=100.0% |
| Regime | FAIL | bull:23t/+67.9%, chop:2t/+8.7% |

**Result: 1/4 gates passed**

---

### WF tune: cooldown=7 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+106.6%, Trades=38, WR=55.3%, Sharpe=0.65, PF=1.79, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.93, Test Sharpe=-0.05, Ratio=-6% (need >=50%) |
| Bootstrap | **PASS** | p=0.0115, Sharpe CI=[0.35, 5.02], WR CI=[42.1%, 73.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.3%, Median equity=$2,317, Survival=100.0% |
| Regime | FAIL | bull:30t/+91.7%, bear:1t/+3.3%, chop:6t/-3.2%, volatile:1t/+1.2% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=7** | **PASS** | FAIL | **PASS** | FAIL | **1.08** | **+52.6%** | 21 |
| WF tune: cooldown=7 [multi-TF] | FAIL | **PASS** | **PASS** | FAIL | 0.65 | +106.6% | 38 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.86 | +76.9% | 19 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | FAIL | 0.80 | +76.3% | 25 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.77 | +73.9% | 26 |

---

## 5. Final Recommendation

**NU partially validates.** Best config: WF tune: cooldown=7 (2/4 gates).

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+52.6%, Trades=21, WR=52.4%, Sharpe=1.08, PF=1.60, DD=-19.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.15, Test Sharpe=0.76, Ratio=66% (need >=50%) |
| Bootstrap | FAIL | p=0.0847, Sharpe CI=[-0.98, 5.70], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.5%, Median equity=$1,622, Survival=100.0% |
| Regime | FAIL | bull:19t/+51.7%, chop:2t/+5.9% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

