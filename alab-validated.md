# ALAB (Astera Labs) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 37.8 minutes
**Category:** AI speculative (HIGH RISK)

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

ALAB — AI data center connectivity ICs — IPO Apr 2024, extremely high beta, narrative-driven. AI speculative (HIGH RISK).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 4 | 25.0% | -7.9% | -1.71 | 0.54 | -13.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 9 | 11.1% | -32.5% | -1.10 | 0.18 | -42.6% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 4 | 25.0% | -7.9% | -1.71 | 0.54 | -13.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 4 | 25.0% | -8.1% | -1.68 | 0.57 | -15.0% |
| Alt D: Tech rules (13 rules, 10%/5%) | 9 | 11.1% | -32.5% | -1.10 | 0.18 | -42.6% |
| Alt E: ai_speculative rules (3 rules, 10%/5%) | 4 | 25.0% | -2.7% | -2.65 | 0.79 | -16.5% |
| Alt F: AI spec wide (20%/10%) | 2 | 0.0% | -28.1% | -1.31 | 0.00 | -29.5% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-32.5%, Trades=9, WR=11.1%, Sharpe=-1.10, PF=0.18, DD=-42.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Error: float division by zero |
| Bootstrap | FAIL | p=0.9497, Sharpe CI=[-30.40, 0.54], WR CI=[0.0%, 33.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-43.6%, Median equity=$647, Survival=100.0% |
| Regime | **PASS** | bull:9t/-40.3% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=8% | 9 | 55.6% | +22.9% | 0.65 |
| WF tune: cooldown=7 | 7 | 14.3% | -23.9% | -0.75 |
| MC tune: max_loss=3.0% | 10 | 10.0% | -32.4% | -1.05 |
| WF tune: PT=12% | 9 | 11.1% | -32.5% | -1.10 |
| WF tune: conf=0.45 | 9 | 11.1% | -32.5% | -1.10 |
| WF tune: ATR stops x2.5 | 9 | 11.1% | -32.5% | -1.10 |
| WF tune: + tech_ema_pullback | 9 | 11.1% | -32.5% | -1.10 |
| WF tune: + tech_mean_reversion | 9 | 11.1% | -32.5% | -1.10 |
| BS tune: conf=0.4 | 9 | 11.1% | -32.5% | -1.10 |
| BS tune: full rules (10) | 9 | 11.1% | -32.5% | -1.10 |
| BS tune: tech rules (13) | 9 | 11.1% | -32.5% | -1.10 |
| MC tune: ATR stops x2.0 | 9 | 11.1% | -32.5% | -1.10 |
| MC tune: max_loss=4.0% | 10 | 10.0% | -33.0% | -1.13 |
| WF tune: conf=0.55 | 5 | 20.0% | -10.6% | -1.18 |
| WF tune: conf=0.6 | 5 | 20.0% | -10.6% | -1.18 |
| WF tune: conf=0.65 | 5 | 20.0% | -10.6% | -1.18 |
| BS tune: ai_speculative rules | 5 | 20.0% | -10.6% | -1.18 |
| WF tune: PT=15% | 9 | 0.0% | -47.1% | -1.49 |
| WF tune: cooldown=7 [multi-TF] | 8 | 62.5% | +32.3% | 0.70 |

### Full Validation of Top Candidates

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+22.9%, Trades=9, WR=55.6%, Sharpe=0.65, PF=1.96, DD=-12.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Error: float division by zero |
| Bootstrap | FAIL | p=0.1506, Sharpe CI=[-2.57, 9.63], WR CI=[22.2%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.0%, Median equity=$1,258, Survival=100.0% |
| Regime | FAIL | bull:9t/+26.3% |

**Result: 1/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=-23.9%, Trades=7, WR=14.3%, Sharpe=-0.75, PF=0.23, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Error: float division by zero |
| Bootstrap | FAIL | p=0.8999, Sharpe CI=[-30.04, 1.83], WR CI=[0.0%, 42.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.8%, Median equity=$736, Survival=100.0% |
| Regime | **PASS** | bull:7t/-27.8% |

**Result: 2/4 gates passed**

---

### MC tune: max_loss=3.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 3 bars

**Performance:** Return=-32.4%, Trades=10, WR=10.0%, Sharpe=-1.05, PF=0.18, DD=-42.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Error: float division by zero |
| Bootstrap | FAIL | p=0.9501, Sharpe CI=[-30.46, 0.46], WR CI=[0.0%, 30.0%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-42.6%, Median equity=$659, Survival=100.0% |
| Regime | **PASS** | bull:10t/-38.9% |

**Result: 1/4 gates passed**

---

### WF tune: cooldown=7 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+32.3%, Trades=8, WR=62.5%, Sharpe=0.70, PF=2.92, DD=-8.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.00, Test Sharpe=0.70, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0695, Sharpe CI=[-1.64, 11.18], WR CI=[25.0%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-16.9%, Median equity=$1,370, Survival=100.0% |
| Regime | FAIL | bull:8t/+34.7% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=7** | FAIL | FAIL | **PASS** | **PASS** | **-0.75** | **-23.9%** | 7 |
| WF tune: cooldown=7 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | 0.70 | +32.3% | 8 |
| WF tune: PT=8% | FAIL | FAIL | **PASS** | FAIL | 0.65 | +22.9% | 9 |
| MC tune: max_loss=3.0% | FAIL | FAIL | FAIL | **PASS** | -1.05 | -32.4% | 10 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | **PASS** | -1.10 | -32.5% | 9 |

---

## 5. Final Recommendation

**ALAB partially validates.** Best config: WF tune: cooldown=7 (2/4 gates).

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=-23.9%, Trades=7, WR=14.3%, Sharpe=-0.75, PF=0.23, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Error: float division by zero |
| Bootstrap | FAIL | p=0.8999, Sharpe CI=[-30.04, 1.83], WR CI=[0.0%, 42.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.8%, Median equity=$736, Survival=100.0% |
| Regime | **PASS** | bull:7t/-27.8% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

