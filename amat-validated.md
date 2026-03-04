# AMAT (Applied Materials) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 45.3 minutes
**Category:** Semi equipment

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

AMAT — Semiconductor equipment (broad-based) — diversified semi capex exposure. Semi equipment.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 15 | 60.0% | +59.2% | 0.49 | 1.99 | -30.3% |
| Alt A: Full general rules (10 rules, 10%/5%) | 27 | 63.0% | +193.4% | 0.98 | 2.87 | -28.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 16 | 56.2% | +62.1% | 0.49 | 2.00 | -27.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 13 | 61.5% | +86.3% | 0.58 | 2.48 | -28.2% |
| Alt D: Tech rules (13 rules, 10%/5%) | 33 | 51.5% | +110.0% | 0.60 | 1.82 | -46.0% |
| Alt E: semi_equip rules (4 rules, 10%/5%) | 24 | 41.7% | +8.1% | 0.09 | 1.11 | -32.0% |
| Alt F: Semi wider PT+stops (15%/7%) | 17 | 41.2% | +35.3% | 0.32 | 1.51 | -32.9% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+193.4%, Trades=27, WR=63.0%, Sharpe=0.98, PF=2.87, DD=-28.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.92, Test Sharpe=0.82, Ratio=89% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[1.28, 8.06], WR CI=[44.4%, 81.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.6%, Median equity=$3,537, Survival=100.0% |
| Regime | FAIL | bull:25t/+116.2%, bear:2t/+23.8% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=15% | 20 | 65.0% | +267.7% | 1.09 |
| Regime tune: PT=12% | 24 | 62.5% | +225.0% | 0.98 |
| Regime tune: + tech_ema_pullback | 27 | 63.0% | +193.4% | 0.98 |
| Regime tune: + tech_mean_reversion | 27 | 63.0% | +193.4% | 0.98 |
| Regime tune: full rules (10) | 27 | 63.0% | +193.4% | 0.98 |
| Regime tune: conf=0.65 | 25 | 60.0% | +140.3% | 0.94 |
| Regime tune: tighter stop 4% | 29 | 55.2% | +155.4% | 0.77 |
| Regime tune: tech rules (13) | 33 | 51.5% | +110.0% | 0.60 |
| Regime tune: PT=15% [multi-TF] | 63 | 47.6% | +148.2% | 0.83 |

### Full Validation of Top Candidates

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+267.7%, Trades=20, WR=65.0%, Sharpe=1.09, PF=4.12, DD=-25.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.93, Test Sharpe=0.94, Ratio=101% (need >=50%) |
| Bootstrap | **PASS** | p=0.0010, Sharpe CI=[1.96, 10.28], WR CI=[45.0%, 85.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.1%, Median equity=$4,237, Survival=100.0% |
| Regime | FAIL | bull:18t/+129.9%, bear:1t/+15.3%, chop:1t/+16.1% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+225.0%, Trades=24, WR=62.5%, Sharpe=0.98, PF=3.24, DD=-26.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.96, Test Sharpe=0.79, Ratio=83% (need >=50%) |
| Bootstrap | **PASS** | p=0.0014, Sharpe CI=[1.47, 8.45], WR CI=[41.7%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.2%, Median equity=$3,828, Survival=100.0% |
| Regime | FAIL | bull:22t/+120.5%, bear:1t/+15.3%, chop:1t/+14.4% |

**Result: 3/4 gates passed**

---

### Regime tune: + tech_ema_pullback

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+193.4%, Trades=27, WR=63.0%, Sharpe=0.98, PF=2.87, DD=-28.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.92, Test Sharpe=0.82, Ratio=89% (need >=50%) |
| Bootstrap | **PASS** | p=0.0018, Sharpe CI=[1.28, 8.06], WR CI=[44.4%, 81.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.6%, Median equity=$3,537, Survival=100.0% |
| Regime | FAIL | bull:25t/+116.2%, bear:2t/+23.8% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+148.2%, Trades=63, WR=47.6%, Sharpe=0.83, PF=1.91, DD=-33.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=1.00, Ratio=118% (need >=50%) |
| Bootstrap | **PASS** | p=0.0097, Sharpe CI=[0.35, 3.53], WR CI=[38.1%, 63.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.6%, Median equity=$3,073, Survival=100.0% |
| Regime | **PASS** | bull:40t/+91.8%, bear:6t/+13.0%, chop:11t/+26.8%, volatile:5t/+2.2%, crisis:1t/-5.5% |

**Result: 4/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=15% [multi-TF]** | **PASS** | **PASS** | **PASS** | **PASS** | **0.83** | **+148.2%** | 63 |
| Regime tune: PT=15% | **PASS** | **PASS** | **PASS** | FAIL | 1.09 | +267.7% | 20 |
| Regime tune: PT=12% | **PASS** | **PASS** | **PASS** | FAIL | 0.98 | +225.0% | 24 |
| Alt A: Full general rules (10 rules, 10%/5%) | **PASS** | **PASS** | **PASS** | FAIL | 0.98 | +193.4% | 27 |
| Regime tune: + tech_ema_pullback | **PASS** | **PASS** | **PASS** | FAIL | 0.98 | +193.4% | 27 |

---

## 5. Final Recommendation

**AMAT fully validates.** Best config: Regime tune: PT=15% [multi-TF] (4/4 gates).

### Regime tune: PT=15% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+148.2%, Trades=63, WR=47.6%, Sharpe=0.83, PF=1.91, DD=-33.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=1.00, Ratio=118% (need >=50%) |
| Bootstrap | **PASS** | p=0.0097, Sharpe CI=[0.35, 3.53], WR CI=[38.1%, 63.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.6%, Median equity=$3,073, Survival=100.0% |
| Regime | **PASS** | bull:40t/+91.8%, bear:6t/+13.0%, chop:11t/+26.8%, volatile:5t/+2.2%, crisis:1t/-5.5% |

**Result: 4/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

