# S (SentinelOne) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 7.7 minutes
**Category:** Tech / Cybersecurity (growth)

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

S — Cloud cybersecurity — beta ~1.4, AI-powered endpoint security, not yet profitable. Tech / Cybersecurity (growth).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 6 | 33.3% | -4.6% | -0.39 | 0.82 | -11.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 37.5% | -16.5% | -0.50 | 0.74 | -34.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 7 | 28.6% | -13.1% | -1.00 | 0.61 | -16.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 5 | 20.0% | -11.1% | -0.90 | 0.51 | -17.5% |
| Alt D: Recommended rules (3 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt E: Tech full (13 rules, 10%/5%) | 16 | 37.5% | -16.5% | -0.50 | 0.74 | -34.1% |
| Alt F: Tech momentum (15%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt G: Tech reversion (8%/4%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-16.5%, Trades=16, WR=37.5%, Sharpe=-0.50, PF=0.74, DD=-34.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.42, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6311, Sharpe CI=[-5.00, 3.05], WR CI=[12.5%, 62.5%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.8%, Median equity=$852, Survival=100.0% |
| Regime | **PASS** | bull:16t/-10.2% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| MC tune: max_loss=3.0% | 19 | 36.8% | +8.9% | 0.16 |
| WF tune: PT=7% | 18 | 50.0% | +3.5% | -0.03 |
| WF tune: PT=6% | 19 | 52.6% | +0.0% | -0.12 |
| WF tune: PT=8% | 17 | 47.1% | -2.8% | -0.26 |
| WF tune: cooldown=7 | 13 | 46.2% | -3.9% | -0.27 |
| MC tune: max_loss=4.0% | 16 | 37.5% | -10.9% | -0.38 |
| WF tune: conf=0.55 | 17 | 41.2% | -15.3% | -0.44 |
| WF tune: conf=0.6 | 17 | 41.2% | -15.3% | -0.44 |
| WF tune: PT=12% | 15 | 40.0% | -12.0% | -0.50 |
| WF tune: conf=0.45 | 16 | 37.5% | -16.5% | -0.50 |
| WF tune: ATR stops x2.5 | 16 | 37.5% | -16.5% | -0.50 |
| BS tune: conf=0.4 | 16 | 37.5% | -16.5% | -0.50 |
| BS tune: full rules (10) | 16 | 37.5% | -16.5% | -0.50 |
| MC tune: ATR stops x2.0 | 16 | 37.5% | -16.5% | -0.50 |
| WF tune: PT=15% | 14 | 35.7% | -14.0% | -0.51 |
| WF tune: conf=0.65 | 11 | 36.4% | -15.3% | -0.74 |
| WF tune: recommended rules | 0 | 0.0% | +0.0% | 0.00 |

### Full Validation of Top Candidates

### MC tune: max_loss=3.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 3 bars

**Performance:** Return=+8.9%, Trades=19, WR=36.8%, Sharpe=0.16, PF=1.16, DD=-28.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.13, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3086, Sharpe CI=[-2.80, 4.01], WR CI=[15.8%, 57.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.9%, Median equity=$1,132, Survival=100.0% |
| Regime | FAIL | bull:19t/+17.8% |

**Result: 1/4 gates passed**

---

### WF tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+3.5%, Trades=18, WR=50.0%, Sharpe=-0.03, PF=1.05, DD=-26.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.62, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3610, Sharpe CI=[-3.03, 4.32], WR CI=[27.8%, 72.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.8%, Median equity=$1,077, Survival=100.0% |
| Regime | FAIL | bull:18t/+13.3% |

**Result: 1/4 gates passed**

---

### WF tune: PT=6%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.0%, Trades=19, WR=52.6%, Sharpe=-0.12, PF=1.00, DD=-26.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.66, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3953, Sharpe CI=[-3.07, 4.03], WR CI=[31.6%, 73.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.1%, Median equity=$1,039, Survival=100.0% |
| Regime | FAIL | bull:19t/+9.3% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **MC tune: max_loss=3.0%** | FAIL | FAIL | **PASS** | FAIL | **0.16** | **+8.9%** | 19 |
| WF tune: PT=7% | FAIL | FAIL | **PASS** | FAIL | -0.03 | +3.5% | 18 |
| WF tune: PT=6% | FAIL | FAIL | **PASS** | FAIL | -0.12 | +0.0% | 19 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | **PASS** | -0.50 | -16.5% | 16 |

---

## 5. Final Recommendation

**S partially validates.** Best config: MC tune: max_loss=3.0% (1/4 gates).

### MC tune: max_loss=3.0%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 3 bars

**Performance:** Return=+8.9%, Trades=19, WR=36.8%, Sharpe=0.16, PF=1.16, DD=-28.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.13, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3086, Sharpe CI=[-2.80, 4.01], WR CI=[15.8%, 57.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.9%, Median equity=$1,132, Survival=100.0% |
| Regime | FAIL | bull:19t/+17.8% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

