# NEE (NextEra Energy) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 66.7 minutes
**Category:** Regulated utility (renewable hybrid)

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

NEE — Largest US utility, 70% regulated (FP&L) + 30% renewables — very rate-sensitive. Regulated utility (renewable hybrid).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 22.2% | -13.8% | -0.63 | 0.52 | -22.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 37.5% | +0.9% | -0.12 | 0.93 | -24.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 9 | 11.1% | -22.2% | -1.38 | 0.23 | -23.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 12.5% | -20.7% | -0.65 | 0.27 | -28.7% |
| Alt D: Utility rules (13 rules, 10%/5%) | 27 | 44.4% | +0.5% | -0.14 | 0.95 | -30.8% |
| Alt E: regulated lean (4 rules, 10%/5%) | 19 | 42.1% | -5.0% | -0.46 | 0.90 | -25.0% |
| Alt F: Regulated tight (7%/4%, conf=0.55, cooldown=7) | 21 | 38.1% | -17.1% | -0.82 | 0.67 | -30.4% |
| Alt G: Regulated moderate (8%/4%) | 23 | 43.5% | -1.7% | -0.44 | 0.97 | -21.4% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.9%, Trades=16, WR=37.5%, Sharpe=-0.12, PF=0.93, DD=-24.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=0.56, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4195, Sharpe CI=[-3.87, 3.98], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.3%, Median equity=$1,034, Survival=100.0% |
| Regime | FAIL | bull:12t/+17.4%, bear:1t/-8.3%, volatile:3t/-1.7% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.55 | 17 | 52.9% | +21.8% | 0.59 |
| WF tune: conf=0.6 | 16 | 50.0% | +10.1% | 0.14 |
| WF tune: PT=15% | 16 | 37.5% | +9.8% | 0.11 |
| WF tune: conf=0.65 | 14 | 42.9% | +7.1% | 0.05 |
| WF tune: cooldown=7 | 16 | 37.5% | +3.0% | -0.06 |
| WF tune: PT=8% | 19 | 42.1% | +1.7% | -0.08 |
| Regime tune: tighter stop 4% | 17 | 35.3% | +1.1% | -0.12 |
| WF tune: conf=0.45 | 16 | 37.5% | +0.9% | -0.12 |
| WF tune: ATR stops x2.5 | 16 | 37.5% | +0.9% | -0.12 |
| WF tune: + utility_mean_reversion | 16 | 37.5% | +0.9% | -0.12 |
| WF tune: + utility_rate_reversion | 16 | 37.5% | +0.9% | -0.12 |
| BS tune: conf=0.4 | 16 | 37.5% | +0.9% | -0.12 |
| BS tune: full rules (10) | 16 | 37.5% | +0.9% | -0.12 |
| BS tune: utility rules (13) | 27 | 44.4% | +0.5% | -0.14 |
| WF tune: PT=12% | 17 | 35.3% | -0.8% | -0.23 |
| WF tune: PT=7% | 20 | 40.0% | -11.0% | -0.45 |
| BS tune: regulated rules | 20 | 45.0% | -9.8% | -0.74 |

### Full Validation of Top Candidates

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.8%, Trades=17, WR=52.9%, Sharpe=0.59, PF=1.91, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.68, Test Sharpe=0.95, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1259, Sharpe CI=[-1.85, 5.12], WR CI=[29.4%, 76.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.3%, Median equity=$1,283, Survival=100.0% |
| Regime | FAIL | bull:14t/+28.1%, bear:2t/+4.2%, volatile:1t/-4.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+10.1%, Trades=16, WR=50.0%, Sharpe=0.14, PF=1.42, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.68, Test Sharpe=0.05, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2447, Sharpe CI=[-2.96, 4.54], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.0%, Median equity=$1,141, Survival=100.0% |
| Regime | FAIL | bull:13t/+15.4%, bear:2t/+4.2%, volatile:1t/-4.3% |

**Result: 1/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+9.8%, Trades=16, WR=37.5%, Sharpe=0.11, PF=1.19, DD=-23.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.07, Test Sharpe=0.58, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3319, Sharpe CI=[-3.73, 4.22], WR CI=[12.5%, 62.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.2%, Median equity=$1,129, Survival=100.0% |
| Regime | FAIL | bull:12t/+29.9%, bear:1t/-8.3%, volatile:3t/-2.9% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.55** | FAIL | FAIL | **PASS** | FAIL | **0.59** | **+21.8%** | 17 |
| WF tune: conf=0.6 | FAIL | FAIL | **PASS** | FAIL | 0.14 | +10.1% | 16 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.11 | +9.8% | 16 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | -0.12 | +0.9% | 16 |

---

## 5. Final Recommendation

**NEE partially validates.** Best config: WF tune: conf=0.55 (1/4 gates).

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.8%, Trades=17, WR=52.9%, Sharpe=0.59, PF=1.91, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.68, Test Sharpe=0.95, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1259, Sharpe CI=[-1.85, 5.12], WR CI=[29.4%, 76.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.3%, Median equity=$1,283, Survival=100.0% |
| Regime | FAIL | bull:14t/+28.1%, bear:2t/+4.2%, volatile:1t/-4.3% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

