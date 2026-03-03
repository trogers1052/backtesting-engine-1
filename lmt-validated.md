# LMT (Lockheed Martin) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily entries + 5-minute exits (multi-TF hybrid)
**Validation Runtime:** 15.0 minutes
**Category:** Large-cap defense

---

## Methodology

Validate-then-tune approach:

1. **Screen Baselines** — Test lean 3-rule + full 10-rule + alternatives
2. **Validate Best** — Run through all 4 statistical validation gates
3. **Diagnose** — Identify which gates pass/fail and why
4. **Targeted Tune** — Sweep only the parameters that address failing gates
5. **Re-validate** — Confirm tuned configs through all 4 gates

### Validation Gates

| Gate | Method | Pass Criteria | Purpose |
|------|--------|---------------|---------|
| Walk-Forward | 70/30 train/test split, 5-day embargo, 10-day purge | Test Sharpe >= 50% of Train Sharpe | Detects parameter overfitting |
| Bootstrap | 10,000 resamples with replacement | p < 0.05 AND Sharpe 95% CI excludes zero | Tests statistical significance |
| Monte Carlo | 10,000 trade-order permutations | Ruin probability < 10% AND P95 drawdown < 40% | Measures worst-case risk |
| Regime | SPY SMA_50/SMA_200 + VIX classification | No single regime contributes >70% of total profit | Detects regime dependency |

---

## 1. Baseline Screening

LMT — Defense prime contractor — F-35, missiles, space. Large-cap defense.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 22.2% | -8.6% | -0.49 | 0.54 | -15.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 27.8% | -3.7% | -0.17 | 0.80 | -20.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 9 | 22.2% | -5.9% | -0.44 | 0.62 | -13.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 22.2% | -7.6% | -0.45 | 0.59 | -15.8% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 8 | 12.5% | -14.6% | -1.22 | 0.24 | -16.6% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-3.7%, Trades=18, WR=27.8%, Sharpe=-0.17, PF=0.80, DD=-20.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.93, Test Sharpe=0.38, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5218, Sharpe CI=[-4.76, 3.12], WR CI=[16.7%, 61.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.3%, Median equity=$948, Survival=100.0% |
| Regime | FAIL | bull:13t/+17.3%, bear:2t/-13.0%, chop:1t/+0.0%, volatile:2t/-4.5% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 19 | 42.1% | +4.6% | 0.00 |
| WF tune: conf=0.55 | 19 | 42.1% | +4.4% | -0.00 |
| WF tune: conf=0.65 | 15 | 33.3% | +2.4% | -0.05 |
| WF tune: PT=8% | 21 | 42.9% | +2.0% | -0.06 |
| Regime tune: tighter stop 4% | 19 | 31.6% | +1.5% | -0.07 |
| WF tune: PT=15% | 16 | 25.0% | -2.5% | -0.12 |
| WF tune: PT=12% | 18 | 27.8% | -2.8% | -0.15 |
| WF tune: conf=0.45 | 18 | 27.8% | -3.7% | -0.17 |
| BS tune: conf=0.4 | 18 | 27.8% | -3.7% | -0.17 |
| BS tune: full rules (10) | 18 | 27.8% | -3.7% | -0.17 |
| BS tune: + volume_breakout | 18 | 27.8% | -3.7% | -0.17 |
| WF tune: cooldown=7 | 16 | 25.0% | -9.1% | -0.33 |
| WF tune: conf=0.6 [multi-TF] | 78 | 55.1% | -1.1% | -0.06 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.6%, Trades=19, WR=42.1%, Sharpe=0.00, PF=1.08, DD=-16.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.01, Test Sharpe=0.36, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3119, Sharpe CI=[-2.94, 3.92], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.6%, Median equity=$1,116, Survival=100.0% |
| Regime | **PASS** | bull:14t/+17.2%, bear:2t/-9.7%, volatile:2t/-4.5%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.4%, Trades=19, WR=42.1%, Sharpe=-0.00, PF=1.07, DD=-16.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.01, Test Sharpe=0.36, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3183, Sharpe CI=[-2.98, 3.90], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.8%, Median equity=$1,112, Survival=100.0% |
| Regime | **PASS** | bull:14t/+16.8%, bear:2t/-9.7%, volatile:2t/-4.5%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+2.4%, Trades=15, WR=33.3%, Sharpe=-0.05, PF=0.99, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-2.18, Test Sharpe=0.36, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3970, Sharpe CI=[-4.19, 4.02], WR CI=[13.3%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.2%, Median equity=$1,048, Survival=100.0% |
| Regime | FAIL | bull:11t/+20.1%, bear:1t/-1.9%, volatile:3t/-10.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-1.1%, Trades=78, WR=55.1%, Sharpe=-0.06, PF=0.97, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.37, Test Sharpe=0.30, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3806, Sharpe CI=[-1.53, 1.74], WR CI=[51.3%, 71.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.5%, Median equity=$1,089, Survival=100.0% |
| Regime | FAIL | bull:61t/+34.0%, bear:4t/+2.1%, chop:5t/-7.8%, volatile:7t/-8.3%, crisis:1t/-7.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | FAIL | **PASS** | **PASS** | **0.00** | **+4.6%** | 19 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | **PASS** | -0.00 | +4.4% | 19 |
| WF tune: conf=0.65 | FAIL | FAIL | **PASS** | FAIL | -0.05 | +2.4% | 15 |
| WF tune: conf=0.6 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.06 | -1.1% | 78 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | -0.17 | -3.7% | 18 |

---

## 5. Final Recommendation

**LMT partially validates.** Best config: WF tune: conf=0.6 (2/4 gates).

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.6%, Trades=19, WR=42.1%, Sharpe=0.00, PF=1.08, DD=-16.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.01, Test Sharpe=0.36, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3119, Sharpe CI=[-2.94, 3.92], WR CI=[26.3%, 68.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.6%, Median equity=$1,116, Survival=100.0% |
| Regime | **PASS** | bull:14t/+17.2%, bear:2t/-9.7%, volatile:2t/-4.5%, crisis:1t/+11.6% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

