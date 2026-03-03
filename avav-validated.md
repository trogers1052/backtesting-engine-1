# AVAV (AeroVironment) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily entries + 5-minute exits (multi-TF hybrid)
**Validation Runtime:** 14.6 minutes
**Category:** Mid-cap defense tech

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

AVAV — Small drones, loitering munitions (Switchblade), defense tech. Mid-cap defense tech.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 22 | 31.8% | -12.1% | -0.29 | 0.86 | -26.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 40 | 45.0% | +70.0% | 0.44 | 1.42 | -25.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 25 | 28.0% | -12.6% | -0.20 | 0.85 | -27.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 23 | 30.4% | -10.4% | -0.20 | 0.89 | -29.6% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 17 | 23.5% | -23.9% | -0.47 | 0.62 | -33.4% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+70.0%, Trades=40, WR=45.0%, Sharpe=0.44, PF=1.42, DD=-25.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.94, Test Sharpe=-0.61, Ratio=-66% (need >=50%) |
| Bootstrap | FAIL | p=0.0835, Sharpe CI=[-0.67, 3.70], WR CI=[30.0%, 60.0%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-45.3%, Median equity=$1,949, Survival=99.9% |
| Regime | FAIL | bull:37t/+68.8%, bear:1t/+4.5%, volatile:2t/+11.9% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: + volume_breakout | 43 | 46.5% | +104.0% | 0.62 |
| WF tune: PT=15% | 34 | 41.2% | +89.3% | 0.58 |
| WF tune: PT=8% | 44 | 52.3% | +93.9% | 0.45 |
| WF tune: conf=0.45 | 40 | 45.0% | +70.3% | 0.44 |
| BS tune: conf=0.4 | 40 | 45.0% | +70.3% | 0.44 |
| BS tune: full rules (10) | 40 | 45.0% | +70.0% | 0.44 |
| WF tune: PT=12% | 40 | 42.5% | +84.2% | 0.43 |
| MC tune: max_loss=3.0% | 54 | 35.2% | +28.3% | 0.24 |
| MC tune: max_loss=4.0% | 49 | 38.8% | +21.7% | 0.19 |
| WF tune: conf=0.55 | 32 | 46.9% | +18.2% | 0.17 |
| WF tune: cooldown=7 | 36 | 41.7% | +7.1% | 0.07 |
| WF tune: conf=0.6 | 31 | 45.2% | +2.1% | 0.05 |
| WF tune: conf=0.65 | 29 | 37.9% | -15.3% | -0.11 |
| BS tune: + volume_breakout [multi-TF] | 72 | 43.1% | -18.9% | -0.12 |

### Full Validation of Top Candidates

### BS tune: + volume_breakout

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+104.0%, Trades=43, WR=46.5%, Sharpe=0.62, PF=1.55, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.94, Test Sharpe=0.74, Ratio=79% (need >=50%) |
| Bootstrap | FAIL | p=0.0431, Sharpe CI=[-0.29, 3.91], WR CI=[32.6%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.5%, Median equity=$2,410, Survival=99.9% |
| Regime | **PASS** | bull:37t/+68.8%, bear:1t/+4.5%, chop:3t/+23.2%, volatile:2t/+11.9% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+89.3%, Trades=34, WR=41.2%, Sharpe=0.58, PF=1.57, DD=-37.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.86, Test Sharpe=-0.19, Ratio=-22% (need >=50%) |
| Bootstrap | FAIL | p=0.0731, Sharpe CI=[-0.64, 4.02], WR CI=[26.5%, 58.8%] |
| Monte Carlo | FAIL | Ruin=0.2%, P95 DD=-45.1%, Median equity=$2,139, Survival=99.8% |
| Regime | FAIL | bull:31t/+75.4%, bear:2t/+4.7%, volatile:1t/+18.0% |

**Result: 0/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+93.9%, Trades=44, WR=52.3%, Sharpe=0.45, PF=1.54, DD=-27.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.73, Test Sharpe=-0.21, Ratio=-28% (need >=50%) |
| Bootstrap | FAIL | p=0.0422, Sharpe CI=[-0.27, 3.93], WR CI=[38.6%, 65.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-42.6%, Median equity=$2,325, Survival=99.9% |
| Regime | FAIL | bull:38t/+94.6%, bear:3t/-6.3%, chop:1t/-4.8%, volatile:2t/+19.3% |

**Result: 0/4 gates passed**

---

### BS tune: + volume_breakout [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-18.9%, Trades=72, WR=43.1%, Sharpe=-0.12, PF=0.91, DD=-51.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.75, Test Sharpe=-0.69, Ratio=-92% (need >=50%) |
| Bootstrap | FAIL | p=0.5134, Sharpe CI=[-1.81, 1.57], WR CI=[31.9%, 54.2%] |
| Monte Carlo | FAIL | Ruin=5.9%, P95 DD=-58.9%, Median equity=$878, Survival=94.2% |
| Regime | FAIL | bull:49t/+2.6%, bear:7t/-7.7%, chop:9t/+8.1%, volatile:7t/-0.8% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: + volume_breakout** | **PASS** | FAIL | FAIL | **PASS** | **0.62** | **+104.0%** | 43 |
| WF tune: PT=15% | FAIL | FAIL | FAIL | FAIL | 0.58 | +89.3% | 34 |
| WF tune: PT=8% | FAIL | FAIL | FAIL | FAIL | 0.45 | +93.9% | 44 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | 0.44 | +70.0% | 40 |
| BS tune: + volume_breakout [multi-TF] | FAIL | FAIL | FAIL | FAIL | -0.12 | -18.9% | 72 |

---

## 5. Final Recommendation

**AVAV partially validates.** Best config: BS tune: + volume_breakout (2/4 gates).

### BS tune: + volume_breakout

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+104.0%, Trades=43, WR=46.5%, Sharpe=0.62, PF=1.55, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.94, Test Sharpe=0.74, Ratio=79% (need >=50%) |
| Bootstrap | FAIL | p=0.0431, Sharpe CI=[-0.29, 3.91], WR CI=[32.6%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.5%, Median equity=$2,410, Survival=99.9% |
| Regime | **PASS** | bull:37t/+68.8%, bear:1t/+4.5%, chop:3t/+23.2%, volatile:2t/+11.9% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

