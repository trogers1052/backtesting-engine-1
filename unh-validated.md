# UNH (UnitedHealth Group) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 70.2 minutes
**Category:** Managed care (mean-reverting w/ gap risk)

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

UNH — Managed care — beta 0.42, 3rd largest in S&P 500 Healthcare, political gap risk. Managed care (mean-reverting w/ gap risk).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 6 | 16.7% | -18.2% | -1.08 | 0.24 | -19.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 20.0% | -19.3% | -0.74 | 0.24 | -22.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 6 | 16.7% | -17.1% | -1.00 | 0.25 | -17.8% |
| Alt C: Wider PT (3 rules, 12%/5%) | 5 | 0.0% | -23.9% | -1.27 | 0.00 | -24.5% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 20 | 35.0% | -25.4% | -0.90 | 0.35 | -28.1% |
| Alt E: managed_care lean (4 rules, 10%/5%) | 17 | 41.2% | -10.0% | -1.28 | 0.65 | -14.5% |
| Alt F: Managed care balanced (10%/6%) | 15 | 46.7% | -10.0% | -1.31 | 0.65 | -15.3% |
| Alt G: Managed care + financial rules (10%/7%) | 14 | 42.9% | -17.2% | -1.05 | 0.43 | -20.1% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-19.3%, Trades=10, WR=20.0%, Sharpe=-0.74, PF=0.24, DD=-22.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.75, Test Sharpe=-0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9227, Sharpe CI=[-10.55, 1.23], WR CI=[0.0%, 50.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.8%, Median equity=$727, Survival=100.0% |
| Regime | **PASS** | bull:7t/-16.5%, chop:2t/-6.6%, volatile:1t/-6.1% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=6% | 14 | 42.9% | -0.6% | -0.26 |
| WF tune: PT=7% | 14 | 35.7% | -14.4% | -0.63 |
| WF tune: PT=8% | 11 | 27.3% | -13.5% | -0.70 |
| WF tune: conf=0.65 | 9 | 22.2% | -13.3% | -0.71 |
| WF tune: cooldown=7 | 9 | 22.2% | -19.0% | -0.73 |
| WF tune: conf=0.45 | 10 | 20.0% | -19.3% | -0.74 |
| WF tune: ATR stops x2.5 | 10 | 20.0% | -19.3% | -0.74 |
| WF tune: + hc_mean_reversion | 10 | 20.0% | -19.3% | -0.74 |
| WF tune: + hc_pullback | 10 | 20.0% | -19.3% | -0.74 |
| BS tune: conf=0.4 | 10 | 20.0% | -19.3% | -0.74 |
| BS tune: full rules (10) | 10 | 20.0% | -19.3% | -0.74 |
| WF tune: conf=0.55 | 14 | 35.7% | -15.6% | -0.75 |
| WF tune: PT=15% | 7 | 14.3% | -17.3% | -0.77 |
| WF tune: PT=12% | 11 | 18.2% | -21.1% | -0.80 |
| WF tune: conf=0.6 | 14 | 35.7% | -17.1% | -0.82 |
| BS tune: healthcare rules (13) | 20 | 35.0% | -25.4% | -0.90 |
| BS tune: managed_care rules | 17 | 41.2% | -11.7% | -1.24 |
| WF tune: PT=6% [multi-TF] | 94 | 44.7% | +3.9% | -0.06 |

### Full Validation of Top Candidates

### WF tune: PT=6%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-0.6%, Trades=14, WR=42.9%, Sharpe=-0.26, PF=0.97, DD=-12.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.03, Test Sharpe=-0.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4782, Sharpe CI=[-3.98, 4.35], WR CI=[21.4%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.8%, Median equity=$998, Survival=100.0% |
| Regime | **PASS** | bull:10t/+9.0%, bear:2t/-13.1%, chop:1t/-0.2%, volatile:1t/+6.2% |

**Result: 2/4 gates passed**

---

### WF tune: PT=7%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-14.4%, Trades=14, WR=35.7%, Sharpe=-0.63, PF=0.51, DD=-17.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.03, Test Sharpe=-0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7914, Sharpe CI=[-5.75, 2.41], WR CI=[14.3%, 64.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.1%, Median equity=$797, Survival=100.0% |
| Regime | FAIL | bull:10t/-12.3%, bear:2t/-13.1%, chop:1t/-0.2%, volatile:1t/+6.2% |

**Result: 1/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-13.5%, Trades=11, WR=27.3%, Sharpe=-0.70, PF=0.46, DD=-16.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.76, Test Sharpe=-0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7929, Sharpe CI=[-7.74, 2.69], WR CI=[0.0%, 54.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.2%, Median equity=$806, Survival=100.0% |
| Regime | **PASS** | bull:8t/-7.2%, chop:2t/-5.2%, volatile:1t/-6.1% |

**Result: 2/4 gates passed**

---

### WF tune: PT=6% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+3.9%, Trades=94, WR=44.7%, Sharpe=-0.06, PF=1.08, DD=-11.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.28, Test Sharpe=0.17, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2185, Sharpe CI=[-0.93, 2.00], WR CI=[41.5%, 61.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.6%, Median equity=$1,224, Survival=100.0% |
| Regime | FAIL | bull:82t/+23.8%, bear:2t/+0.6%, chop:7t/-0.3%, volatile:3t/-0.5% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=6%** | FAIL | FAIL | **PASS** | **PASS** | **-0.26** | **-0.6%** | 14 |
| WF tune: PT=8% | FAIL | FAIL | **PASS** | **PASS** | -0.70 | -13.5% | 11 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | -0.74 | -19.3% | 10 |
| WF tune: PT=6% [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.06 | +3.9% | 94 |
| WF tune: PT=7% | FAIL | FAIL | **PASS** | FAIL | -0.63 | -14.4% | 14 |

---

## 5. Final Recommendation

**UNH partially validates.** Best config: WF tune: PT=6% (2/4 gates).

### WF tune: PT=6%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-0.6%, Trades=14, WR=42.9%, Sharpe=-0.26, PF=0.97, DD=-12.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.03, Test Sharpe=-0.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4782, Sharpe CI=[-3.98, 4.35], WR CI=[21.4%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.8%, Median equity=$998, Survival=100.0% |
| Regime | **PASS** | bull:10t/+9.0%, bear:2t/-13.1%, chop:1t/-0.2%, volatile:1t/+6.2% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

