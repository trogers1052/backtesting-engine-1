# NOW (ServiceNow) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 30.0 minutes
**Category:** Enterprise SaaS

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

NOW — Enterprise IT workflow SaaS — consistent grower, EMA-respecting trend. Enterprise SaaS.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 50.0% | +21.2% | 0.26 | 1.56 | -13.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 21 | 52.4% | +46.7% | 0.39 | 1.51 | -22.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 46.2% | +19.3% | 0.25 | 1.49 | -14.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 12 | 50.0% | +35.6% | 0.37 | 1.92 | -12.3% |
| Alt D: Tech rules (13 rules, 10%/5%) | 26 | 50.0% | +33.2% | 0.30 | 1.32 | -27.3% |
| Alt E: saas rules (3 rules, 10%/5%) | 18 | 38.9% | -13.0% | -0.53 | 0.78 | -26.7% |
| Alt F: SaaS momentum (12%/6%) | 15 | 46.7% | +16.0% | 0.18 | 1.30 | -24.5% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+46.7%, Trades=21, WR=52.4%, Sharpe=0.39, PF=1.51, DD=-22.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.43, Test Sharpe=-0.92, Ratio=-212% (need >=50%) |
| Bootstrap | FAIL | p=0.1011, Sharpe CI=[-1.04, 5.52], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.9%, Median equity=$1,566, Survival=100.0% |
| Regime | FAIL | bull:19t/+69.3%, chop:2t/-15.5% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=12% | 18 | 55.6% | +70.2% | 0.51 |
| WF tune: conf=0.65 | 19 | 52.6% | +47.5% | 0.40 |
| WF tune: conf=0.45 | 21 | 52.4% | +46.7% | 0.39 |
| WF tune: ATR stops x2.5 | 21 | 52.4% | +46.7% | 0.39 |
| WF tune: + tech_ema_pullback | 21 | 52.4% | +46.7% | 0.39 |
| WF tune: + tech_mean_reversion | 21 | 52.4% | +46.7% | 0.39 |
| BS tune: conf=0.4 | 21 | 52.4% | +46.7% | 0.39 |
| BS tune: full rules (10) | 21 | 52.4% | +46.7% | 0.39 |
| WF tune: PT=15% | 16 | 43.8% | +51.2% | 0.35 |
| BS tune: tech rules (13) | 26 | 50.0% | +33.2% | 0.30 |
| Regime tune: tighter stop 4% | 22 | 45.5% | +29.3% | 0.30 |
| WF tune: PT=8% | 26 | 50.0% | +25.7% | 0.27 |
| WF tune: conf=0.6 | 17 | 41.2% | +12.9% | 0.15 |
| WF tune: conf=0.55 | 17 | 41.2% | +12.8% | 0.15 |
| WF tune: cooldown=7 | 19 | 36.8% | -7.4% | -0.06 |
| BS tune: saas rules | 17 | 41.2% | +0.1% | -0.08 |

### Full Validation of Top Candidates

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+70.2%, Trades=18, WR=55.6%, Sharpe=0.51, PF=1.90, DD=-24.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.92, Ratio=-164% (need >=50%) |
| Bootstrap | FAIL | p=0.0452, Sharpe CI=[-0.45, 7.22], WR CI=[33.3%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.9%, Median equity=$1,849, Survival=100.0% |
| Regime | FAIL | bull:16t/+86.5%, chop:2t/-15.5% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+47.5%, Trades=19, WR=52.6%, Sharpe=0.40, PF=1.74, DD=-18.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.79, Test Sharpe=-1.05, Ratio=-134% (need >=50%) |
| Bootstrap | FAIL | p=0.0713, Sharpe CI=[-0.85, 6.49], WR CI=[31.6%, 73.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$1,620, Survival=100.0% |
| Regime | FAIL | bull:17t/+52.1%, chop:1t/-7.4%, crisis:1t/+10.8% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+46.7%, Trades=21, WR=52.4%, Sharpe=0.39, PF=1.51, DD=-22.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.43, Test Sharpe=-0.92, Ratio=-212% (need >=50%) |
| Bootstrap | FAIL | p=0.1011, Sharpe CI=[-1.04, 5.52], WR CI=[33.3%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.9%, Median equity=$1,566, Survival=100.0% |
| Regime | FAIL | bull:19t/+69.3%, chop:2t/-15.5% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=12%** | FAIL | FAIL | **PASS** | FAIL | **0.51** | **+70.2%** | 18 |
| WF tune: conf=0.65 | FAIL | FAIL | **PASS** | FAIL | 0.40 | +47.5% | 19 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.39 | +46.7% | 21 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.39 | +46.7% | 21 |

---

## 5. Final Recommendation

**NOW partially validates.** Best config: WF tune: PT=12% (1/4 gates).

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+70.2%, Trades=18, WR=55.6%, Sharpe=0.51, PF=1.90, DD=-24.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.92, Ratio=-164% (need >=50%) |
| Bootstrap | FAIL | p=0.0452, Sharpe CI=[-0.45, 7.22], WR CI=[33.3%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.9%, Median equity=$1,849, Survival=100.0% |
| Regime | FAIL | bull:16t/+86.5%, chop:2t/-15.5% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

