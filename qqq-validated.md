# QQQ (Invesco QQQ Trust) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 51.7 minutes
**Category:** Tech ETF

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

QQQ — Nasdaq-100 ETF — benchmark, well-documented mean-reversion strategies. Tech ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 54.5% | +28.1% | 0.49 | 2.02 | -11.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 15 | 60.0% | +59.0% | 0.61 | 2.62 | -13.7% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 54.5% | +34.2% | 0.58 | 2.68 | -8.8% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 62.5% | +40.9% | 0.55 | 4.06 | -10.2% |
| Alt D: Tech rules (13 rules, 10%/5%) | 13 | 61.5% | +46.5% | 0.45 | 3.05 | -12.1% |
| Alt E: tech_etf rules (3 rules, 10%/5%) | 12 | 50.0% | +10.1% | 0.14 | 1.33 | -11.9% |
| Alt F: QQQ tight (6%/3%) | 19 | 42.1% | -0.2% | -0.18 | 0.99 | -12.5% |
| Alt G: QQQ (8%/4%) | 13 | 53.8% | +14.7% | 0.25 | 1.60 | -13.3% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+59.0%, Trades=15, WR=60.0%, Sharpe=0.61, PF=2.62, DD=-13.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0207, Sharpe CI=[0.11, 10.66], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.9%, Median equity=$1,802, Survival=100.0% |
| Regime | FAIL | bull:13t/+58.4%, chop:2t/+6.1% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 15 | 60.0% | +59.0% | 0.61 |
| WF tune: + tech_ema_pullback | 15 | 60.0% | +59.0% | 0.61 |
| WF tune: + tech_mean_reversion | 15 | 60.0% | +59.0% | 0.61 |
| Regime tune: full rules (10) | 15 | 60.0% | +59.0% | 0.61 |
| WF tune: PT=8% | 15 | 66.7% | +64.6% | 0.60 |
| WF tune: ATR stops x2.5 | 17 | 52.9% | +56.9% | 0.57 |
| WF tune: PT=15% | 10 | 50.0% | +55.6% | 0.56 |
| Regime tune: tighter stop 4% | 17 | 52.9% | +56.0% | 0.56 |
| WF tune: cooldown=7 | 11 | 54.5% | +44.6% | 0.55 |
| WF tune: PT=12% | 11 | 54.5% | +55.6% | 0.52 |
| Regime tune: tech rules (13) | 13 | 61.5% | +46.5% | 0.45 |
| WF tune: conf=0.55 | 15 | 60.0% | +21.4% | 0.34 |
| WF tune: conf=0.6 | 15 | 60.0% | +21.4% | 0.34 |
| WF tune: conf=0.65 | 15 | 60.0% | +20.4% | 0.34 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | 42 | 50.0% | +61.4% | 0.64 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+59.0%, Trades=15, WR=60.0%, Sharpe=0.61, PF=2.62, DD=-13.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0207, Sharpe CI=[0.11, 10.66], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.9%, Median equity=$1,802, Survival=100.0% |
| Regime | FAIL | bull:13t/+58.4%, chop:2t/+6.1% |

**Result: 2/4 gates passed**

---

### WF tune: + tech_ema_pullback

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+59.0%, Trades=15, WR=60.0%, Sharpe=0.61, PF=2.62, DD=-13.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0207, Sharpe CI=[0.11, 10.66], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.9%, Median equity=$1,802, Survival=100.0% |
| Regime | FAIL | bull:13t/+58.4%, chop:2t/+6.1% |

**Result: 2/4 gates passed**

---

### WF tune: + tech_mean_reversion

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_mean_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+59.0%, Trades=15, WR=60.0%, Sharpe=0.61, PF=2.62, DD=-13.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0207, Sharpe CI=[0.11, 10.66], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.9%, Median equity=$1,802, Survival=100.0% |
| Regime | FAIL | bull:13t/+58.4%, chop:2t/+6.1% |

**Result: 2/4 gates passed**

---

### Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+61.4%, Trades=42, WR=50.0%, Sharpe=0.64, PF=2.28, DD=-20.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.04, Test Sharpe=0.30, Ratio=29% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.46, 4.52], WR CI=[45.2%, 73.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.4%, Median equity=$1,913, Survival=100.0% |
| Regime | FAIL | bull:15t/+52.8%, bear:11t/+1.4%, chop:10t/+9.4%, volatile:5t/+8.2%, crisis:1t/-2.1% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]** | FAIL | **PASS** | **PASS** | FAIL | **0.64** | **+61.4%** | 42 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | **PASS** | **PASS** | FAIL | 0.61 | +59.0% | 15 |
| WF tune: conf=0.45 | FAIL | **PASS** | **PASS** | FAIL | 0.61 | +59.0% | 15 |
| WF tune: + tech_ema_pullback | FAIL | **PASS** | **PASS** | FAIL | 0.61 | +59.0% | 15 |
| WF tune: + tech_mean_reversion | FAIL | **PASS** | **PASS** | FAIL | 0.61 | +59.0% | 15 |

---

## 5. Final Recommendation

**QQQ partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] (2/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+61.4%, Trades=42, WR=50.0%, Sharpe=0.64, PF=2.28, DD=-20.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.04, Test Sharpe=0.30, Ratio=29% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.46, 4.52], WR CI=[45.2%, 73.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.4%, Median equity=$1,913, Survival=100.0% |
| Regime | FAIL | bull:15t/+52.8%, bear:11t/+1.4%, chop:10t/+9.4%, volatile:5t/+8.2%, crisis:1t/-2.1% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

