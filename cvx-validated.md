# CVX (Chevron) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 20.5 minutes
**Category:** Large-cap integrated oil

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

CVX — Second-largest US integrated oil major — Permian, LNG, refining. Large-cap integrated oil.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 11.1% | -34.2% | -0.93 | 0.12 | -42.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 15 | 26.7% | -15.2% | -0.27 | 0.52 | -34.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 9 | 11.1% | -31.0% | -0.92 | 0.13 | -39.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 11.1% | -33.5% | -0.87 | 0.14 | -42.8% |
| Alt D: Energy rules (14 rules, 10%/5%) | 22 | 40.9% | -11.0% | -0.18 | 0.76 | -33.9% |
| Alt E: integrated sector rules (3 rules, 10%/5%) | 16 | 37.5% | -21.0% | -0.36 | 0.50 | -42.0% |

**Best baseline selected for validation: Alt D: Energy rules (14 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Energy rules (14 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-11.0%, Trades=22, WR=40.9%, Sharpe=-0.18, PF=0.76, DD=-33.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.73, Test Sharpe=0.82, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5632, Sharpe CI=[-3.76, 2.78], WR CI=[27.3%, 68.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.8%, Median equity=$918, Survival=100.0% |
| Regime | FAIL | bull:16t/-3.7%, bear:1t/-1.8%, chop:2t/+6.1%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=8% | 27 | 44.4% | -1.1% | -0.03 |
| BS tune: energy rules (14) | 22 | 40.9% | -11.0% | -0.18 |
| BS tune: conf=0.4 | 21 | 38.1% | -13.7% | -0.18 |
| Regime tune: tighter stop 4% | 23 | 39.1% | -11.8% | -0.18 |
| WF tune: conf=0.45 | 22 | 36.4% | -12.4% | -0.19 |
| WF tune: ATR stops x2.5 | 23 | 39.1% | -12.8% | -0.19 |
| BS tune: full rules (10) | 15 | 26.7% | -15.2% | -0.27 |
| WF tune: PT=15% | 22 | 36.4% | -15.8% | -0.27 |
| WF tune: PT=12% | 22 | 36.4% | -18.4% | -0.31 |
| BS tune: sector-specific rules | 15 | 40.0% | -17.0% | -0.32 |
| WF tune: cooldown=7 | 22 | 36.4% | -18.8% | -0.34 |
| WF tune: conf=0.55 | 23 | 30.4% | -22.1% | -0.48 |
| WF tune: conf=0.65 | 20 | 30.0% | -23.4% | -0.54 |
| WF tune: conf=0.6 | 24 | 29.2% | -27.9% | -0.58 |

### Full Validation of Top Candidates

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-1.1%, Trades=27, WR=44.4%, Sharpe=-0.03, PF=0.92, DD=-31.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.15, Test Sharpe=0.82, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4217, Sharpe CI=[-2.57, 3.09], WR CI=[29.6%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.0%, Median equity=$1,037, Survival=100.0% |
| Regime | FAIL | bull:18t/-0.5%, bear:3t/+1.9%, chop:3t/-0.8%, volatile:3t/+7.9% |

**Result: 1/4 gates passed**

---

### BS tune: energy rules (14)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-11.0%, Trades=22, WR=40.9%, Sharpe=-0.18, PF=0.76, DD=-33.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.73, Test Sharpe=0.82, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5632, Sharpe CI=[-3.76, 2.78], WR CI=[27.3%, 68.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.8%, Median equity=$918, Survival=100.0% |
| Regime | FAIL | bull:16t/-3.7%, bear:1t/-1.8%, chop:2t/+6.1%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-13.7%, Trades=21, WR=38.1%, Sharpe=-0.18, PF=0.73, DD=-36.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.57, Test Sharpe=0.82, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5815, Sharpe CI=[-3.99, 2.79], WR CI=[23.8%, 61.9%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.5%, Median equity=$894, Survival=100.0% |
| Regime | FAIL | bull:15t/-5.9%, bear:1t/-1.8%, chop:2t/+6.1%, volatile:3t/-4.2% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=8%** | FAIL | FAIL | **PASS** | FAIL | **-0.03** | **-1.1%** | 27 |
| Alt D: Energy rules (14 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | -0.18 | -11.0% | 22 |
| BS tune: energy rules (14) | FAIL | FAIL | **PASS** | FAIL | -0.18 | -11.0% | 22 |
| BS tune: conf=0.4 | FAIL | FAIL | FAIL | FAIL | -0.18 | -13.7% | 21 |

---

## 5. Final Recommendation

**CVX partially validates.** Best config: WF tune: PT=8% (1/4 gates).

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-1.1%, Trades=27, WR=44.4%, Sharpe=-0.03, PF=0.92, DD=-31.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.15, Test Sharpe=0.82, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4217, Sharpe CI=[-2.57, 3.09], WR CI=[29.6%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.0%, Median equity=$1,037, Survival=100.0% |
| Regime | FAIL | bull:18t/-0.5%, bear:3t/+1.9%, chop:3t/-0.8%, volatile:3t/+7.9% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

