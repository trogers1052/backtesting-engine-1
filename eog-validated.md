# EOG (EOG Resources) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 21.6 minutes
**Category:** Large-cap E&P

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

EOG — Premium US shale E&P — Eagle Ford, Permian, Powder River Basin. Large-cap E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 9.1% | -41.5% | -0.97 | 0.18 | -47.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 22 | 27.3% | -26.9% | -0.63 | 0.61 | -42.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 8.3% | -31.7% | -0.75 | 0.22 | -39.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 11 | 9.1% | -37.9% | -0.76 | 0.29 | -46.9% |
| Alt D: Energy rules (14 rules, 10%/5%) | 34 | 26.5% | -33.6% | -0.53 | 0.62 | -50.2% |
| Alt E: upstream sector rules (3 rules, 10%/5%) | 20 | 25.0% | -40.0% | -0.87 | 0.32 | -47.5% |
| Alt F: upstream rules wider stops (10%/6%) | 20 | 25.0% | -46.6% | -0.91 | 0.29 | -53.2% |

**Best baseline selected for validation: Alt D: Energy rules (14 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Energy rules (14 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-33.6%, Trades=34, WR=26.5%, Sharpe=-0.53, PF=0.62, DD=-50.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.85, Test Sharpe=-0.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8197, Sharpe CI=[-4.35, 1.25], WR CI=[14.7%, 44.1%] |
| Monte Carlo | FAIL | Ruin=6.5%, P95 DD=-55.9%, Median equity=$655, Survival=93.5% |
| Regime | FAIL | bull:23t/-28.2%, bear:3t/-15.7%, chop:5t/-7.9%, volatile:3t/+17.3% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| MC tune: ATR stops x2.0 | 36 | 25.0% | -31.0% | -0.49 |
| WF tune: ATR stops x2.5 | 34 | 26.5% | -32.2% | -0.52 |
| WF tune: conf=0.45 | 34 | 26.5% | -33.6% | -0.53 |
| BS tune: conf=0.4 | 34 | 26.5% | -33.6% | -0.53 |
| BS tune: energy rules (14) | 34 | 26.5% | -33.6% | -0.53 |
| WF tune: PT=8% | 33 | 27.3% | -35.3% | -0.56 |
| WF tune: PT=15% | 23 | 17.4% | -36.9% | -0.58 |
| WF tune: PT=12% | 25 | 20.0% | -37.0% | -0.58 |
| BS tune: full rules (10) | 22 | 27.3% | -26.9% | -0.63 |
| MC tune: max_loss=4.0% | 35 | 20.0% | -44.5% | -0.70 |
| MC tune: max_loss=3.0% | 46 | 19.6% | -40.1% | -0.74 |
| WF tune: cooldown=7 | 25 | 24.0% | -45.0% | -0.76 |
| BS tune: sector-specific rules | 23 | 17.4% | -44.5% | -0.76 |
| WF tune: conf=0.6 | 29 | 20.7% | -46.6% | -0.79 |
| WF tune: conf=0.65 | 20 | 20.0% | -37.2% | -0.80 |
| WF tune: conf=0.55 | 29 | 20.7% | -46.9% | -0.80 |

### Full Validation of Top Candidates

### MC tune: ATR stops x2.0

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.0

**Performance:** Return=-31.0%, Trades=36, WR=25.0%, Sharpe=-0.49, PF=0.64, DD=-47.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.77, Test Sharpe=0.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7992, Sharpe CI=[-4.19, 1.32], WR CI=[13.9%, 41.7%] |
| Monte Carlo | FAIL | Ruin=3.8%, P95 DD=-53.9%, Median equity=$683, Survival=96.2% |
| Regime | FAIL | bull:25t/-24.5%, bear:3t/-15.7%, chop:5t/-7.9%, volatile:3t/+17.3% |

**Result: 0/4 gates passed**

---

### WF tune: ATR stops x2.5

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=-32.2%, Trades=34, WR=26.5%, Sharpe=-0.52, PF=0.63, DD=-49.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.85, Test Sharpe=0.06, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8035, Sharpe CI=[-4.25, 1.32], WR CI=[14.7%, 44.1%] |
| Monte Carlo | FAIL | Ruin=5.0%, P95 DD=-54.9%, Median equity=$673, Survival=95.0% |
| Regime | FAIL | bull:23t/-25.7%, bear:3t/-15.7%, chop:5t/-7.9%, volatile:3t/+17.3% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-33.6%, Trades=34, WR=26.5%, Sharpe=-0.53, PF=0.62, DD=-50.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.85, Test Sharpe=-0.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8197, Sharpe CI=[-4.35, 1.25], WR CI=[14.7%, 44.1%] |
| Monte Carlo | FAIL | Ruin=6.5%, P95 DD=-55.9%, Median equity=$655, Survival=93.5% |
| Regime | FAIL | bull:23t/-28.2%, bear:3t/-15.7%, chop:5t/-7.9%, volatile:3t/+17.3% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **MC tune: ATR stops x2.0** | FAIL | FAIL | FAIL | FAIL | **-0.49** | **-31.0%** | 36 |
| WF tune: ATR stops x2.5 | FAIL | FAIL | FAIL | FAIL | -0.52 | -32.2% | 34 |
| Alt D: Energy rules (14 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | -0.53 | -33.6% | 34 |
| WF tune: conf=0.45 | FAIL | FAIL | FAIL | FAIL | -0.53 | -33.6% | 34 |

---

## 5. Final Recommendation

**EOG partially validates.** Best config: MC tune: ATR stops x2.0 (0/4 gates).

### MC tune: ATR stops x2.0

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.0

**Performance:** Return=-31.0%, Trades=36, WR=25.0%, Sharpe=-0.49, PF=0.64, DD=-47.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.77, Test Sharpe=0.02, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7992, Sharpe CI=[-4.19, 1.32], WR CI=[13.9%, 41.7%] |
| Monte Carlo | FAIL | Ruin=3.8%, P95 DD=-53.9%, Median equity=$683, Survival=96.2% |
| Regime | FAIL | bull:25t/-24.5%, bear:3t/-15.7%, chop:5t/-7.9%, volatile:3t/+17.3% |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

