# URA (Global X Uranium ETF) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 37.5 minutes
**Category:** Uranium ETF

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

URA — Basket of uranium miners and nuclear component manufacturers. Uranium ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 32 | 40.6% | -3.7% | -0.03 | 0.95 | -37.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 40 | 32.5% | -21.9% | -0.37 | 0.81 | -37.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 35 | 40.0% | +13.2% | 0.16 | 1.15 | -26.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 30 | 40.0% | +0.9% | -0.01 | 0.98 | -32.2% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 44 | 38.6% | +4.2% | 0.06 | 1.03 | -30.9% |
| Alt E: uranium rules (3 rules, 10%/5%) | 32 | 40.6% | -3.7% | -0.03 | 0.95 | -37.6% |
| Alt F: CCJ-proven (10%/6%, conf=0.65, cd=7) | 26 | 38.5% | -5.4% | -0.00 | 0.93 | -35.9% |

**Best baseline selected for validation: Alt B: Tighter stops (3 rules, 10%/4%)**

---

## 2. Full Validation

### Alt B: Tighter stops (3 rules, 10%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+13.2%, Trades=35, WR=40.0%, Sharpe=0.16, PF=1.15, DD=-26.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.96, Test Sharpe=1.28, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2695, Sharpe CI=[-1.77, 2.98], WR CI=[25.7%, 57.1%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-41.9%, Median equity=$1,227, Survival=99.9% |
| Regime | FAIL | bull:30t/+30.5%, bear:2t/+3.9%, chop:1t/+1.8%, volatile:2t/-5.9% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| MC tune: max_loss=3.0% | 39 | 35.9% | +28.4% | 0.37 |
| BS tune: full mining rules (14) | 51 | 37.3% | +19.6% | 0.21 |
| WF tune: conf=0.65 | 31 | 35.5% | +17.7% | 0.21 |
| WF tune: PT=12% | 34 | 38.2% | +15.6% | 0.19 |
| WF tune: conf=0.6 | 35 | 40.0% | +13.5% | 0.16 |
| WF tune: conf=0.45 | 35 | 40.0% | +13.2% | 0.16 |
| WF tune: conf=0.55 | 35 | 40.0% | +13.2% | 0.16 |
| WF tune: ATR stops x2.5 | 35 | 40.0% | +13.2% | 0.16 |
| WF tune: + miner_metal_ratio | 35 | 40.0% | +13.2% | 0.16 |
| BS tune: conf=0.4 | 35 | 40.0% | +13.2% | 0.16 |
| MC tune: ATR stops x2.0 | 35 | 40.0% | +13.2% | 0.16 |
| WF tune: + commodity_breakout | 37 | 37.8% | +12.9% | 0.16 |
| BS tune: + volume_breakout | 36 | 38.9% | +9.1% | 0.09 |
| WF tune: cooldown=3 | 38 | 34.2% | +4.4% | 0.04 |
| WF tune: PT=8% | 36 | 38.9% | +1.7% | -0.02 |
| WF tune: PT=15% | 27 | 33.3% | -3.3% | -0.15 |
| WF tune: cooldown=7 | 33 | 36.4% | -2.0% | -0.16 |
| BS tune: full rules (10) | 48 | 31.2% | -11.2% | -0.20 |

### Full Validation of Top Candidates

### MC tune: max_loss=3.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=+28.4%, Trades=39, WR=35.9%, Sharpe=0.37, PF=1.31, DD=-26.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.16, Test Sharpe=1.30, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1779, Sharpe CI=[-1.33, 3.12], WR CI=[23.1%, 53.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.0%, Median equity=$1,401, Survival=100.0% |
| Regime | FAIL | bull:34t/+44.9%, bear:2t/+2.7%, chop:1t/+1.8%, volatile:2t/-5.9% |

**Result: 1/4 gates passed**

---

### BS tune: full mining rules (14)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.6%, Trades=51, WR=37.3%, Sharpe=0.21, PF=1.12, DD=-28.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.75, Test Sharpe=1.26, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2248, Sharpe CI=[-1.35, 2.62], WR CI=[25.5%, 52.9%] |
| Monte Carlo | FAIL | Ruin=0.4%, P95 DD=-48.5%, Median equity=$1,336, Survival=99.6% |
| Regime | FAIL | bull:39t/+3.5%, bear:4t/+8.5%, chop:6t/+37.5%, volatile:2t/-5.9% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+17.7%, Trades=31, WR=35.5%, Sharpe=0.21, PF=1.20, DD=-30.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.29, Test Sharpe=1.29, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2435, Sharpe CI=[-1.80, 3.30], WR CI=[19.4%, 51.6%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.5%, Median equity=$1,261, Survival=100.0% |
| Regime | FAIL | bull:26t/+38.5%, bear:2t/+6.7%, chop:1t/-1.9%, volatile:2t/-10.4% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **MC tune: max_loss=3.0%** | FAIL | FAIL | **PASS** | FAIL | **0.37** | **+28.4%** | 39 |
| BS tune: full mining rules (14) | FAIL | FAIL | FAIL | FAIL | 0.21 | +19.6% | 51 |
| WF tune: conf=0.65 | FAIL | FAIL | FAIL | FAIL | 0.21 | +17.7% | 31 |
| Alt B: Tighter stops (3 rules, 10%/4%) | FAIL | FAIL | FAIL | FAIL | 0.16 | +13.2% | 35 |

---

## 5. Final Recommendation

**URA partially validates.** Best config: MC tune: max_loss=3.0% (1/4 gates).

### MC tune: max_loss=3.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=+28.4%, Trades=39, WR=35.9%, Sharpe=0.37, PF=1.31, DD=-26.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.16, Test Sharpe=1.30, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1779, Sharpe CI=[-1.33, 3.12], WR CI=[23.1%, 53.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.0%, Median equity=$1,401, Survival=100.0% |
| Regime | FAIL | bull:34t/+44.9%, bear:2t/+2.7%, chop:1t/+1.8%, volatile:2t/-5.9% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

