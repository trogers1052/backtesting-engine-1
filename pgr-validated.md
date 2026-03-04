# PGR (Progressive Corporation) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 65.2 minutes
**Category:** Insurance

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

PGR — Auto/home insurance — direct-to-consumer, Snapshot telematics. Insurance.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 50.0% | +14.2% | 0.15 | 1.30 | -28.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 50.0% | +21.5% | 0.24 | 1.39 | -24.6% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 33.3% | -2.6% | -0.04 | 0.95 | -26.8% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 11 | 54.5% | +6.6% | 0.07 | 1.15 | -27.7% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 19 | 47.4% | +20.3% | 0.20 | 1.30 | -29.8% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 14 | 50.0% | +13.1% | 0.15 | 1.22 | -33.7% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.5%, Trades=16, WR=50.0%, Sharpe=0.24, PF=1.39, DD=-24.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.72, Test Sharpe=-0.89, Ratio=-123% (need >=50%) |
| Bootstrap | FAIL | p=0.1847, Sharpe CI=[-2.14, 5.53], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.1%, Median equity=$1,281, Survival=100.0% |
| Regime | FAIL | bull:10t/+31.9%, chop:3t/-1.8%, volatile:2t/+6.6%, crisis:1t/-6.0% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 16 | 56.2% | +37.0% | 0.46 |
| WF tune: conf=0.55 | 22 | 59.1% | +36.8% | 0.40 |
| WF tune: conf=0.6 | 21 | 57.1% | +36.7% | 0.40 |
| Regime tune: tighter stop 4% | 17 | 52.9% | +34.1% | 0.36 |
| WF tune: PT=15% | 16 | 43.8% | +33.0% | 0.32 |
| WF tune: PT=12% | 13 | 53.8% | +30.1% | 0.29 |
| WF tune: conf=0.45 | 16 | 50.0% | +21.5% | 0.24 |
| BS tune: conf=0.4 | 16 | 50.0% | +21.5% | 0.24 |
| BS tune: full rules (10) | 16 | 50.0% | +21.5% | 0.24 |
| BS tune: + volume_breakout | 16 | 50.0% | +21.5% | 0.24 |
| BS tune: + financial_mean_reversion | 16 | 50.0% | +21.5% | 0.24 |
| BS tune: financial rules (12) | 19 | 47.4% | +20.3% | 0.20 |
| Regime tune: + financial_seasonality | 19 | 47.4% | +20.3% | 0.20 |
| WF tune: cooldown=7 | 16 | 50.0% | +17.5% | 0.19 |
| WF tune: PT=8% | 19 | 52.6% | +14.9% | 0.17 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+37.0%, Trades=16, WR=56.2%, Sharpe=0.46, PF=2.18, DD=-17.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.99, Test Sharpe=-1.07, Ratio=-108% (need >=50%) |
| Bootstrap | FAIL | p=0.0505, Sharpe CI=[-0.59, 6.75], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.3%, Median equity=$1,482, Survival=100.0% |
| Regime | FAIL | bull:11t/+33.0%, chop:3t/-3.4%, volatile:2t/+13.6% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.8%, Trades=22, WR=59.1%, Sharpe=0.40, PF=1.63, DD=-23.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.04, Test Sharpe=-1.03, Ratio=-99% (need >=50%) |
| Bootstrap | FAIL | p=0.0948, Sharpe CI=[-1.02, 5.48], WR CI=[36.4%, 77.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.7%, Median equity=$1,453, Survival=100.0% |
| Regime | FAIL | bull:15t/+52.0%, chop:4t/-16.5%, volatile:2t/+13.6%, crisis:1t/-6.0% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.7%, Trades=21, WR=57.1%, Sharpe=0.40, PF=1.63, DD=-23.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=1.04, Test Sharpe=-1.03, Ratio=-99% (need >=50%) |
| Bootstrap | FAIL | p=0.0992, Sharpe CI=[-1.12, 5.59], WR CI=[38.1%, 76.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.8%, Median equity=$1,450, Survival=100.0% |
| Regime | FAIL | bull:14t/+51.8%, chop:4t/-16.5%, volatile:2t/+13.6%, crisis:1t/-6.0% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | FAIL | FAIL | **PASS** | FAIL | **0.46** | **+37.0%** | 16 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.40 | +36.8% | 22 |
| WF tune: conf=0.6 | FAIL | FAIL | **PASS** | FAIL | 0.40 | +36.7% | 21 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.24 | +21.5% | 16 |

---

## 5. Final Recommendation

**PGR partially validates.** Best config: WF tune: conf=0.65 (1/4 gates).

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+37.0%, Trades=16, WR=56.2%, Sharpe=0.46, PF=2.18, DD=-17.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.99, Test Sharpe=-1.07, Ratio=-108% (need >=50%) |
| Bootstrap | FAIL | p=0.0505, Sharpe CI=[-0.59, 6.75], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.3%, Median equity=$1,482, Survival=100.0% |
| Regime | FAIL | bull:11t/+33.0%, chop:3t/-3.4%, volatile:2t/+13.6% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

