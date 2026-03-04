# ET (Energy Transfer) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 31.8 minutes
**Category:** Large-cap midstream

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

ET — Midstream MLP — pipelines, terminals, natural gas processing. Large-cap midstream.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 36.4% | +3.5% | 0.02 | 1.07 | -23.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 20 | 40.0% | +4.6% | 0.07 | 1.05 | -31.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 36.4% | +5.7% | 0.05 | 1.14 | -23.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 37.5% | +8.5% | 0.09 | 1.29 | -23.9% |
| Alt D: Energy rules (14 rules, 10%/5%) | 23 | 43.5% | +8.7% | 0.11 | 1.05 | -31.0% |
| Alt E: midstream sector rules (3 rules, 10%/5%) | 12 | 41.7% | +5.3% | 0.02 | 1.06 | -15.5% |
| Alt F: Midstream 8% PT / 4% stop | 16 | 43.8% | +7.5% | 0.07 | 1.10 | -18.1% |

**Best baseline selected for validation: Alt D: Energy rules (14 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Energy rules (14 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+8.7%, Trades=23, WR=43.5%, Sharpe=0.11, PF=1.05, DD=-31.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.47, Test Sharpe=0.13, Ratio=27% (need >=50%) |
| Bootstrap | FAIL | p=0.3062, Sharpe CI=[-2.42, 3.83], WR CI=[26.1%, 69.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.8%, Median equity=$1,139, Survival=100.0% |
| Regime | FAIL | bull:17t/+30.2%, bear:2t/+3.7%, chop:1t/-15.8%, volatile:3t/+2.4% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: sector-specific rules | 15 | 53.3% | +37.0% | 0.44 |
| WF tune: cooldown=7 | 15 | 53.3% | +27.5% | 0.27 |
| WF tune: conf=0.6 | 17 | 47.1% | +19.4% | 0.22 |
| WF tune: conf=0.55 | 21 | 42.9% | +14.4% | 0.15 |
| WF tune: conf=0.45 | 23 | 43.5% | +8.7% | 0.11 |
| BS tune: conf=0.4 | 23 | 43.5% | +8.7% | 0.11 |
| BS tune: energy rules (14) | 23 | 43.5% | +8.7% | 0.11 |
| WF tune: PT=12% | 17 | 41.2% | +8.3% | 0.11 |
| WF tune: PT=15% | 15 | 40.0% | +6.8% | 0.09 |
| Regime tune: tighter stop 4% | 26 | 38.5% | +4.1% | 0.08 |
| BS tune: full rules (10) | 20 | 40.0% | +4.6% | 0.07 |
| WF tune: ATR stops x2.5 | 24 | 41.7% | +4.4% | 0.07 |
| WF tune: PT=8% | 23 | 43.5% | +1.0% | 0.04 |
| WF tune: conf=0.65 | 14 | 21.4% | -0.8% | -0.03 |
| WF tune: conf=0.6 [multi-TF] | 73 | 43.8% | -10.2% | -0.19 |

### Full Validation of Top Candidates

### BS tune: sector-specific rules

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+37.0%, Trades=15, WR=53.3%, Sharpe=0.44, PF=1.96, DD=-11.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.43, Test Sharpe=0.13, Ratio=30% (need >=50%) |
| Bootstrap | FAIL | p=0.0710, Sharpe CI=[-0.98, 7.19], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.9%, Median equity=$1,434, Survival=100.0% |
| Regime | FAIL | bull:13t/+35.7%, bear:1t/+10.0%, volatile:1t/-5.8% |

**Result: 1/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+27.5%, Trades=15, WR=53.3%, Sharpe=0.27, PF=1.50, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.58, Test Sharpe=-0.04, Ratio=-7% (need >=50%) |
| Bootstrap | FAIL | p=0.1553, Sharpe CI=[-1.73, 6.98], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.6%, Median equity=$1,328, Survival=100.0% |
| Regime | FAIL | bull:11t/+46.8%, chop:1t/-15.8%, volatile:3t/+2.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.4%, Trades=17, WR=47.1%, Sharpe=0.22, PF=1.48, DD=-20.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.01, Test Sharpe=0.35, Ratio=3044% (need >=50%) |
| Bootstrap | FAIL | p=0.1694, Sharpe CI=[-1.92, 5.42], WR CI=[29.4%, 76.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.1%, Median equity=$1,247, Survival=100.0% |
| Regime | FAIL | bull:12t/+43.1%, bear:3t/-4.1%, volatile:2t/-13.7% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.6 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-10.2%, Trades=73, WR=43.8%, Sharpe=-0.19, PF=0.82, DD=-22.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.23, Test Sharpe=0.31, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4537, Sharpe CI=[-1.66, 1.65], WR CI=[47.9%, 69.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$1,026, Survival=100.0% |
| Regime | FAIL | bull:53t/+19.3%, bear:5t/-10.8%, chop:9t/+3.5%, volatile:6t/-5.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | **PASS** | FAIL | **PASS** | FAIL | **0.22** | **+19.4%** | 17 |
| BS tune: sector-specific rules | FAIL | FAIL | **PASS** | FAIL | 0.44 | +37.0% | 15 |
| WF tune: cooldown=7 | FAIL | FAIL | **PASS** | FAIL | 0.27 | +27.5% | 15 |
| Alt D: Energy rules (14 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.11 | +8.7% | 23 |
| WF tune: conf=0.6 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.19 | -10.2% | 73 |

---

## 5. Final Recommendation

**ET partially validates.** Best config: WF tune: conf=0.6 (2/4 gates).

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+19.4%, Trades=17, WR=47.1%, Sharpe=0.22, PF=1.48, DD=-20.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.01, Test Sharpe=0.35, Ratio=3044% (need >=50%) |
| Bootstrap | FAIL | p=0.1694, Sharpe CI=[-1.92, 5.42], WR CI=[29.4%, 76.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.1%, Median equity=$1,247, Survival=100.0% |
| Regime | FAIL | bull:12t/+43.1%, bear:3t/-4.1%, volatile:2t/-13.7% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

