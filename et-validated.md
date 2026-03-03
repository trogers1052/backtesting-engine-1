# ET (Energy Transfer) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 16.4 minutes
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
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 20 | 40.0% | +4.6% | 0.07 | 1.05 | -31.0% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+4.6%, Trades=20, WR=40.0%, Sharpe=0.07, PF=1.05, DD=-31.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.47, Test Sharpe=-7.57, Ratio=-1607% (need >=50%) |
| Bootstrap | FAIL | p=0.3539, Sharpe CI=[-2.83, 3.98], WR CI=[25.0%, 65.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.0%, Median equity=$1,087, Survival=100.0% |
| Regime | FAIL | bull:14t/+25.4%, bear:2t/+3.7%, chop:1t/-15.8%, volatile:3t/+2.4% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 12 | 50.0% | +24.5% | 0.24 |
| WF tune: conf=0.6 | 15 | 40.0% | +13.0% | 0.14 |
| WF tune: conf=0.55 | 18 | 38.9% | +10.1% | 0.11 |
| WF tune: conf=0.45 | 20 | 40.0% | +4.6% | 0.07 |
| BS tune: conf=0.4 | 20 | 40.0% | +4.6% | 0.07 |
| BS tune: full rules (10) | 20 | 40.0% | +4.6% | 0.07 |
| BS tune: energy rules (12) | 20 | 40.0% | +4.6% | 0.07 |
| BS tune: + volume_breakout | 20 | 40.0% | +4.6% | 0.07 |
| BS tune: + commodity_breakout | 20 | 40.0% | +4.6% | 0.07 |
| Regime tune: + dollar_weakness | 20 | 40.0% | +4.6% | 0.07 |
| WF tune: PT=12% | 14 | 35.7% | +4.2% | 0.07 |
| Regime tune: tighter stop 4% | 23 | 34.8% | +0.2% | 0.05 |
| WF tune: PT=15% | 12 | 33.3% | +2.7% | 0.04 |
| WF tune: PT=8% | 20 | 40.0% | -2.8% | -0.01 |
| WF tune: conf=0.65 | 14 | 21.4% | -8.0% | -0.14 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+24.5%, Trades=12, WR=50.0%, Sharpe=0.24, PF=1.53, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.58, Test Sharpe=-7.57, Ratio=-1315% (need >=50%) |
| Bootstrap | FAIL | p=0.1771, Sharpe CI=[-2.26, 8.15], WR CI=[33.3%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.6%, Median equity=$1,288, Survival=100.0% |
| Regime | FAIL | bull:8t/+43.6%, chop:1t/-15.8%, volatile:3t/+2.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+13.0%, Trades=15, WR=40.0%, Sharpe=0.14, PF=1.42, DD=-20.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.01, Test Sharpe=-7.57, Ratio=-66709% (need >=50%) |
| Bootstrap | FAIL | p=0.2364, Sharpe CI=[-2.69, 5.09], WR CI=[20.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.8%, Median equity=$1,172, Survival=100.0% |
| Regime | FAIL | bull:10t/+36.8%, bear:3t/-4.1%, volatile:2t/-13.7% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+10.1%, Trades=18, WR=38.9%, Sharpe=0.11, PF=1.23, DD=-20.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.01, Test Sharpe=-7.57, Ratio=-66709% (need >=50%) |
| Bootstrap | FAIL | p=0.2776, Sharpe CI=[-2.85, 4.33], WR CI=[22.2%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.2%, Median equity=$1,148, Survival=100.0% |
| Regime | FAIL | bull:13t/+35.3%, bear:3t/-4.1%, volatile:2t/-13.7% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=7** | FAIL | FAIL | **PASS** | FAIL | **0.24** | **+24.5%** | 12 |
| WF tune: conf=0.6 | FAIL | FAIL | **PASS** | FAIL | 0.14 | +13.0% | 15 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.11 | +10.1% | 18 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.07 | +4.6% | 20 |

---

## 5. Final Recommendation

**ET partially validates.** Best config: WF tune: cooldown=7 (1/4 gates).

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+24.5%, Trades=12, WR=50.0%, Sharpe=0.24, PF=1.53, DD=-23.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.58, Test Sharpe=-7.57, Ratio=-1315% (need >=50%) |
| Bootstrap | FAIL | p=0.1771, Sharpe CI=[-2.26, 8.15], WR CI=[33.3%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.6%, Median equity=$1,288, Survival=100.0% |
| Regime | FAIL | bull:8t/+43.6%, chop:1t/-15.8%, volatile:3t/+2.9% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

