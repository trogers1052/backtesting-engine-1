# CVX (Chevron) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 16.2 minutes
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
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 15 | 26.7% | -15.2% | -0.27 | 0.52 | -34.3% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-15.2%, Trades=15, WR=26.7%, Sharpe=-0.27, PF=0.52, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.61, Test Sharpe=0.56, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6726, Sharpe CI=[-6.22, 2.70], WR CI=[13.3%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median equity=$856, Survival=100.0% |
| Regime | FAIL | bull:10t/-6.4%, bear:1t/-1.8%, chop:1t/+1.2%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 15 | 26.7% | -15.2% | -0.27 |
| BS tune: conf=0.4 | 15 | 26.7% | -15.2% | -0.27 |
| BS tune: full rules (10) | 15 | 26.7% | -15.2% | -0.27 |
| BS tune: energy rules (12) | 15 | 26.7% | -15.2% | -0.27 |
| BS tune: + volume_breakout | 15 | 26.7% | -15.2% | -0.27 |
| BS tune: + commodity_breakout | 15 | 26.7% | -15.2% | -0.27 |
| Regime tune: + dollar_weakness | 15 | 26.7% | -15.2% | -0.27 |
| WF tune: PT=8% | 21 | 33.3% | -17.3% | -0.31 |
| Regime tune: tighter stop 4% | 17 | 23.5% | -18.9% | -0.31 |
| WF tune: PT=15% | 16 | 25.0% | -18.6% | -0.33 |
| WF tune: PT=12% | 16 | 25.0% | -21.0% | -0.40 |
| WF tune: cooldown=7 | 16 | 25.0% | -20.5% | -0.41 |
| WF tune: conf=0.65 | 16 | 18.8% | -24.7% | -0.63 |
| WF tune: conf=0.55 | 17 | 17.6% | -26.2% | -0.68 |
| WF tune: conf=0.6 | 17 | 17.6% | -26.0% | -0.71 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-15.2%, Trades=15, WR=26.7%, Sharpe=-0.27, PF=0.52, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.61, Test Sharpe=0.56, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6726, Sharpe CI=[-6.22, 2.70], WR CI=[13.3%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median equity=$856, Survival=100.0% |
| Regime | FAIL | bull:10t/-6.4%, bear:1t/-1.8%, chop:1t/+1.2%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-15.2%, Trades=15, WR=26.7%, Sharpe=-0.27, PF=0.52, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.61, Test Sharpe=0.56, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6726, Sharpe CI=[-6.22, 2.70], WR CI=[13.3%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median equity=$856, Survival=100.0% |
| Regime | FAIL | bull:10t/-6.4%, bear:1t/-1.8%, chop:1t/+1.2%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-15.2%, Trades=15, WR=26.7%, Sharpe=-0.27, PF=0.52, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.61, Test Sharpe=0.56, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6726, Sharpe CI=[-6.22, 2.70], WR CI=[13.3%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median equity=$856, Survival=100.0% |
| Regime | FAIL | bull:10t/-6.4%, bear:1t/-1.8%, chop:1t/+1.2%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%)** | FAIL | FAIL | **PASS** | FAIL | **-0.27** | **-15.2%** | 15 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | -0.27 | -15.2% | 15 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | FAIL | -0.27 | -15.2% | 15 |
| BS tune: full rules (10) | FAIL | FAIL | **PASS** | FAIL | -0.27 | -15.2% | 15 |

---

## 5. Final Recommendation

**CVX partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) (1/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-15.2%, Trades=15, WR=26.7%, Sharpe=-0.27, PF=0.52, DD=-34.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.61, Test Sharpe=0.56, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6726, Sharpe CI=[-6.22, 2.70], WR CI=[13.3%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median equity=$856, Survival=100.0% |
| Regime | FAIL | bull:10t/-6.4%, bear:1t/-1.8%, chop:1t/+1.2%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

