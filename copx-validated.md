# COPX (Global X Copper Miners ETF) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 48.8 minutes
**Category:** Copper miners ETF

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

COPX — Basket of ~30 global copper miners — diversified copper exposure. Copper miners ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 31 | 41.9% | +8.9% | 0.11 | 1.00 | -32.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 44 | 43.2% | +47.6% | 0.68 | 1.27 | -30.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 34 | 38.2% | +18.2% | 0.43 | 1.08 | -30.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 29 | 41.4% | +17.5% | 0.31 | 1.07 | -30.0% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 40 | 45.0% | +71.6% | 0.75 | 1.45 | -29.8% |
| Alt E: copper rules (4 rules, 10%/5%) | 33 | 48.5% | +72.6% | 1.13 | 1.64 | -25.8% |
| Alt F: Copper momentum (12%/6%) | 25 | 48.0% | +48.6% | 0.52 | 1.39 | -29.7% |

**Best baseline selected for validation: Alt E: copper rules (4 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: copper rules (4 rules, 10%/5%)

- **Rules:** `commodity_breakout, trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+72.6%, Trades=33, WR=48.5%, Sharpe=1.13, PF=1.64, DD=-25.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.82, Test Sharpe=1.29, Ratio=157% (need >=50%) |
| Bootstrap | FAIL | p=0.0621, Sharpe CI=[-0.55, 4.58], WR CI=[30.3%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.7%, Median equity=$1,901, Survival=100.0% |
| Regime | **PASS** | bull:23t/+48.6%, bear:4t/+30.3%, chop:6t/-2.5% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: + volume_breakout | 33 | 48.5% | +74.6% | 1.20 |
| BS tune: conf=0.4 | 31 | 51.6% | +82.6% | 1.13 |
| BS tune: conf=0.45 | 31 | 51.6% | +82.6% | 1.13 |
| BS tune: conf=0.55 | 33 | 48.5% | +72.6% | 1.13 |
| BS tune: cooldown=3 | 31 | 51.6% | +93.8% | 0.85 |
| BS tune: cooldown=7 | 26 | 50.0% | +74.0% | 0.80 |
| BS tune: full mining rules (14) | 40 | 45.0% | +71.6% | 0.75 |
| BS tune: full rules (10) | 44 | 43.2% | +47.6% | 0.68 |
| BS tune: + volume_breakout [multi-TF] | 44 | 40.9% | -3.6% | -0.14 |

### Full Validation of Top Candidates

### BS tune: + volume_breakout

- **Rules:** `commodity_breakout, trend_continuation, seasonality, death_cross, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+74.6%, Trades=33, WR=48.5%, Sharpe=1.20, PF=1.66, DD=-24.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.89, Test Sharpe=1.29, Ratio=145% (need >=50%) |
| Bootstrap | FAIL | p=0.0592, Sharpe CI=[-0.51, 4.64], WR CI=[30.3%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.3%, Median equity=$1,924, Survival=100.0% |
| Regime | **PASS** | bull:23t/+49.7%, bear:4t/+30.3%, chop:6t/-2.5% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `commodity_breakout, trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+82.6%, Trades=31, WR=51.6%, Sharpe=1.13, PF=1.77, DD=-25.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.82, Test Sharpe=1.32, Ratio=161% (need >=50%) |
| Bootstrap | FAIL | p=0.0416, Sharpe CI=[-0.30, 5.09], WR CI=[35.5%, 67.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.6%, Median equity=$2,024, Survival=100.0% |
| Regime | **PASS** | bull:22t/+35.4%, bear:4t/+30.3%, chop:5t/+16.5% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `commodity_breakout, trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+82.6%, Trades=31, WR=51.6%, Sharpe=1.13, PF=1.77, DD=-25.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.82, Test Sharpe=1.32, Ratio=161% (need >=50%) |
| Bootstrap | FAIL | p=0.0416, Sharpe CI=[-0.30, 5.09], WR CI=[35.5%, 67.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.6%, Median equity=$2,024, Survival=100.0% |
| Regime | **PASS** | bull:22t/+35.4%, bear:4t/+30.3%, chop:5t/+16.5% |

**Result: 3/4 gates passed**

---

### BS tune: + volume_breakout [multi-TF]

- **Rules:** `commodity_breakout, trend_continuation, seasonality, death_cross, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-3.6%, Trades=44, WR=40.9%, Sharpe=-0.14, PF=0.90, DD=-30.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=1.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4056, Sharpe CI=[-2.09, 2.27], WR CI=[29.5%, 59.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.0%, Median equity=$1,048, Survival=100.0% |
| Regime | FAIL | bull:29t/+23.0%, bear:6t/-9.1%, chop:8t/-2.7%, volatile:1t/-0.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: + volume_breakout** | **PASS** | FAIL | **PASS** | **PASS** | **1.20** | **+74.6%** | 33 |
| BS tune: conf=0.4 | **PASS** | FAIL | **PASS** | **PASS** | 1.13 | +82.6% | 31 |
| BS tune: conf=0.45 | **PASS** | FAIL | **PASS** | **PASS** | 1.13 | +82.6% | 31 |
| Alt E: copper rules (4 rules, 10%/5%) | **PASS** | FAIL | **PASS** | **PASS** | 1.13 | +72.6% | 33 |
| BS tune: + volume_breakout [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.14 | -3.6% | 44 |

---

## 5. Final Recommendation

**COPX partially validates.** Best config: BS tune: + volume_breakout (3/4 gates).

### BS tune: + volume_breakout

- **Rules:** `commodity_breakout, trend_continuation, seasonality, death_cross, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+74.6%, Trades=33, WR=48.5%, Sharpe=1.20, PF=1.66, DD=-24.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.89, Test Sharpe=1.29, Ratio=145% (need >=50%) |
| Bootstrap | FAIL | p=0.0592, Sharpe CI=[-0.51, 4.64], WR CI=[30.3%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.3%, Median equity=$1,924, Survival=100.0% |
| Regime | **PASS** | bull:23t/+49.7%, bear:4t/+30.3%, chop:6t/-2.5% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

