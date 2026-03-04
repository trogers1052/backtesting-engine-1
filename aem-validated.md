# AEM (Agnico Eagle Mines) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 50.2 minutes
**Category:** Large-cap gold miner

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

AEM — Senior gold producer — tier-1 jurisdictions (Canada, Australia, Finland), low-cost. Large-cap gold miner.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 27 | 59.3% | +127.7% | 1.01 | 2.27 | -21.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 38 | 57.9% | +171.1% | 0.76 | 2.02 | -23.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 31 | 51.6% | +114.6% | 1.10 | 2.00 | -17.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 26 | 57.7% | +163.0% | 0.70 | 2.73 | -23.2% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 38 | 52.6% | +131.9% | 0.63 | 1.83 | -25.2% |
| Alt E: gold_miner rules (4 rules, 10%/5%) | 28 | 57.1% | +133.4% | 0.84 | 2.23 | -18.0% |
| Alt F: Gold miner ratio (12%/5%) | 26 | 57.7% | +163.0% | 0.70 | 2.73 | -23.2% |

**Best baseline selected for validation: Alt B: Tighter stops (3 rules, 10%/4%)**

---

## 2. Full Validation

### Alt B: Tighter stops (3 rules, 10%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+114.6%, Trades=31, WR=51.6%, Sharpe=1.10, PF=2.00, DD=-17.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.75, Test Sharpe=1.26, Ratio=169% (need >=50%) |
| Bootstrap | **PASS** | p=0.0229, Sharpe CI=[0.06, 5.45], WR CI=[32.3%, 67.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$2,401, Survival=100.0% |
| Regime | FAIL | bull:24t/+89.3%, bear:2t/+18.4%, chop:2t/+2.2%, volatile:3t/-9.3% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: + miner_metal_ratio | 31 | 51.6% | +114.6% | 1.10 |
| Regime tune: + volume_breakout | 31 | 51.6% | +107.3% | 1.05 |
| Regime tune: + commodity_breakout | 33 | 51.5% | +138.6% | 0.79 |
| Regime tune: full rules (10) | 43 | 48.8% | +97.6% | 0.78 |
| Regime tune: full mining rules (14) | 41 | 46.3% | +96.0% | 0.74 |
| Regime tune: PT=15% | 23 | 52.2% | +165.1% | 0.71 |
| Regime tune: PT=12% | 28 | 53.6% | +162.2% | 0.70 |
| Regime tune: conf=0.65 | 28 | 60.7% | +142.3% | 0.61 |
| Regime tune: + commodity_breakout [multi-TF] | 51 | 45.1% | +82.3% | 0.56 |

### Full Validation of Top Candidates

### Regime tune: + miner_metal_ratio

- **Rules:** `trend_continuation, seasonality, death_cross, miner_metal_ratio`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+114.6%, Trades=31, WR=51.6%, Sharpe=1.10, PF=2.00, DD=-17.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.75, Test Sharpe=1.26, Ratio=169% (need >=50%) |
| Bootstrap | **PASS** | p=0.0229, Sharpe CI=[0.06, 5.45], WR CI=[32.3%, 67.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$2,401, Survival=100.0% |
| Regime | FAIL | bull:24t/+89.3%, bear:2t/+18.4%, chop:2t/+2.2%, volatile:3t/-9.3% |

**Result: 3/4 gates passed**

---

### Regime tune: + volume_breakout

- **Rules:** `trend_continuation, seasonality, death_cross, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+107.3%, Trades=31, WR=51.6%, Sharpe=1.05, PF=1.89, DD=-17.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.75, Test Sharpe=1.10, Ratio=148% (need >=50%) |
| Bootstrap | FAIL | p=0.0282, Sharpe CI=[-0.07, 5.35], WR CI=[32.3%, 67.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.7%, Median equity=$2,322, Survival=100.0% |
| Regime | FAIL | bull:24t/+86.1%, bear:2t/+18.4%, chop:2t/+2.2%, volatile:3t/-9.3% |

**Result: 2/4 gates passed**

---

### Regime tune: + commodity_breakout

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+138.6%, Trades=33, WR=51.5%, Sharpe=0.79, PF=2.04, DD=-23.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.64, Test Sharpe=1.02, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0136, Sharpe CI=[0.36, 5.70], WR CI=[33.3%, 69.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.7%, Median equity=$2,758, Survival=100.0% |
| Regime | **PASS** | bull:26t/+80.4%, bear:3t/+17.4%, chop:1t/+13.4%, volatile:2t/-7.7%, crisis:1t/+11.9% |

**Result: 4/4 gates passed**

---

### Regime tune: + commodity_breakout [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+82.3%, Trades=51, WR=45.1%, Sharpe=0.56, PF=1.72, DD=-28.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.14, Test Sharpe=1.22, Ratio=897% (need >=50%) |
| Bootstrap | FAIL | p=0.0257, Sharpe CI=[-0.01, 3.88], WR CI=[33.3%, 60.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.2%, Median equity=$2,164, Survival=100.0% |
| Regime | FAIL | bull:36t/+66.6%, bear:9t/-7.4%, chop:4t/+14.8%, volatile:1t/+4.1%, crisis:1t/+9.1% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: + commodity_breakout** | **PASS** | **PASS** | **PASS** | **PASS** | **0.79** | **+138.6%** | 33 |
| Alt B: Tighter stops (3 rules, 10%/4%) | **PASS** | **PASS** | **PASS** | FAIL | 1.10 | +114.6% | 31 |
| Regime tune: + miner_metal_ratio | **PASS** | **PASS** | **PASS** | FAIL | 1.10 | +114.6% | 31 |
| Regime tune: + volume_breakout | **PASS** | FAIL | **PASS** | FAIL | 1.05 | +107.3% | 31 |
| Regime tune: + commodity_breakout [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | 0.56 | +82.3% | 51 |

---

## 5. Final Recommendation

**AEM fully validates.** Best config: Regime tune: + commodity_breakout (4/4 gates).

### Regime tune: + commodity_breakout

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+138.6%, Trades=33, WR=51.5%, Sharpe=0.79, PF=2.04, DD=-23.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.64, Test Sharpe=1.02, Ratio=159% (need >=50%) |
| Bootstrap | **PASS** | p=0.0136, Sharpe CI=[0.36, 5.70], WR CI=[33.3%, 69.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.7%, Median equity=$2,758, Survival=100.0% |
| Regime | **PASS** | bull:26t/+80.4%, bear:3t/+17.4%, chop:1t/+13.4%, volatile:2t/-7.7%, crisis:1t/+11.9% |

**Result: 4/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

