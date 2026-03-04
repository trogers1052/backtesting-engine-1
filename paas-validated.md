# PAAS (Pan American Silver) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 59.6 minutes
**Category:** Large-cap silver miner

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

PAAS — Largest primary silver producer in Americas — also gold, zinc, lead. Large-cap silver miner.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 31 | 51.6% | +39.7% | 0.32 | 1.36 | -36.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 42 | 40.5% | -12.7% | -0.00 | 0.92 | -41.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 32 | 40.6% | +20.5% | 0.19 | 1.22 | -40.2% |
| Alt C: Wider PT (3 rules, 12%/5%) | 28 | 46.4% | +56.5% | 0.38 | 1.54 | -37.8% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 41 | 36.6% | -28.9% | -0.09 | 0.80 | -50.9% |
| Alt E: silver_miner rules (4 rules, 10%/5%) | 31 | 51.6% | +37.6% | 0.30 | 1.34 | -37.9% |
| Alt F: Silver miner strict risk (10%/4%) | 30 | 43.3% | +56.9% | 0.32 | 1.57 | -33.9% |
| Alt G: Silver miner wider PT (12%/4%) | 31 | 38.7% | +17.5% | 0.17 | 1.19 | -41.3% |

**Best baseline selected for validation: Alt C: Wider PT (3 rules, 12%/5%)**

---

## 2. Full Validation

### Alt C: Wider PT (3 rules, 12%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+56.5%, Trades=28, WR=46.4%, Sharpe=0.38, PF=1.54, DD=-37.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.56, Test Sharpe=0.81, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1106, Sharpe CI=[-1.11, 4.55], WR CI=[28.6%, 64.3%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.5%, Median equity=$1,700, Survival=100.0% |
| Regime | FAIL | bull:22t/+71.9%, bear:1t/+13.5%, chop:3t/-5.1%, volatile:2t/-11.8% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 28 | 53.6% | +104.5% | 0.59 |
| WF tune: PT=15% | 27 | 44.4% | +75.8% | 0.51 |
| WF tune: cooldown=7 | 28 | 42.9% | +57.2% | 0.48 |
| WF tune: conf=0.45 | 28 | 46.4% | +56.5% | 0.38 |
| WF tune: conf=0.55 | 28 | 46.4% | +56.5% | 0.38 |
| WF tune: ATR stops x2.5 | 28 | 46.4% | +56.5% | 0.38 |
| WF tune: + miner_metal_ratio | 28 | 46.4% | +56.5% | 0.38 |
| BS tune: conf=0.4 | 28 | 46.4% | +56.5% | 0.38 |
| MC tune: ATR stops x2.0 | 28 | 46.4% | +56.5% | 0.38 |
| WF tune: conf=0.65 | 21 | 42.9% | +47.7% | 0.37 |
| BS tune: silver_miner rules | 28 | 46.4% | +53.6% | 0.36 |
| BS tune: + volume_breakout | 28 | 46.4% | +53.6% | 0.36 |
| WF tune: cooldown=3 | 35 | 42.9% | +31.5% | 0.25 |
| WF tune: + commodity_breakout | 28 | 39.3% | +20.9% | 0.23 |
| BS tune: full rules (10) | 41 | 41.5% | +20.3% | 0.19 |
| WF tune: PT=8% | 34 | 50.0% | +15.7% | 0.19 |
| MC tune: max_loss=4.0% | 31 | 38.7% | +18.8% | 0.18 |
| BS tune: full mining rules (14) | 39 | 38.5% | +11.0% | 0.16 |
| MC tune: max_loss=3.0% | 34 | 32.4% | +3.7% | 0.00 |
| WF tune: conf=0.6 [multi-TF] | 49 | 44.9% | +48.0% | 0.37 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+104.5%, Trades=28, WR=53.6%, Sharpe=0.59, PF=1.98, DD=-27.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.18, Test Sharpe=0.81, Ratio=442% (need >=50%) |
| Bootstrap | FAIL | p=0.0338, Sharpe CI=[-0.21, 5.46], WR CI=[35.7%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.3%, Median equity=$2,249, Survival=100.0% |
| Regime | FAIL | bull:21t/+85.3%, bear:1t/+12.4%, chop:4t/-0.3%, volatile:2t/-1.7% |

**Result: 2/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+75.8%, Trades=27, WR=44.4%, Sharpe=0.51, PF=1.70, DD=-37.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.16, Test Sharpe=1.05, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0893, Sharpe CI=[-0.98, 4.55], WR CI=[25.9%, 63.0%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-44.3%, Median equity=$1,889, Survival=99.9% |
| Regime | FAIL | bull:21t/+80.6%, bear:1t/+13.5%, chop:3t/-0.3%, volatile:2t/-11.8% |

**Result: 0/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+57.2%, Trades=28, WR=42.9%, Sharpe=0.48, PF=1.49, DD=-28.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.14, Test Sharpe=0.88, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0954, Sharpe CI=[-0.95, 4.47], WR CI=[28.6%, 64.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.3%, Median equity=$1,730, Survival=100.0% |
| Regime | FAIL | bull:21t/+82.9%, bear:2t/+6.9%, chop:3t/-6.9%, volatile:2t/-14.3% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.6 [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+48.0%, Trades=49, WR=44.9%, Sharpe=0.37, PF=1.45, DD=-20.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.16, Test Sharpe=0.99, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1026, Sharpe CI=[-0.79, 3.21], WR CI=[30.6%, 59.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.9%, Median equity=$1,668, Survival=100.0% |
| Regime | FAIL | bull:37t/+63.2%, bear:6t/-15.9%, chop:4t/+2.7%, volatile:1t/-0.1%, crisis:1t/+12.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | **PASS** | FAIL | **PASS** | FAIL | **0.59** | **+104.5%** | 28 |
| WF tune: conf=0.6 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | 0.37 | +48.0% | 49 |
| WF tune: PT=15% | FAIL | FAIL | FAIL | FAIL | 0.51 | +75.8% | 27 |
| WF tune: cooldown=7 | FAIL | FAIL | FAIL | FAIL | 0.48 | +57.2% | 28 |
| Alt C: Wider PT (3 rules, 12%/5%) | FAIL | FAIL | FAIL | FAIL | 0.38 | +56.5% | 28 |

---

## 5. Final Recommendation

**PAAS partially validates.** Best config: WF tune: conf=0.6 (2/4 gates).

### WF tune: conf=0.6

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+104.5%, Trades=28, WR=53.6%, Sharpe=0.59, PF=1.98, DD=-27.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.18, Test Sharpe=0.81, Ratio=442% (need >=50%) |
| Bootstrap | FAIL | p=0.0338, Sharpe CI=[-0.21, 5.46], WR CI=[35.7%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.3%, Median equity=$2,249, Survival=100.0% |
| Regime | FAIL | bull:21t/+85.3%, bear:1t/+12.4%, chop:4t/-0.3%, volatile:2t/-1.7% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

