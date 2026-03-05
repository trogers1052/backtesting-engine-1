# COST (Costco) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 32.2 minutes
**Category:** Retail growth (MOMENTUM)

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

COST — Warehouse retail — beta ~1.0, P/E ~50, membership model, MOMENTUM stock. Retail growth (MOMENTUM).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 50.0% | +13.9% | 0.18 | 1.53 | -14.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 15 | 60.0% | +43.3% | 0.54 | 2.56 | -9.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 40.0% | -0.7% | -0.14 | 0.97 | -14.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 10 | 50.0% | +21.2% | 0.27 | 1.86 | -13.9% |
| Alt D: Staples rules (13 rules, 10%/5%) | 23 | 52.2% | +42.3% | 0.50 | 2.16 | -11.9% |
| Alt E: retail_growth lean (3 rules, 10%/5%) | 11 | 54.5% | +23.4% | 0.39 | 1.97 | -10.6% |
| Alt F: COST momentum (15%/8%) | 10 | 60.0% | +43.7% | 0.55 | 2.87 | -12.0% |
| Alt G: COST tech rules (12%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt H: COST wide (20%/10%) | 8 | 62.5% | +46.1% | 0.58 | 3.55 | -8.6% |

**Best baseline selected for validation: Alt F: COST momentum (15%/8%)**

---

## 2. Full Validation

### Alt F: COST momentum (15%/8%)

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+43.7%, Trades=10, WR=60.0%, Sharpe=0.55, PF=2.87, DD=-12.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.63, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0313, Sharpe CI=[-0.23, 12.83], WR CI=[40.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.7%, Median equity=$1,841, Survival=100.0% |
| Regime | FAIL | bull:7t/+69.9%, chop:1t/-8.1%, volatile:2t/+7.0% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: ATR stops x2.5 | 11 | 54.5% | +46.8% | 0.69 |
| WF tune: conf=0.65 | 9 | 66.7% | +48.6% | 0.68 |
| WF tune: PT=8% | 13 | 69.2% | +38.6% | 0.64 |
| WF tune: PT=7% | 16 | 75.0% | +50.6% | 0.63 |
| Regime tune: tighter stop 4% | 12 | 41.7% | +30.2% | 0.59 |
| WF tune: PT=6% | 17 | 76.5% | +43.2% | 0.57 |
| WF tune: cooldown=7 | 10 | 60.0% | +43.5% | 0.55 |
| WF tune: conf=0.45 | 10 | 60.0% | +43.7% | 0.55 |
| WF tune: conf=0.55 | 10 | 60.0% | +43.7% | 0.55 |
| WF tune: conf=0.6 | 10 | 60.0% | +43.7% | 0.55 |
| WF tune: + staples_mean_reversion | 10 | 60.0% | +43.7% | 0.55 |
| WF tune: + staples_pullback | 10 | 60.0% | +43.7% | 0.55 |
| BS tune: conf=0.4 | 10 | 60.0% | +43.7% | 0.55 |
| BS tune: full rules (10) | 10 | 60.0% | +41.2% | 0.51 |
| WF tune: PT=12% | 10 | 60.0% | +32.4% | 0.48 |
| BS tune: staples rules (13) | 17 | 52.9% | +39.4% | 0.47 |
| WF tune: ATR stops x2.5 [multi-TF] | 11 | 45.5% | +31.9% | 0.48 |

### Full Validation of Top Candidates

### WF tune: ATR stops x2.5

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+46.8%, Trades=11, WR=54.5%, Sharpe=0.69, PF=3.14, DD=-7.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.69, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | **PASS** | p=0.0158, Sharpe CI=[0.39, 12.00], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.6%, Median equity=$1,923, Survival=100.0% |
| Regime | FAIL | bull:8t/+69.5%, chop:1t/-6.2%, volatile:2t/+9.5% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.65
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+48.6%, Trades=9, WR=66.7%, Sharpe=0.68, PF=3.69, DD=-11.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.88, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[1.13, 282.26], WR CI=[44.4%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-16.3%, Median equity=$2,022, Survival=100.0% |
| Regime | FAIL | bull:7t/+69.9%, chop:1t/-8.1%, volatile:1t/+15.9% |

**Result: 2/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+38.6%, Trades=13, WR=69.2%, Sharpe=0.64, PF=3.15, DD=-9.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.41, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | **PASS** | p=0.0156, Sharpe CI=[0.43, 22.08], WR CI=[53.8%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.7%, Median equity=$1,689, Survival=100.0% |
| Regime | FAIL | bull:10t/+57.8%, chop:1t/-1.6%, volatile:2t/+0.1% |

**Result: 2/4 gates passed**

---

### WF tune: ATR stops x2.5 [multi-TF]

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+31.9%, Trades=11, WR=45.5%, Sharpe=0.48, PF=2.45, DD=-9.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.66, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0503, Sharpe CI=[-0.85, 9.28], WR CI=[27.3%, 81.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.4%, Median equity=$1,600, Survival=100.0% |
| Regime | FAIL | bull:9t/+65.8%, bear:2t/-12.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: ATR stops x2.5** | FAIL | **PASS** | **PASS** | FAIL | **0.69** | **+46.8%** | 11 |
| WF tune: conf=0.65 | FAIL | **PASS** | **PASS** | FAIL | 0.68 | +48.6% | 9 |
| WF tune: PT=8% | FAIL | **PASS** | **PASS** | FAIL | 0.64 | +38.6% | 13 |
| Alt F: COST momentum (15%/8%) | FAIL | FAIL | **PASS** | FAIL | 0.55 | +43.7% | 10 |
| WF tune: ATR stops x2.5 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | 0.48 | +31.9% | 11 |

---

## 5. Final Recommendation

**COST partially validates.** Best config: WF tune: ATR stops x2.5 (2/4 gates).

### WF tune: ATR stops x2.5

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+46.8%, Trades=11, WR=54.5%, Sharpe=0.69, PF=3.14, DD=-7.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.69, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | **PASS** | p=0.0158, Sharpe CI=[0.39, 12.00], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.6%, Median equity=$1,923, Survival=100.0% |
| Regime | FAIL | bull:8t/+69.5%, chop:1t/-6.2%, volatile:2t/+9.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

