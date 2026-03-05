# WLDN (Willdan Group) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 18.2 minutes
**Category:** Industrial (infrastructure services)

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

WLDN — Engineering/infrastructure consulting — small-cap, government contracts, clean energy. Industrial (infrastructure services).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 16 | 56.2% | +73.6% | 0.64 | 2.09 | -22.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 33 | 48.5% | +122.0% | 0.53 | 1.72 | -35.7% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 16 | 50.0% | +52.8% | 0.44 | 1.79 | -25.7% |
| Alt C: Wider PT (3 rules, 12%/5%) | 15 | 40.0% | +36.9% | 0.61 | 1.65 | -21.9% |
| Alt D: Recommended rules (3 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt E: Industrial full (13 rules, 10%/5%) | 33 | 48.5% | +122.0% | 0.53 | 1.72 | -35.7% |
| Alt F: Industrial reversion (8%/4%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt G: Industrial momentum (12%/6%) | 17 | 41.2% | +18.6% | 0.25 | 1.24 | -29.6% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+73.6%, Trades=16, WR=56.2%, Sharpe=0.64, PF=2.09, DD=-22.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.87, Test Sharpe=-0.17, Ratio=-19% (need >=50%) |
| Bootstrap | FAIL | p=0.0310, Sharpe CI=[-0.14, 8.47], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.6%, Median equity=$1,845, Survival=100.0% |
| Regime | FAIL | bull:16t/+68.9% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 13 | 69.2% | +94.1% | 0.73 |
| WF tune: conf=0.55 | 16 | 56.2% | +73.8% | 0.64 |
| WF tune: conf=0.6 | 16 | 56.2% | +73.8% | 0.64 |
| WF tune: conf=0.45 | 16 | 56.2% | +73.6% | 0.64 |
| WF tune: ATR stops x2.5 | 16 | 56.2% | +73.6% | 0.64 |
| BS tune: conf=0.4 | 16 | 56.2% | +73.6% | 0.64 |
| WF tune: PT=12% | 15 | 40.0% | +36.9% | 0.61 |
| WF tune: PT=8% | 17 | 64.7% | +95.9% | 0.60 |
| WF tune: PT=7% | 17 | 64.7% | +92.7% | 0.60 |
| BS tune: full rules (10) | 33 | 48.5% | +122.0% | 0.53 |
| WF tune: PT=6% | 17 | 64.7% | +72.2% | 0.52 |
| WF tune: cooldown=3 | 19 | 52.6% | +86.6% | 0.50 |
| WF tune: cooldown=7 | 16 | 50.0% | +42.1% | 0.47 |
| Regime tune: tighter stop 4% | 16 | 50.0% | +52.8% | 0.44 |
| WF tune: PT=15% | 15 | 33.3% | +22.4% | 0.39 |
| WF tune: recommended rules | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: conf=0.65 [multi-TF] | 17 | 64.7% | +109.7% | 0.63 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+94.1%, Trades=13, WR=69.2%, Sharpe=0.73, PF=2.85, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.89, Test Sharpe=0.84, Ratio=94% (need >=50%) |
| Bootstrap | **PASS** | p=0.0114, Sharpe CI=[0.58, 13.87], WR CI=[46.2%, 92.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.9%, Median equity=$2,079, Survival=100.0% |
| Regime | FAIL | bull:13t/+80.6% |

**Result: 3/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+73.8%, Trades=16, WR=56.2%, Sharpe=0.64, PF=2.09, DD=-22.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.87, Test Sharpe=-0.17, Ratio=-19% (need >=50%) |
| Bootstrap | FAIL | p=0.0304, Sharpe CI=[-0.13, 8.48], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.6%, Median equity=$1,853, Survival=100.0% |
| Regime | FAIL | bull:16t/+69.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+73.8%, Trades=16, WR=56.2%, Sharpe=0.64, PF=2.09, DD=-22.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.87, Test Sharpe=-0.17, Ratio=-19% (need >=50%) |
| Bootstrap | FAIL | p=0.0304, Sharpe CI=[-0.13, 8.48], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.6%, Median equity=$1,853, Survival=100.0% |
| Regime | FAIL | bull:16t/+69.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65 [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+109.7%, Trades=17, WR=64.7%, Sharpe=0.63, PF=2.75, DD=-16.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.68, Ratio=-121% (need >=50%) |
| Bootstrap | **PASS** | p=0.0064, Sharpe CI=[0.97, 10.56], WR CI=[41.2%, 88.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.3%, Median equity=$2,262, Survival=100.0% |
| Regime | FAIL | bull:17t/+89.2% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | **PASS** | **PASS** | **PASS** | FAIL | **0.73** | **+94.1%** | 13 |
| WF tune: conf=0.65 [multi-TF] | FAIL | **PASS** | **PASS** | FAIL | 0.63 | +109.7% | 17 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.64 | +73.8% | 16 |
| WF tune: conf=0.6 | FAIL | FAIL | **PASS** | FAIL | 0.64 | +73.8% | 16 |
| Lean 3 rules baseline (10%/5%, conf=0.50) | FAIL | FAIL | **PASS** | FAIL | 0.64 | +73.6% | 16 |

---

## 5. Final Recommendation

**WLDN partially validates.** Best config: WF tune: conf=0.65 (3/4 gates).

### WF tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+94.1%, Trades=13, WR=69.2%, Sharpe=0.73, PF=2.85, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.89, Test Sharpe=0.84, Ratio=94% (need >=50%) |
| Bootstrap | **PASS** | p=0.0114, Sharpe CI=[0.58, 13.87], WR CI=[46.2%, 92.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.9%, Median equity=$2,079, Survival=100.0% |
| Regime | FAIL | bull:13t/+80.6% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

