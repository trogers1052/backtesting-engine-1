# TMDX (TransMedics) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 57.2 minutes
**Category:** Med-tech growth (MOMENTUM)

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

TMDX — Organ transplant tech — beta 1.1-2.05, high-growth (+37% YoY), ROE 54%. Med-tech growth (MOMENTUM).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 20 | 35.0% | +0.3% | -0.02 | 0.94 | -25.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 40 | 40.0% | +11.3% | 0.14 | 1.06 | -42.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 21 | 33.3% | +0.7% | -0.02 | 0.95 | -25.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 18 | 27.8% | -3.9% | -0.12 | 0.88 | -24.9% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 44 | 47.7% | +51.8% | 0.33 | 1.23 | -33.6% |
| Alt E: med_growth lean (3 rules, 10%/5%) | 20 | 35.0% | +0.3% | -0.02 | 0.94 | -25.0% |
| Alt F: Momentum (15%/8%) | 17 | 41.2% | +18.2% | 0.25 | 1.11 | -33.2% |
| Alt G: Tech-style (12%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt H: Wide momentum (20%/10%) | 13 | 46.2% | +68.8% | 0.78 | 1.59 | -32.5% |

**Best baseline selected for validation: Alt H: Wide momentum (20%/10%)**

---

## 2. Full Validation

### Alt H: Wide momentum (20%/10%)

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 20%
- **Min Confidence:** 0.5
- **Max Loss:** 10.0%
- **Cooldown:** 5 bars

**Performance:** Return=+68.8%, Trades=13, WR=46.2%, Sharpe=0.78, PF=1.59, DD=-32.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.81, Test Sharpe=0.28, Ratio=35% (need >=50%) |
| Bootstrap | FAIL | p=0.1158, Sharpe CI=[-1.63, 7.24], WR CI=[30.8%, 76.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.0%, Median equity=$1,785, Survival=99.9% |
| Regime | FAIL | bull:10t/+78.2%, bear:1t/+26.1%, chop:2t/-26.2% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 13 | 46.2% | +68.8% | 0.78 |
| WF tune: conf=0.55 | 13 | 46.2% | +68.8% | 0.78 |
| WF tune: conf=0.6 | 13 | 46.2% | +68.8% | 0.78 |
| WF tune: cooldown=3 | 13 | 46.2% | +68.8% | 0.78 |
| WF tune: + hc_mean_reversion | 13 | 46.2% | +68.8% | 0.78 |
| WF tune: + hc_pullback | 13 | 46.2% | +68.8% | 0.78 |
| BS tune: conf=0.4 | 13 | 46.2% | +68.8% | 0.78 |
| MC tune: ATR stops x2.0 | 13 | 46.2% | +68.8% | 0.78 |
| WF tune: cooldown=7 | 12 | 50.0% | +84.6% | 0.72 |
| WF tune: PT=15% | 14 | 50.0% | +48.9% | 0.62 |
| WF tune: ATR stops x2.5 | 14 | 42.9% | +49.2% | 0.55 |
| WF tune: PT=8% | 18 | 66.7% | +44.5% | 0.53 |
| MC tune: max_loss=3.0% | 18 | 27.8% | +64.1% | 0.45 |
| BS tune: healthcare rules (13) | 27 | 48.1% | +86.5% | 0.42 |
| WF tune: conf=0.65 | 12 | 41.7% | +37.1% | 0.40 |
| WF tune: PT=12% | 14 | 50.0% | +22.7% | 0.35 |
| WF tune: PT=7% | 18 | 66.7% | +26.5% | 0.32 |
| BS tune: full rules (10) | 21 | 38.1% | +42.8% | 0.31 |
| MC tune: max_loss=4.0% | 17 | 23.5% | +21.6% | 0.19 |
| WF tune: PT=6% | 18 | 66.7% | +15.8% | 0.18 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 20%
- **Min Confidence:** 0.45
- **Max Loss:** 10.0%
- **Cooldown:** 5 bars

**Performance:** Return=+68.8%, Trades=13, WR=46.2%, Sharpe=0.78, PF=1.59, DD=-32.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.81, Test Sharpe=0.28, Ratio=35% (need >=50%) |
| Bootstrap | FAIL | p=0.1158, Sharpe CI=[-1.63, 7.24], WR CI=[30.8%, 76.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.0%, Median equity=$1,785, Survival=99.9% |
| Regime | FAIL | bull:10t/+78.2%, bear:1t/+26.1%, chop:2t/-26.2% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 20%
- **Min Confidence:** 0.55
- **Max Loss:** 10.0%
- **Cooldown:** 5 bars

**Performance:** Return=+68.8%, Trades=13, WR=46.2%, Sharpe=0.78, PF=1.59, DD=-32.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.81, Test Sharpe=0.28, Ratio=35% (need >=50%) |
| Bootstrap | FAIL | p=0.1158, Sharpe CI=[-1.63, 7.24], WR CI=[30.8%, 76.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.0%, Median equity=$1,785, Survival=99.9% |
| Regime | FAIL | bull:10t/+78.2%, bear:1t/+26.1%, chop:2t/-26.2% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 20%
- **Min Confidence:** 0.6
- **Max Loss:** 10.0%
- **Cooldown:** 5 bars

**Performance:** Return=+68.8%, Trades=13, WR=46.2%, Sharpe=0.78, PF=1.59, DD=-32.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.81, Test Sharpe=0.28, Ratio=35% (need >=50%) |
| Bootstrap | FAIL | p=0.1158, Sharpe CI=[-1.63, 7.24], WR CI=[30.8%, 76.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.0%, Median equity=$1,785, Survival=99.9% |
| Regime | FAIL | bull:10t/+78.2%, bear:1t/+26.1%, chop:2t/-26.2% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt H: Wide momentum (20%/10%)** | FAIL | FAIL | FAIL | FAIL | **0.78** | **+68.8%** | 13 |
| WF tune: conf=0.45 | FAIL | FAIL | FAIL | FAIL | 0.78 | +68.8% | 13 |
| WF tune: conf=0.55 | FAIL | FAIL | FAIL | FAIL | 0.78 | +68.8% | 13 |
| WF tune: conf=0.6 | FAIL | FAIL | FAIL | FAIL | 0.78 | +68.8% | 13 |

---

## 5. Final Recommendation

**TMDX partially validates.** Best config: Alt H: Wide momentum (20%/10%) (0/4 gates).

### Alt H: Wide momentum (20%/10%)

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 20%
- **Min Confidence:** 0.5
- **Max Loss:** 10.0%
- **Cooldown:** 5 bars

**Performance:** Return=+68.8%, Trades=13, WR=46.2%, Sharpe=0.78, PF=1.59, DD=-32.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.81, Test Sharpe=0.28, Ratio=35% (need >=50%) |
| Bootstrap | FAIL | p=0.1158, Sharpe CI=[-1.63, 7.24], WR CI=[30.8%, 76.9%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.0%, Median equity=$1,785, Survival=99.9% |
| Regime | FAIL | bull:10t/+78.2%, bear:1t/+26.1%, chop:2t/-26.2% |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

