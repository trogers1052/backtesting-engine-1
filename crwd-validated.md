# CRWD (CrowdStrike) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 47.4 minutes
**Category:** Cybersecurity SaaS

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

CRWD — Cybersecurity endpoint protection — high beta, momentum stock. Cybersecurity SaaS.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 19 | 52.6% | +54.4% | 0.58 | 1.77 | -14.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 40 | 52.5% | +104.0% | 0.60 | 1.45 | -24.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 19 | 42.1% | +29.9% | 0.32 | 1.47 | -22.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 17 | 52.9% | +54.2% | 0.50 | 1.88 | -15.8% |
| Alt D: Tech rules (13 rules, 10%/5%) | 49 | 53.1% | +140.9% | 0.66 | 1.49 | -25.6% |
| Alt E: saas rules (3 rules, 10%/5%) | 27 | 55.6% | +100.6% | 0.74 | 1.89 | -19.7% |
| Alt F: SaaS momentum (12%/6%) | 23 | 60.9% | +146.8% | 0.70 | 2.29 | -19.5% |

**Best baseline selected for validation: Alt E: saas rules (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: saas rules (3 rules, 10%/5%)

- **Rules:** `tech_ema_pullback, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+100.6%, Trades=27, WR=55.6%, Sharpe=0.74, PF=1.89, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.65, Test Sharpe=-1.03, Ratio=-158% (need >=50%) |
| Bootstrap | FAIL | p=0.0307, Sharpe CI=[-0.12, 5.82], WR CI=[37.0%, 74.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.5%, Median equity=$2,265, Survival=100.0% |
| Regime | FAIL | bull:24t/+80.1%, bear:2t/+2.9%, chop:1t/+12.0% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=15% | 22 | 54.5% | +139.9% | 0.75 |
| WF tune: conf=0.45 | 27 | 55.6% | +100.6% | 0.74 |
| WF tune: conf=0.55 | 27 | 55.6% | +100.6% | 0.74 |
| WF tune: ATR stops x2.5 | 27 | 55.6% | +100.6% | 0.74 |
| WF tune: + tech_mean_reversion | 27 | 55.6% | +100.6% | 0.74 |
| BS tune: conf=0.4 | 27 | 55.6% | +100.6% | 0.74 |
| WF tune: PT=8% | 27 | 59.3% | +109.3% | 0.72 |
| WF tune: cooldown=3 | 33 | 51.5% | +84.7% | 0.68 |
| WF tune: conf=0.65 | 19 | 73.7% | +165.0% | 0.68 |
| WF tune: conf=0.6 | 26 | 53.8% | +85.5% | 0.66 |
| WF tune: cooldown=7 | 24 | 62.5% | +140.3% | 0.66 |
| BS tune: tech rules (13) | 49 | 53.1% | +140.9% | 0.66 |
| WF tune: PT=12% | 25 | 56.0% | +132.2% | 0.64 |
| BS tune: full rules (10) | 40 | 52.5% | +104.0% | 0.60 |
| Regime tune: tighter stop 4% | 34 | 47.1% | +54.9% | 0.49 |
| WF tune: PT=15% [multi-TF] | 38 | 44.7% | +4.6% | 0.01 |

### Full Validation of Top Candidates

### WF tune: PT=15%

- **Rules:** `tech_ema_pullback, tech_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+139.9%, Trades=22, WR=54.5%, Sharpe=0.75, PF=2.32, DD=-20.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-1.21, Ratio=-182% (need >=50%) |
| Bootstrap | **PASS** | p=0.0195, Sharpe CI=[0.19, 6.63], WR CI=[31.8%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.1%, Median equity=$2,591, Survival=100.0% |
| Regime | FAIL | bull:19t/+108.6%, bear:2t/+7.9%, volatile:1t/-5.8% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `tech_ema_pullback, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+100.6%, Trades=27, WR=55.6%, Sharpe=0.74, PF=1.89, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.65, Test Sharpe=-1.03, Ratio=-158% (need >=50%) |
| Bootstrap | FAIL | p=0.0307, Sharpe CI=[-0.12, 5.82], WR CI=[37.0%, 74.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.5%, Median equity=$2,265, Survival=100.0% |
| Regime | FAIL | bull:24t/+80.1%, bear:2t/+2.9%, chop:1t/+12.0% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `tech_ema_pullback, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+100.6%, Trades=27, WR=55.6%, Sharpe=0.74, PF=1.89, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.65, Test Sharpe=-1.03, Ratio=-158% (need >=50%) |
| Bootstrap | FAIL | p=0.0307, Sharpe CI=[-0.12, 5.82], WR CI=[37.0%, 74.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.5%, Median equity=$2,265, Survival=100.0% |
| Regime | FAIL | bull:24t/+80.1%, bear:2t/+2.9%, chop:1t/+12.0% |

**Result: 1/4 gates passed**

---

### WF tune: PT=15% [multi-TF]

- **Rules:** `tech_ema_pullback, tech_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+4.6%, Trades=38, WR=44.7%, Sharpe=0.01, PF=1.05, DD=-28.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.07, Test Sharpe=-1.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2708, Sharpe CI=[-1.85, 2.71], WR CI=[28.9%, 60.5%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-41.4%, Median equity=$1,214, Survival=100.0% |
| Regime | **PASS** | bull:33t/+7.2%, bear:1t/-5.0%, chop:3t/+12.4%, volatile:1t/+15.0% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=15%** | FAIL | **PASS** | **PASS** | FAIL | **0.75** | **+139.9%** | 22 |
| Alt E: saas rules (3 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.74 | +100.6% | 27 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.74 | +100.6% | 27 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.74 | +100.6% | 27 |
| WF tune: PT=15% [multi-TF] | FAIL | FAIL | FAIL | **PASS** | 0.01 | +4.6% | 38 |

---

## 5. Final Recommendation

**CRWD partially validates.** Best config: WF tune: PT=15% (2/4 gates).

### WF tune: PT=15%

- **Rules:** `tech_ema_pullback, tech_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+139.9%, Trades=22, WR=54.5%, Sharpe=0.75, PF=2.32, DD=-20.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-1.21, Ratio=-182% (need >=50%) |
| Bootstrap | **PASS** | p=0.0195, Sharpe CI=[0.19, 6.63], WR CI=[31.8%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.1%, Median equity=$2,591, Survival=100.0% |
| Regime | FAIL | bull:19t/+108.6%, bear:2t/+7.9%, volatile:1t/-5.8% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

