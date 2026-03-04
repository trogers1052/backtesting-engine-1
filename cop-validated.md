# COP (ConocoPhillips) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 30.6 minutes
**Category:** Large-cap E&P

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

COP — Independent E&P — Permian, Eagle Ford, Bakken, Alaska, global LNG. Large-cap E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 22.2% | -20.9% | -0.54 | 0.38 | -31.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 13 | 23.1% | -31.7% | -0.60 | 0.29 | -41.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 20.0% | -21.3% | -0.52 | 0.38 | -32.2% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 22.2% | -19.7% | -0.48 | 0.42 | -31.8% |
| Alt D: Energy rules (14 rules, 10%/5%) | 22 | 31.8% | -26.7% | -0.45 | 0.48 | -42.6% |
| Alt E: upstream sector rules (3 rules, 10%/5%) | 15 | 26.7% | -24.2% | -0.41 | 0.40 | -40.7% |
| Alt F: upstream rules wider stops (10%/6%) | 15 | 26.7% | -24.3% | -0.41 | 0.40 | -40.8% |

**Best baseline selected for validation: Alt F: upstream rules wider stops (10%/6%)**

---

## 2. Full Validation

### Alt F: upstream rules wider stops (10%/6%)

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 6.0%
- **Cooldown:** 5 bars

**Performance:** Return=-24.3%, Trades=15, WR=26.7%, Sharpe=-0.41, PF=0.40, DD=-40.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.43, Test Sharpe=0.87, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7946, Sharpe CI=[-6.85, 2.08], WR CI=[13.3%, 60.0%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-43.9%, Median equity=$741, Survival=99.9% |
| Regime | **PASS** | bull:11t/-1.5%, chop:2t/-3.0%, volatile:2t/-20.6% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| MC tune: max_loss=3.0% | 17 | 23.5% | -10.7% | -0.27 |
| WF tune: PT=8% | 17 | 35.3% | -20.0% | -0.31 |
| MC tune: max_loss=4.0% | 17 | 23.5% | -17.5% | -0.36 |
| MC tune: ATR stops x2.0 | 15 | 26.7% | -22.8% | -0.40 |
| WF tune: conf=0.65 | 12 | 25.0% | -24.7% | -0.41 |
| WF tune: conf=0.45 | 15 | 26.7% | -24.3% | -0.41 |
| WF tune: conf=0.55 | 15 | 26.7% | -24.3% | -0.41 |
| WF tune: conf=0.6 | 15 | 26.7% | -24.3% | -0.41 |
| BS tune: conf=0.4 | 15 | 26.7% | -24.3% | -0.41 |
| BS tune: + energy_mean_reversion | 15 | 26.7% | -24.3% | -0.41 |
| WF tune: ATR stops x2.5 | 15 | 26.7% | -23.6% | -0.41 |
| WF tune: cooldown=3 | 16 | 31.2% | -25.0% | -0.44 |
| WF tune: PT=12% | 13 | 23.1% | -25.6% | -0.46 |
| WF tune: PT=15% | 13 | 23.1% | -26.3% | -0.49 |
| BS tune: energy rules (14) | 21 | 33.3% | -33.0% | -0.54 |
| WF tune: cooldown=7 | 13 | 23.1% | -29.5% | -0.55 |
| BS tune: full rules (10) | 13 | 23.1% | -33.8% | -0.62 |
| MC tune: max_loss=4.0% [multi-TF] | 37 | 43.2% | -24.1% | -0.63 |

### Full Validation of Top Candidates

### MC tune: max_loss=3.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=-10.7%, Trades=17, WR=23.5%, Sharpe=-0.27, PF=0.59, DD=-29.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-2.03, Test Sharpe=0.87, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6098, Sharpe CI=[-5.72, 2.82], WR CI=[11.8%, 52.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.5%, Median equity=$909, Survival=100.0% |
| Regime | FAIL | bull:13t/+3.8%, chop:2t/-3.0%, volatile:2t/-7.0% |

**Result: 1/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 6.0%
- **Cooldown:** 5 bars

**Performance:** Return=-20.0%, Trades=17, WR=35.3%, Sharpe=-0.31, PF=0.57, DD=-40.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.36, Test Sharpe=1.09, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7157, Sharpe CI=[-5.21, 2.40], WR CI=[17.6%, 64.7%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-42.0%, Median equity=$797, Survival=100.0% |
| Regime | FAIL | bull:13t/+2.5%, chop:2t/+0.3%, volatile:2t/-20.6% |

**Result: 0/4 gates passed**

---

### MC tune: max_loss=4.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-17.5%, Trades=17, WR=23.5%, Sharpe=-0.36, PF=0.49, DD=-35.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.83, Test Sharpe=0.87, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7229, Sharpe CI=[-6.76, 2.33], WR CI=[11.8%, 52.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.8%, Median equity=$828, Survival=100.0% |
| Regime | **PASS** | bull:13t/-4.7%, chop:2t/-3.0%, volatile:2t/-7.5% |

**Result: 2/4 gates passed**

---

### MC tune: max_loss=4.0% [multi-TF]

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-24.1%, Trades=37, WR=43.2%, Sharpe=-0.63, PF=0.42, DD=-39.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.92, Test Sharpe=0.13, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8647, Sharpe CI=[-4.23, 0.98], WR CI=[32.4%, 64.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.2%, Median equity=$795, Survival=100.0% |
| Regime | **PASS** | bull:31t/-4.5%, bear:2t/-7.7%, chop:3t/-4.6%, volatile:1t/-4.1% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **MC tune: max_loss=4.0%** | FAIL | FAIL | **PASS** | **PASS** | **-0.36** | **-17.5%** | 17 |
| MC tune: max_loss=4.0% [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | -0.63 | -24.1% | 37 |
| MC tune: max_loss=3.0% | FAIL | FAIL | **PASS** | FAIL | -0.27 | -10.7% | 17 |
| Alt F: upstream rules wider stops (10%/6%) | FAIL | FAIL | FAIL | **PASS** | -0.41 | -24.3% | 15 |
| WF tune: PT=8% | FAIL | FAIL | FAIL | FAIL | -0.31 | -20.0% | 17 |

---

## 5. Final Recommendation

**COP partially validates.** Best config: MC tune: max_loss=4.0% (2/4 gates).

### MC tune: max_loss=4.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-17.5%, Trades=17, WR=23.5%, Sharpe=-0.36, PF=0.49, DD=-35.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.83, Test Sharpe=0.87, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7229, Sharpe CI=[-6.76, 2.33], WR CI=[11.8%, 52.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.8%, Median equity=$828, Survival=100.0% |
| Regime | **PASS** | bull:13t/-4.7%, chop:2t/-3.0%, volatile:2t/-7.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

