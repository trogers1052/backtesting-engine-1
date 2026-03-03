# EQNR (Equinor) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 18.7 minutes
**Category:** Large-cap international oil

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

EQNR — Norwegian state oil company — North Sea, offshore wind, LNG. Large-cap international oil.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 6 | 16.7% | -17.5% | -0.58 | 0.02 | -26.5% |
| Alt A: Full general rules (10 rules, 10%/5%) | 8 | 12.5% | -26.5% | -0.98 | 0.04 | -38.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 6 | 16.7% | -15.8% | -0.55 | 0.02 | -25.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 6 | 16.7% | -17.5% | -0.58 | 0.02 | -26.5% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 8 | 12.5% | -26.5% | -0.98 | 0.04 | -38.1% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-17.5%, Trades=6, WR=16.7%, Sharpe=-0.58, PF=0.02, DD=-26.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.90, Test Sharpe=-0.23, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8700, Sharpe CI=[-14.21, 2.08], WR CI=[0.0%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.4%, Median equity=$824, Survival=100.0% |
| Regime | FAIL | bull:4t/-5.8%, bear:1t/+0.8%, chop:1t/-12.8% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 6 | 16.7% | -15.8% | -0.55 |
| WF tune: PT=8% | 8 | 25.0% | -14.9% | -0.56 |
| WF tune: conf=0.65 | 6 | 16.7% | -17.0% | -0.57 |
| WF tune: PT=12% | 6 | 16.7% | -17.5% | -0.58 |
| WF tune: PT=15% | 6 | 16.7% | -17.5% | -0.58 |
| WF tune: conf=0.45 | 6 | 16.7% | -17.5% | -0.58 |
| WF tune: conf=0.55 | 6 | 16.7% | -17.5% | -0.58 |
| WF tune: conf=0.6 | 6 | 16.7% | -17.5% | -0.58 |
| WF tune: cooldown=3 | 6 | 16.7% | -17.5% | -0.58 |
| WF tune: cooldown=7 | 6 | 16.7% | -17.5% | -0.58 |
| BS tune: conf=0.4 | 6 | 16.7% | -17.5% | -0.58 |
| BS tune: + volume_breakout | 6 | 16.7% | -17.5% | -0.58 |
| BS tune: + commodity_breakout | 6 | 16.7% | -17.5% | -0.58 |
| Regime tune: + dollar_weakness | 6 | 16.7% | -17.5% | -0.58 |
| BS tune: full rules (10) | 8 | 12.5% | -26.5% | -0.98 |
| BS tune: energy rules (12) | 8 | 12.5% | -26.5% | -0.98 |
| WF tune: conf=0.65 [multi-TF] | 4 | 25.0% | -7.0% | -0.34 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-15.8%, Trades=6, WR=16.7%, Sharpe=-0.55, PF=0.02, DD=-25.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.90, Test Sharpe=-0.15, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8496, Sharpe CI=[-12.19, 2.50], WR CI=[0.0%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.8%, Median equity=$842, Survival=100.0% |
| Regime | FAIL | bull:4t/-3.8%, bear:1t/+0.8%, chop:1t/-12.8% |

**Result: 1/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-14.9%, Trades=8, WR=25.0%, Sharpe=-0.56, PF=0.27, DD=-24.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.98, Test Sharpe=-0.23, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7772, Sharpe CI=[-13.55, 3.48], WR CI=[0.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.2%, Median equity=$853, Survival=100.0% |
| Regime | FAIL | bull:6t/-1.8%, bear:1t/+0.8%, chop:1t/-12.8% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-17.0%, Trades=6, WR=16.7%, Sharpe=-0.57, PF=0.02, DD=-25.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.87, Test Sharpe=-0.23, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8658, Sharpe CI=[-14.87, 2.15], WR CI=[0.0%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.0%, Median equity=$829, Survival=100.0% |
| Regime | **PASS** | bull:4t/-5.8%, bear:2t/-11.5% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.65 [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-7.0%, Trades=4, WR=25.0%, Sharpe=-0.34, PF=0.53, DD=-17.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.82, Test Sharpe=0.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7431, Sharpe CI=[-1123.00, 5.88], WR CI=[0.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.3%, Median equity=$932, Survival=100.0% |
| Regime | **PASS** | bull:3t/-1.0%, chop:1t/-5.2% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65 [multi-TF]** | FAIL | FAIL | **PASS** | **PASS** | **-0.34** | **-7.0%** | 4 |
| WF tune: conf=0.65 | FAIL | FAIL | **PASS** | **PASS** | -0.57 | -17.0% | 6 |
| Regime tune: tighter stop 4% | FAIL | FAIL | **PASS** | FAIL | -0.55 | -15.8% | 6 |
| WF tune: PT=8% | FAIL | FAIL | **PASS** | FAIL | -0.56 | -14.9% | 8 |
| Lean 3 rules baseline (10%/5%, conf=0.50) | FAIL | FAIL | **PASS** | FAIL | -0.58 | -17.5% | 6 |

---

## 5. Final Recommendation

**EQNR partially validates.** Best config: WF tune: conf=0.65 [multi-TF] (2/4 gates).

### WF tune: conf=0.65 [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-7.0%, Trades=4, WR=25.0%, Sharpe=-0.34, PF=0.53, DD=-17.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.82, Test Sharpe=0.03, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7431, Sharpe CI=[-1123.00, 5.88], WR CI=[0.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.3%, Median equity=$932, Survival=100.0% |
| Regime | **PASS** | bull:3t/-1.0%, chop:1t/-5.2% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

