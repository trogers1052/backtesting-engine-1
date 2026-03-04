# EPD (Enterprise Products Partners) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 29.7 minutes
**Category:** Large-cap midstream

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

EPD — Midstream MLP — pipelines, storage, NGL processing. Large-cap midstream.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 8 | 50.0% | +7.0% | 0.05 | 1.30 | -15.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 10 | 50.0% | +11.4% | 0.16 | 1.47 | -15.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 36.4% | -1.0% | -0.15 | 0.95 | -15.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 42.9% | +3.9% | -0.01 | 1.17 | -15.4% |
| Alt D: Energy rules (14 rules, 10%/5%) | 15 | 46.7% | +12.7% | 0.18 | 1.45 | -14.9% |
| Alt E: midstream sector rules (3 rules, 10%/5%) | 13 | 53.8% | +19.7% | 0.35 | 1.73 | -14.0% |
| Alt F: Midstream 8% PT / 4% stop | 14 | 50.0% | +8.7% | 0.12 | 1.19 | -13.8% |

**Best baseline selected for validation: Alt E: midstream sector rules (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: midstream sector rules (3 rules, 10%/5%)

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+19.7%, Trades=13, WR=53.8%, Sharpe=0.35, PF=1.73, DD=-14.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.42, Test Sharpe=0.45, Ratio=107% (need >=50%) |
| Bootstrap | FAIL | p=0.1675, Sharpe CI=[-2.06, 7.09], WR CI=[30.8%, 76.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.3%, Median equity=$1,240, Survival=100.0% |
| Regime | FAIL | bull:11t/+24.5%, bear:2t/+0.3% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 13 | 53.8% | +21.0% | 0.39 |
| BS tune: conf=0.45 | 13 | 53.8% | +19.7% | 0.35 |
| BS tune: conf=0.55 | 13 | 53.8% | +19.7% | 0.35 |
| BS tune: + energy_momentum | 13 | 53.8% | +19.7% | 0.35 |
| BS tune: + energy_mean_reversion | 13 | 53.8% | +19.7% | 0.35 |
| Regime tune: tighter stop 4% | 13 | 46.2% | +16.8% | 0.34 |
| Regime tune: PT=12% | 12 | 50.0% | +19.9% | 0.31 |
| Regime tune: conf=0.65 | 9 | 55.6% | +16.4% | 0.27 |
| BS tune: cooldown=7 | 12 | 50.0% | +16.1% | 0.25 |
| BS tune: cooldown=3 | 13 | 46.2% | +13.7% | 0.22 |
| Regime tune: PT=15% | 11 | 45.5% | +14.1% | 0.22 |
| BS tune: energy rules (14) | 15 | 46.7% | +12.7% | 0.18 |
| BS tune: full rules (10) | 10 | 50.0% | +11.4% | 0.16 |
| BS tune: conf=0.4 [multi-TF] | 26 | 42.3% | +10.2% | 0.14 |

### Full Validation of Top Candidates

### BS tune: conf=0.4

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+21.0%, Trades=13, WR=53.8%, Sharpe=0.39, PF=1.81, DD=-13.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.42, Test Sharpe=0.58, Ratio=137% (need >=50%) |
| Bootstrap | FAIL | p=0.1559, Sharpe CI=[-1.92, 7.27], WR CI=[30.8%, 84.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.6%, Median equity=$1,255, Survival=100.0% |
| Regime | FAIL | bull:11t/+25.7%, bear:2t/+0.3% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+19.7%, Trades=13, WR=53.8%, Sharpe=0.35, PF=1.73, DD=-14.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.42, Test Sharpe=0.45, Ratio=107% (need >=50%) |
| Bootstrap | FAIL | p=0.1675, Sharpe CI=[-2.06, 7.09], WR CI=[30.8%, 76.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.3%, Median equity=$1,240, Survival=100.0% |
| Regime | FAIL | bull:11t/+24.5%, bear:2t/+0.3% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.55

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+19.7%, Trades=13, WR=53.8%, Sharpe=0.35, PF=1.73, DD=-14.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.42, Test Sharpe=0.45, Ratio=107% (need >=50%) |
| Bootstrap | FAIL | p=0.1675, Sharpe CI=[-2.06, 7.09], WR CI=[30.8%, 76.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.3%, Median equity=$1,240, Survival=100.0% |
| Regime | FAIL | bull:11t/+24.5%, bear:2t/+0.3% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+10.2%, Trades=26, WR=42.3%, Sharpe=0.14, PF=1.20, DD=-17.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.32, Test Sharpe=0.20, Ratio=61% (need >=50%) |
| Bootstrap | FAIL | p=0.2421, Sharpe CI=[-2.09, 3.54], WR CI=[30.8%, 69.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.8%, Median equity=$1,164, Survival=100.0% |
| Regime | FAIL | bull:21t/+15.6%, bear:1t/-1.1%, chop:4t/+3.4% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | **PASS** | FAIL | **PASS** | FAIL | **0.39** | **+21.0%** | 13 |
| Alt E: midstream sector rules (3 rules, 10%/5%) | **PASS** | FAIL | **PASS** | FAIL | 0.35 | +19.7% | 13 |
| BS tune: conf=0.45 | **PASS** | FAIL | **PASS** | FAIL | 0.35 | +19.7% | 13 |
| BS tune: conf=0.55 | **PASS** | FAIL | **PASS** | FAIL | 0.35 | +19.7% | 13 |
| BS tune: conf=0.4 [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | 0.14 | +10.2% | 26 |

---

## 5. Final Recommendation

**EPD partially validates.** Best config: BS tune: conf=0.4 (2/4 gates).

### BS tune: conf=0.4

- **Rules:** `midstream_yield_reversion, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+21.0%, Trades=13, WR=53.8%, Sharpe=0.39, PF=1.81, DD=-13.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.42, Test Sharpe=0.58, Ratio=137% (need >=50%) |
| Bootstrap | FAIL | p=0.1559, Sharpe CI=[-1.92, 7.27], WR CI=[30.8%, 84.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.6%, Median equity=$1,255, Survival=100.0% |
| Regime | FAIL | bull:11t/+25.7%, bear:2t/+0.3% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

