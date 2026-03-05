# BEP (Brookfield Renewable Partners) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 76.6 minutes
**Category:** Clean energy yieldco

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

BEP — Pure-play global renewable energy (hydro/wind/solar) — yieldco, 5%+ dividend. Clean energy yieldco.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 22.2% | -12.5% | -0.33 | 0.59 | -24.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 25.0% | -20.2% | -0.22 | 0.49 | -41.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 25.0% | -22.8% | -0.42 | 0.39 | -33.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 22.2% | -13.2% | -0.36 | 0.57 | -24.9% |
| Alt D: Utility rules (13 rules, 10%/5%) | 23 | 34.8% | -18.8% | -0.25 | 0.64 | -43.3% |
| Alt E: yieldco lean (4 rules, 10%/5%) | 17 | 35.3% | -18.0% | -0.37 | 0.63 | -34.1% |
| Alt F: Yieldco reversion (8%/5%, conf=0.50) | 16 | 56.2% | +24.1% | 0.52 | 1.70 | -15.9% |
| Alt G: Yieldco + midstream rule (10%/5%) | 16 | 37.5% | -12.0% | -0.22 | 0.73 | -34.3% |

**Best baseline selected for validation: Alt F: Yieldco reversion (8%/5%, conf=0.50)**

---

## 2. Full Validation

### Alt F: Yieldco reversion (8%/5%, conf=0.50)

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+24.1%, Trades=16, WR=56.2%, Sharpe=0.52, PF=1.70, DD=-15.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.54, Test Sharpe=1.12, Ratio=206% (need >=50%) |
| Bootstrap | FAIL | p=0.1280, Sharpe CI=[-1.54, 6.66], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.3%, Median equity=$1,295, Survival=100.0% |
| Regime | FAIL | bull:15t/+39.4%, chop:1t/-10.4% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: cooldown=3 | 17 | 58.8% | +44.1% | 1.03 |
| BS tune: conf=0.55 | 16 | 56.2% | +24.1% | 0.52 |
| BS tune: cooldown=7 | 15 | 53.3% | +23.0% | 0.36 |
| BS tune: yieldco rules | 17 | 52.9% | +15.4% | 0.34 |
| Regime tune: + utility_rate_reversion | 17 | 52.9% | +15.4% | 0.34 |
| Regime tune: conf=0.65 | 14 | 50.0% | +12.2% | 0.24 |
| BS tune: conf=0.45 | 16 | 50.0% | +10.8% | 0.18 |
| BS tune: conf=0.4 | 16 | 50.0% | +4.6% | -0.04 |
| Regime tune: tighter stop 4% | 19 | 42.1% | -1.3% | -0.10 |
| Regime tune: PT=15% | 16 | 31.2% | -7.3% | -0.17 |
| Regime tune: PT=12% | 16 | 37.5% | -11.0% | -0.21 |
| BS tune: full rules (10) | 18 | 27.8% | -23.6% | -0.29 |
| BS tune: utility rules (13) | 25 | 36.0% | -22.3% | -0.32 |
| BS tune: cooldown=3 [multi-TF] | 36 | 50.0% | -9.4% | -0.32 |

### Full Validation of Top Candidates

### BS tune: cooldown=3

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+44.1%, Trades=17, WR=58.8%, Sharpe=1.03, PF=2.34, DD=-15.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.75, Test Sharpe=1.14, Ratio=151% (need >=50%) |
| Bootstrap | FAIL | p=0.0402, Sharpe CI=[-0.35, 7.60], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.3%, Median equity=$1,521, Survival=100.0% |
| Regime | FAIL | bull:16t/+56.1%, chop:1t/-10.4% |

**Result: 2/4 gates passed**

---

### BS tune: conf=0.55

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+24.1%, Trades=16, WR=56.2%, Sharpe=0.52, PF=1.70, DD=-15.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.54, Test Sharpe=1.12, Ratio=206% (need >=50%) |
| Bootstrap | FAIL | p=0.1280, Sharpe CI=[-1.54, 6.66], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.3%, Median equity=$1,295, Survival=100.0% |
| Regime | FAIL | bull:15t/+39.4%, chop:1t/-10.4% |

**Result: 2/4 gates passed**

---

### BS tune: cooldown=7

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+23.0%, Trades=15, WR=53.3%, Sharpe=0.36, PF=1.72, DD=-17.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.72, Test Sharpe=0.61, Ratio=84% (need >=50%) |
| Bootstrap | FAIL | p=0.1341, Sharpe CI=[-1.54, 6.99], WR CI=[26.7%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,274, Survival=100.0% |
| Regime | FAIL | bull:14t/+37.6%, chop:1t/-10.4% |

**Result: 2/4 gates passed**

---

### BS tune: cooldown=3 [multi-TF]

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.4%, Trades=36, WR=50.0%, Sharpe=-0.32, PF=0.84, DD=-26.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.14, Test Sharpe=0.73, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5256, Sharpe CI=[-2.71, 2.24], WR CI=[33.3%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.4%, Median equity=$966, Survival=100.0% |
| Regime | FAIL | bull:30t/-12.3%, chop:5t/+9.5%, volatile:1t/+2.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: cooldown=3** | **PASS** | FAIL | **PASS** | FAIL | **1.03** | **+44.1%** | 17 |
| Alt F: Yieldco reversion (8%/5%, conf=0.50) | **PASS** | FAIL | **PASS** | FAIL | 0.52 | +24.1% | 16 |
| BS tune: conf=0.55 | **PASS** | FAIL | **PASS** | FAIL | 0.52 | +24.1% | 16 |
| BS tune: cooldown=7 | **PASS** | FAIL | **PASS** | FAIL | 0.36 | +23.0% | 15 |
| BS tune: cooldown=3 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.32 | -9.4% | 36 |

---

## 5. Final Recommendation

**BEP partially validates.** Best config: BS tune: cooldown=3 (2/4 gates).

### BS tune: cooldown=3

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+44.1%, Trades=17, WR=58.8%, Sharpe=1.03, PF=2.34, DD=-15.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.75, Test Sharpe=1.14, Ratio=151% (need >=50%) |
| Bootstrap | FAIL | p=0.0402, Sharpe CI=[-0.35, 7.60], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.3%, Median equity=$1,521, Survival=100.0% |
| Regime | FAIL | bull:16t/+56.1%, chop:1t/-10.4% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

