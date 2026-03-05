# SO (Southern Company) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 78.1 minutes
**Category:** Regulated utility

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

SO — Traditional regulated utility (electric + gas) — classic defensive, beta 0.45. Regulated utility.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 28.6% | -8.5% | -0.59 | 0.67 | -15.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 14 | 35.7% | -5.8% | -0.51 | 0.83 | -16.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 7 | 14.3% | -14.4% | -0.81 | 0.40 | -18.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 6 | 16.7% | -15.0% | -0.92 | 0.39 | -16.5% |
| Alt D: Utility rules (13 rules, 10%/5%) | 27 | 55.6% | +10.5% | 0.15 | 1.22 | -15.6% |
| Alt E: regulated lean (3 rules, 10%/5%) | 19 | 57.9% | +4.1% | -0.02 | 1.11 | -15.7% |
| Alt F: Regulated tight (7%/4%, conf=0.55, cooldown=7) | 20 | 55.0% | +0.0% | -0.19 | 1.00 | -14.1% |
| Alt G: Regulated moderate (8%/4%) | 25 | 56.0% | +11.8% | 0.19 | 1.24 | -16.2% |

**Best baseline selected for validation: Alt G: Regulated moderate (8%/4%)**

---

## 2. Full Validation

### Alt G: Regulated moderate (8%/4%)

- **Rules:** `utility_mean_reversion, utility_rate_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+11.8%, Trades=25, WR=56.0%, Sharpe=0.19, PF=1.24, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.08, Ratio=-14% (need >=50%) |
| Bootstrap | FAIL | p=0.2207, Sharpe CI=[-1.87, 4.02], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.4%, Median equity=$1,190, Survival=100.0% |
| Regime | FAIL | bull:18t/+20.6%, bear:3t/+0.9%, chop:3t/+4.6%, volatile:1t/-5.5% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 25 | 60.0% | +15.5% | 0.30 |
| WF tune: conf=0.45 | 25 | 56.0% | +11.8% | 0.19 |
| WF tune: conf=0.55 | 25 | 56.0% | +11.8% | 0.19 |
| WF tune: ATR stops x2.5 | 25 | 56.0% | +11.8% | 0.19 |
| WF tune: PT=7% | 26 | 57.7% | +12.1% | 0.18 |
| WF tune: conf=0.6 | 23 | 52.2% | +8.6% | 0.10 |
| BS tune: regulated rules | 22 | 54.5% | +6.7% | 0.04 |
| WF tune: PT=12% | 22 | 54.5% | +5.5% | -0.01 |
| WF tune: PT=15% | 20 | 55.0% | +3.2% | -0.04 |
| BS tune: utility rules (13) | 29 | 51.7% | -1.0% | -0.18 |
| WF tune: cooldown=3 | 24 | 54.2% | -1.2% | -0.18 |
| WF tune: cooldown=7 | 20 | 55.0% | -0.1% | -0.21 |
| WF tune: conf=0.65 | 15 | 53.3% | +4.0% | -0.24 |
| BS tune: full rules (10) | 16 | 31.2% | -12.2% | -0.74 |
| BS tune: conf=0.4 [multi-TF] | 36 | 41.7% | -20.0% | -0.63 |

### Full Validation of Top Candidates

### BS tune: conf=0.4

- **Rules:** `utility_mean_reversion, utility_rate_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.4
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+15.5%, Trades=25, WR=60.0%, Sharpe=0.30, PF=1.32, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.56, Test Sharpe=0.70, Ratio=125% (need >=50%) |
| Bootstrap | FAIL | p=0.1861, Sharpe CI=[-1.66, 4.27], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.6%, Median equity=$1,232, Survival=100.0% |
| Regime | FAIL | bull:18t/+24.2%, bear:3t/+0.9%, chop:3t/+4.6%, volatile:1t/-5.5% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `utility_mean_reversion, utility_rate_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.45
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+11.8%, Trades=25, WR=56.0%, Sharpe=0.19, PF=1.24, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.08, Ratio=-14% (need >=50%) |
| Bootstrap | FAIL | p=0.2207, Sharpe CI=[-1.87, 4.02], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.4%, Median equity=$1,190, Survival=100.0% |
| Regime | FAIL | bull:18t/+20.6%, bear:3t/+0.9%, chop:3t/+4.6%, volatile:1t/-5.5% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `utility_mean_reversion, utility_rate_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.55
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+11.8%, Trades=25, WR=56.0%, Sharpe=0.19, PF=1.24, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.08, Ratio=-14% (need >=50%) |
| Bootstrap | FAIL | p=0.2207, Sharpe CI=[-1.87, 4.02], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.4%, Median equity=$1,190, Survival=100.0% |
| Regime | FAIL | bull:18t/+20.6%, bear:3t/+0.9%, chop:3t/+4.6%, volatile:1t/-5.5% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4 [multi-TF]

- **Rules:** `utility_mean_reversion, utility_rate_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.4
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-20.0%, Trades=36, WR=41.7%, Sharpe=-0.63, PF=0.57, DD=-29.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.86, Test Sharpe=0.10, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7952, Sharpe CI=[-4.22, 1.26], WR CI=[30.6%, 63.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.5%, Median equity=$831, Survival=100.0% |
| Regime | FAIL | bull:27t/-11.3%, bear:4t/-3.8%, chop:4t/+3.2%, volatile:1t/-4.3% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.4** | **PASS** | FAIL | **PASS** | FAIL | **0.30** | **+15.5%** | 25 |
| Alt G: Regulated moderate (8%/4%) | FAIL | FAIL | **PASS** | FAIL | 0.19 | +11.8% | 25 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.19 | +11.8% | 25 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.19 | +11.8% | 25 |
| BS tune: conf=0.4 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.63 | -20.0% | 36 |

---

## 5. Final Recommendation

**SO partially validates.** Best config: BS tune: conf=0.4 (2/4 gates).

### BS tune: conf=0.4

- **Rules:** `utility_mean_reversion, utility_rate_reversion, utility_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.4
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+15.5%, Trades=25, WR=60.0%, Sharpe=0.30, PF=1.32, DD=-16.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.56, Test Sharpe=0.70, Ratio=125% (need >=50%) |
| Bootstrap | FAIL | p=0.1861, Sharpe CI=[-1.66, 4.27], WR CI=[40.0%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.6%, Median equity=$1,232, Survival=100.0% |
| Regime | FAIL | bull:18t/+24.2%, bear:3t/+0.9%, chop:3t/+4.6%, volatile:1t/-5.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

