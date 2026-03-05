# AWK (American Water Works) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 63.2 minutes
**Category:** Water utility

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

AWK — Largest US water utility — negative beta, ultra-defensive, 17yr dividend streak. Water utility.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 28.6% | -9.6% | -0.58 | 0.65 | -18.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 11 | 18.2% | -24.8% | -0.89 | 0.27 | -27.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 25.0% | -12.7% | -0.50 | 0.58 | -21.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 28.6% | -4.2% | -0.25 | 0.84 | -17.5% |
| Alt D: Utility rules (13 rules, 10%/5%) | 19 | 42.1% | -24.7% | -0.65 | 0.44 | -34.9% |
| Alt E: water lean (3 rules, 10%/5%) | 14 | 42.9% | -20.6% | -0.55 | 0.47 | -30.9% |
| Alt F: Water ultra-tight (6%/3%, conf=0.55) | 17 | 47.1% | -0.5% | -0.13 | 0.98 | -17.8% |
| Alt G: Water moderate (8%/4%) | 14 | 42.9% | -15.4% | -0.50 | 0.52 | -26.6% |

**Best baseline selected for validation: Alt F: Water ultra-tight (6%/3%, conf=0.55)**

---

## 2. Full Validation

### Alt F: Water ultra-tight (6%/3%, conf=0.55)

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=-0.5%, Trades=17, WR=47.1%, Sharpe=-0.13, PF=0.98, DD=-17.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.14, Test Sharpe=-0.88, Ratio=-639% (need >=50%) |
| Bootstrap | FAIL | p=0.4526, Sharpe CI=[-3.88, 3.71], WR CI=[23.5%, 70.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,018, Survival=100.0% |
| Regime | FAIL | bull:10t/-5.1%, bear:3t/+12.9%, chop:3t/-0.2%, crisis:1t/-4.3% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 16 | 43.8% | +0.3% | -0.10 |
| WF tune: conf=0.45 | 17 | 47.1% | -0.5% | -0.13 |
| WF tune: ATR stops x2.5 | 17 | 47.1% | -0.5% | -0.13 |
| BS tune: conf=0.4 | 17 | 47.1% | -0.5% | -0.13 |
| WF tune: + utility_rate_reversion | 18 | 44.4% | -0.7% | -0.19 |
| WF tune: PT=15% | 16 | 37.5% | -7.2% | -0.33 |
| WF tune: PT=12% | 16 | 37.5% | -7.8% | -0.35 |
| WF tune: cooldown=3 | 22 | 40.9% | -19.2% | -0.43 |
| WF tune: PT=7% | 16 | 37.5% | -11.1% | -0.48 |
| WF tune: PT=8% | 16 | 37.5% | -11.1% | -0.48 |
| WF tune: conf=0.65 | 10 | 30.0% | -1.8% | -0.63 |
| BS tune: utility rules (13) | 28 | 35.7% | -28.1% | -0.72 |
| BS tune: full rules (10) | 15 | 20.0% | -24.0% | -1.23 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.6
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+0.3%, Trades=16, WR=43.8%, Sharpe=-0.10, PF=1.01, DD=-17.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.25, Test Sharpe=-1.03, Ratio=-404% (need >=50%) |
| Bootstrap | FAIL | p=0.4414, Sharpe CI=[-3.89, 3.90], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.0%, Median equity=$1,026, Survival=100.0% |
| Regime | FAIL | bull:9t/-3.7%, bear:3t/+12.1%, chop:3t/-0.2%, crisis:1t/-4.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.45
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=-0.5%, Trades=17, WR=47.1%, Sharpe=-0.13, PF=0.98, DD=-17.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.14, Test Sharpe=-0.88, Ratio=-639% (need >=50%) |
| Bootstrap | FAIL | p=0.4526, Sharpe CI=[-3.88, 3.71], WR CI=[23.5%, 70.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,018, Survival=100.0% |
| Regime | FAIL | bull:10t/-5.1%, bear:3t/+12.9%, chop:3t/-0.2%, crisis:1t/-4.3% |

**Result: 1/4 gates passed**

---

### WF tune: ATR stops x2.5

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=-0.5%, Trades=17, WR=47.1%, Sharpe=-0.13, PF=0.98, DD=-17.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.14, Test Sharpe=-0.88, Ratio=-639% (need >=50%) |
| Bootstrap | FAIL | p=0.4526, Sharpe CI=[-3.88, 3.71], WR CI=[23.5%, 70.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,018, Survival=100.0% |
| Regime | FAIL | bull:10t/-5.1%, bear:3t/+12.9%, chop:3t/-0.2%, crisis:1t/-4.3% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | FAIL | **PASS** | FAIL | **-0.10** | **+0.3%** | 16 |
| Alt F: Water ultra-tight (6%/3%, conf=0.55) | FAIL | FAIL | **PASS** | FAIL | -0.13 | -0.5% | 17 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | -0.13 | -0.5% | 17 |
| WF tune: ATR stops x2.5 | FAIL | FAIL | **PASS** | FAIL | -0.13 | -0.5% | 17 |

---

## 5. Final Recommendation

**AWK partially validates.** Best config: WF tune: conf=0.6 (1/4 gates).

### WF tune: conf=0.6

- **Rules:** `utility_mean_reversion, utility_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.6
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+0.3%, Trades=16, WR=43.8%, Sharpe=-0.10, PF=1.01, DD=-17.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.25, Test Sharpe=-1.03, Ratio=-404% (need >=50%) |
| Bootstrap | FAIL | p=0.4414, Sharpe CI=[-3.89, 3.90], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.0%, Median equity=$1,026, Survival=100.0% |
| Regime | FAIL | bull:9t/-3.7%, bear:3t/+12.1%, chop:3t/-0.2%, crisis:1t/-4.3% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

