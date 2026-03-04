# OXY (Occidental Petroleum) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 22.5 minutes
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

OXY — E&P focused, Permian Basin, Buffett-backed, carbon capture. Large-cap E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 5 | 0.0% | -25.9% | -1.17 | 0.00 | -25.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 11.8% | -45.2% | -1.35 | 0.20 | -47.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 5 | 0.0% | -23.5% | -1.19 | 0.00 | -23.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 5 | 0.0% | -25.9% | -1.17 | 0.00 | -25.9% |
| Alt D: Energy rules (14 rules, 10%/5%) | 28 | 28.6% | -39.1% | -0.67 | 0.49 | -50.8% |
| Alt E: upstream sector rules (3 rules, 10%/5%) | 21 | 28.6% | -30.8% | -0.48 | 0.44 | -43.9% |
| Alt F: upstream rules wider stops (10%/6%) | 21 | 28.6% | -36.4% | -0.54 | 0.38 | -48.3% |

**Best baseline selected for validation: Alt E: upstream sector rules (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: upstream sector rules (3 rules, 10%/5%)

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-30.8%, Trades=21, WR=28.6%, Sharpe=-0.48, PF=0.44, DD=-43.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.45, Test Sharpe=0.74, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8418, Sharpe CI=[-7.32, 1.39], WR CI=[14.3%, 52.4%] |
| Monte Carlo | FAIL | Ruin=0.7%, P95 DD=-48.4%, Median equity=$690, Survival=99.4% |
| Regime | FAIL | bull:14t/-12.4%, bear:1t/-5.1%, chop:3t/+8.5%, volatile:3t/-22.7% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| MC tune: max_loss=3.0% | 22 | 27.3% | -9.7% | -0.24 |
| WF tune: conf=0.65 | 19 | 26.3% | -19.4% | -0.28 |
| BS tune: conf=0.4 | 21 | 28.6% | -29.2% | -0.46 |
| MC tune: ATR stops x2.0 | 21 | 28.6% | -30.0% | -0.47 |
| WF tune: ATR stops x2.5 | 21 | 28.6% | -30.8% | -0.47 |
| WF tune: conf=0.45 | 21 | 28.6% | -30.8% | -0.48 |
| WF tune: conf=0.55 | 21 | 28.6% | -30.8% | -0.48 |
| WF tune: conf=0.6 | 21 | 28.6% | -30.8% | -0.48 |
| BS tune: + energy_mean_reversion | 21 | 28.6% | -30.8% | -0.48 |
| MC tune: max_loss=4.0% | 22 | 27.3% | -25.9% | -0.57 |
| WF tune: cooldown=3 | 24 | 25.0% | -41.7% | -0.59 |
| WF tune: cooldown=7 | 19 | 31.6% | -32.0% | -0.60 |
| WF tune: PT=12% | 19 | 21.1% | -42.8% | -0.65 |
| WF tune: PT=15% | 19 | 21.1% | -42.8% | -0.65 |
| WF tune: PT=8% | 22 | 27.3% | -36.5% | -0.66 |
| BS tune: energy rules (14) | 28 | 28.6% | -39.1% | -0.67 |
| BS tune: full rules (10) | 17 | 11.8% | -45.2% | -1.35 |

### Full Validation of Top Candidates

### MC tune: max_loss=3.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=-9.7%, Trades=22, WR=27.3%, Sharpe=-0.24, PF=0.73, DD=-28.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.55, Test Sharpe=1.20, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5887, Sharpe CI=[-5.56, 2.38], WR CI=[13.6%, 50.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.2%, Median equity=$927, Survival=100.0% |
| Regime | FAIL | bull:15t/-0.2%, bear:1t/-3.2%, chop:3t/+11.3%, volatile:3t/-11.8% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-19.4%, Trades=19, WR=26.3%, Sharpe=-0.28, PF=0.63, DD=-37.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.04, Test Sharpe=1.05, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6755, Sharpe CI=[-5.48, 2.32], WR CI=[10.5%, 52.6%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-44.8%, Median equity=$816, Survival=99.9% |
| Regime | FAIL | bull:12t/-3.0%, bear:1t/-5.1%, chop:3t/+16.7%, volatile:3t/-22.7% |

**Result: 0/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-29.2%, Trades=21, WR=28.6%, Sharpe=-0.46, PF=0.47, DD=-42.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.38, Test Sharpe=0.74, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8164, Sharpe CI=[-6.77, 1.53], WR CI=[14.3%, 52.4%] |
| Monte Carlo | FAIL | Ruin=0.4%, P95 DD=-47.7%, Median equity=$711, Survival=99.6% |
| Regime | FAIL | bull:14t/-9.4%, bear:1t/-5.1%, chop:3t/+8.5%, volatile:3t/-22.7% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **MC tune: max_loss=3.0%** | FAIL | FAIL | **PASS** | FAIL | **-0.24** | **-9.7%** | 22 |
| WF tune: conf=0.65 | FAIL | FAIL | FAIL | FAIL | -0.28 | -19.4% | 19 |
| BS tune: conf=0.4 | FAIL | FAIL | FAIL | FAIL | -0.46 | -29.2% | 21 |
| Alt E: upstream sector rules (3 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | -0.48 | -30.8% | 21 |

---

## 5. Final Recommendation

**OXY partially validates.** Best config: MC tune: max_loss=3.0% (1/4 gates).

### MC tune: max_loss=3.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=-9.7%, Trades=22, WR=27.3%, Sharpe=-0.24, PF=0.73, DD=-28.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.55, Test Sharpe=1.20, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5887, Sharpe CI=[-5.56, 2.38], WR CI=[13.6%, 50.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.2%, Median equity=$927, Survival=100.0% |
| Regime | FAIL | bull:15t/-0.2%, bear:1t/-3.2%, chop:3t/+11.3%, volatile:3t/-11.8% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

