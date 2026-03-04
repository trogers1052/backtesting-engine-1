# AMZN (Amazon) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 50.2 minutes
**Category:** Mega-cap tech

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

AMZN — E-commerce, cloud (AWS), advertising — mega-cap quality growth. Mega-cap tech.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 58.3% | +29.9% | 0.29 | 1.52 | -19.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 25 | 40.0% | +6.0% | 0.07 | 1.05 | -30.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 53.8% | +29.0% | 0.29 | 1.49 | -22.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 12 | 50.0% | +17.7% | 0.22 | 1.31 | -18.4% |
| Alt D: Tech rules (13 rules, 10%/5%) | 28 | 39.3% | +6.4% | 0.06 | 1.06 | -24.7% |
| Alt E: mega_cap rules (4 rules, 10%/5%) | 16 | 50.0% | +26.7% | 0.30 | 1.39 | -19.1% |
| Alt F: Mega-cap balanced (8%/4%, conf=0.55) | 21 | 47.6% | +20.2% | 0.26 | 1.28 | -17.1% |

**Best baseline selected for validation: Alt E: mega_cap rules (4 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: mega_cap rules (4 rules, 10%/5%)

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.7%, Trades=16, WR=50.0%, Sharpe=0.30, PF=1.39, DD=-19.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.84, Test Sharpe=-0.39, Ratio=-46% (need >=50%) |
| Bootstrap | FAIL | p=0.1529, Sharpe CI=[-1.79, 6.17], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.9%, Median equity=$1,374, Survival=100.0% |
| Regime | **PASS** | bull:15t/+27.1%, chop:1t/+12.2% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 16 | 56.2% | +45.8% | 0.43 |
| BS tune: conf=0.4 | 16 | 56.2% | +45.8% | 0.43 |
| WF tune: cooldown=3 | 16 | 50.0% | +29.7% | 0.32 |
| WF tune: conf=0.55 | 16 | 50.0% | +26.7% | 0.30 |
| WF tune: ATR stops x2.5 | 16 | 50.0% | +26.7% | 0.30 |
| WF tune: conf=0.6 | 15 | 46.7% | +22.3% | 0.28 |
| WF tune: cooldown=7 | 14 | 50.0% | +21.0% | 0.28 |
| WF tune: PT=8% | 20 | 50.0% | +19.4% | 0.25 |
| WF tune: PT=12% | 16 | 43.8% | +17.9% | 0.24 |
| WF tune: conf=0.65 | 12 | 50.0% | +15.6% | 0.19 |
| WF tune: PT=15% | 13 | 38.5% | +13.9% | 0.15 |
| BS tune: full rules (10) | 25 | 40.0% | +6.0% | 0.07 |
| BS tune: tech rules (13) | 28 | 39.3% | +6.4% | 0.06 |
| WF tune: cooldown=3 [multi-TF] | 29 | 48.3% | -5.1% | -0.12 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+45.8%, Trades=16, WR=56.2%, Sharpe=0.43, PF=1.65, DD=-17.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.92, Test Sharpe=-0.39, Ratio=-42% (need >=50%) |
| Bootstrap | FAIL | p=0.0928, Sharpe CI=[-1.13, 7.09], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.4%, Median equity=$1,563, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, chop:1t/+12.2% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+45.8%, Trades=16, WR=56.2%, Sharpe=0.43, PF=1.65, DD=-17.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.92, Test Sharpe=-0.39, Ratio=-42% (need >=50%) |
| Bootstrap | FAIL | p=0.0928, Sharpe CI=[-1.13, 7.09], WR CI=[31.2%, 81.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.4%, Median equity=$1,563, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, chop:1t/+12.2% |

**Result: 1/4 gates passed**

---

### WF tune: cooldown=3

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+29.7%, Trades=16, WR=50.0%, Sharpe=0.32, PF=1.43, DD=-18.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.86, Test Sharpe=-0.45, Ratio=-52% (need >=50%) |
| Bootstrap | FAIL | p=0.1595, Sharpe CI=[-1.94, 5.97], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$1,371, Survival=100.0% |
| Regime | **PASS** | bull:15t/+27.2%, chop:1t/+12.2% |

**Result: 2/4 gates passed**

---

### WF tune: cooldown=3 [multi-TF]

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-5.1%, Trades=29, WR=48.3%, Sharpe=-0.12, PF=0.93, DD=-30.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.69, Test Sharpe=-1.58, Ratio=-231% (need >=50%) |
| Bootstrap | FAIL | p=0.4333, Sharpe CI=[-2.56, 2.82], WR CI=[31.0%, 65.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.7%, Median equity=$1,017, Survival=100.0% |
| Regime | **PASS** | bull:26t/+4.0%, chop:3t/+3.5% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=3** | FAIL | FAIL | **PASS** | **PASS** | **0.32** | **+29.7%** | 16 |
| Alt E: mega_cap rules (4 rules, 10%/5%) | FAIL | FAIL | **PASS** | **PASS** | 0.30 | +26.7% | 16 |
| WF tune: cooldown=3 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | -0.12 | -5.1% | 29 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.43 | +45.8% | 16 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | FAIL | 0.43 | +45.8% | 16 |

---

## 5. Final Recommendation

**AMZN partially validates.** Best config: WF tune: cooldown=3 (2/4 gates).

### WF tune: cooldown=3

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+29.7%, Trades=16, WR=50.0%, Sharpe=0.32, PF=1.43, DD=-18.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.86, Test Sharpe=-0.45, Ratio=-52% (need >=50%) |
| Bootstrap | FAIL | p=0.1595, Sharpe CI=[-1.94, 5.97], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$1,371, Survival=100.0% |
| Regime | **PASS** | bull:15t/+27.2%, chop:1t/+12.2% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

