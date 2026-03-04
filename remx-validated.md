# REMX (VanEck Rare Earth/Strategic Metals ETF) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 36.5 minutes
**Category:** Rare earth ETF

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

REMX — Rare earth and strategic metals — geopolitical spike trading, China supply risk. Rare earth ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 21 | 47.6% | -0.7% | -0.01 | 0.99 | -34.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 22 | 40.9% | -0.7% | -0.02 | 0.99 | -34.2% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 23 | 43.5% | +1.8% | 0.01 | 1.03 | -31.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 19 | 36.8% | -9.6% | -0.19 | 0.67 | -34.6% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 31 | 29.0% | -16.1% | -0.11 | 0.75 | -49.4% |
| Alt E: rare_earth rules (4 rules, 10%/5%) | 25 | 40.0% | -18.7% | -0.27 | 0.69 | -42.9% |
| Alt F: Rare earth wide PT (15%/5%) | 23 | 34.8% | -8.7% | -0.06 | 0.71 | -42.9% |
| Alt G: MP-style lean (15%/6%, conf=0.50) | 19 | 36.8% | -3.6% | -0.04 | 0.77 | -37.5% |

**Best baseline selected for validation: Alt B: Tighter stops (3 rules, 10%/4%)**

---

## 2. Full Validation

### Alt B: Tighter stops (3 rules, 10%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+1.8%, Trades=23, WR=43.5%, Sharpe=0.01, PF=1.03, DD=-31.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.55, Test Sharpe=0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3699, Sharpe CI=[-2.86, 3.43], WR CI=[26.1%, 65.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.4%, Median equity=$1,073, Survival=100.0% |
| Regime | FAIL | bull:20t/+26.1%, bear:1t/-5.1%, chop:2t/-7.8% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.6 | 21 | 42.9% | +9.6% | 0.10 |
| BS tune: conf=0.4 | 23 | 43.5% | +8.9% | 0.09 |
| WF tune: conf=0.45 | 23 | 43.5% | +1.8% | 0.01 |
| WF tune: conf=0.55 | 23 | 43.5% | +1.8% | 0.01 |
| WF tune: + miner_metal_ratio | 23 | 43.5% | +1.8% | 0.01 |
| WF tune: ATR stops x2.5 | 23 | 43.5% | -0.4% | -0.03 |
| WF tune: conf=0.65 | 18 | 33.3% | -1.3% | -0.05 |
| WF tune: cooldown=3 | 29 | 37.9% | -8.5% | -0.06 |
| WF tune: PT=15% | 21 | 33.3% | -3.3% | -0.08 |
| WF tune: PT=12% | 21 | 33.3% | -5.6% | -0.14 |
| BS tune: full rules (10) | 27 | 33.3% | -13.2% | -0.15 |
| BS tune: full mining rules (14) | 35 | 25.7% | -23.4% | -0.17 |
| BS tune: rare_earth rules | 27 | 37.0% | -14.1% | -0.20 |
| BS tune: + volume_breakout | 27 | 37.0% | -14.1% | -0.20 |
| WF tune: PT=8% | 25 | 40.0% | -10.6% | -0.26 |
| WF tune: + commodity_breakout | 30 | 26.7% | -28.9% | -0.47 |
| WF tune: cooldown=7 | 22 | 27.3% | -24.2% | -0.67 |

### Full Validation of Top Candidates

### WF tune: conf=0.6

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+9.6%, Trades=21, WR=42.9%, Sharpe=0.10, PF=1.18, DD=-27.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.71, Test Sharpe=0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2979, Sharpe CI=[-2.60, 3.94], WR CI=[23.8%, 61.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:18t/+32.4%, bear:1t/-5.1%, chop:2t/-7.8% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+8.9%, Trades=23, WR=43.5%, Sharpe=0.09, PF=1.16, DD=-28.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.27, Test Sharpe=0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3213, Sharpe CI=[-2.55, 3.66], WR CI=[26.1%, 65.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.6%, Median equity=$1,122, Survival=100.0% |
| Regime | FAIL | bull:20t/+30.7%, bear:1t/-5.1%, chop:2t/-7.8% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+1.8%, Trades=23, WR=43.5%, Sharpe=0.01, PF=1.03, DD=-31.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.55, Test Sharpe=0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3699, Sharpe CI=[-2.86, 3.43], WR CI=[26.1%, 65.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.4%, Median equity=$1,073, Survival=100.0% |
| Regime | FAIL | bull:20t/+26.1%, bear:1t/-5.1%, chop:2t/-7.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.6** | FAIL | FAIL | **PASS** | FAIL | **0.10** | **+9.6%** | 21 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | FAIL | 0.09 | +8.9% | 23 |
| Alt B: Tighter stops (3 rules, 10%/4%) | FAIL | FAIL | **PASS** | FAIL | 0.01 | +1.8% | 23 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.01 | +1.8% | 23 |

---

## 5. Final Recommendation

**REMX partially validates.** Best config: WF tune: conf=0.6 (1/4 gates).

### WF tune: conf=0.6

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+9.6%, Trades=21, WR=42.9%, Sharpe=0.10, PF=1.18, DD=-27.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.71, Test Sharpe=0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2979, Sharpe CI=[-2.60, 3.94], WR CI=[23.8%, 61.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.3%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:18t/+32.4%, bear:1t/-5.1%, chop:2t/-7.8% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

