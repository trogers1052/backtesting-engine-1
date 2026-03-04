# SIL (Global X Silver Miners ETF) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 56.1 minutes
**Category:** Silver miners ETF (HIGH RISK)

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

SIL — Basket of ~30 silver miners — amplified silver exposure, -83% max DD historically. Silver miners ETF (HIGH RISK).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 30 | 46.7% | +46.9% | 0.35 | 1.48 | -28.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 47 | 44.7% | +25.6% | 0.21 | 1.15 | -39.9% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 33 | 42.4% | +20.0% | 0.22 | 1.13 | -27.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 28 | 46.4% | +59.1% | 0.41 | 1.64 | -24.5% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 48 | 45.8% | +42.8% | 0.27 | 1.23 | -34.0% |
| Alt E: silver_miner rules (4 rules, 10%/5%) | 31 | 45.2% | +43.9% | 0.34 | 1.42 | -30.1% |
| Alt F: Silver miner strict risk (10%/4%) | 30 | 40.0% | +13.5% | 0.29 | 1.13 | -26.9% |
| Alt G: Silver miner wider PT (12%/4%) | 33 | 42.4% | +44.7% | 0.34 | 1.35 | -27.0% |

**Best baseline selected for validation: Alt C: Wider PT (3 rules, 12%/5%)**

---

## 2. Full Validation

### Alt C: Wider PT (3 rules, 12%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+59.1%, Trades=28, WR=46.4%, Sharpe=0.41, PF=1.64, DD=-24.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.78, Test Sharpe=0.81, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0921, Sharpe CI=[-0.87, 4.64], WR CI=[28.6%, 64.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.2%, Median equity=$1,744, Survival=100.0% |
| Regime | **PASS** | bull:22t/+51.2%, bear:1t/+13.2%, chop:3t/+13.8%, volatile:2t/-8.7% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| MC tune: max_loss=3.0% | 33 | 39.4% | +44.9% | 0.41 |
| WF tune: conf=0.45 | 28 | 46.4% | +59.1% | 0.41 |
| WF tune: conf=0.55 | 28 | 46.4% | +59.1% | 0.41 |
| WF tune: ATR stops x2.5 | 28 | 46.4% | +59.1% | 0.41 |
| WF tune: + miner_metal_ratio | 28 | 46.4% | +59.1% | 0.41 |
| BS tune: conf=0.4 | 28 | 46.4% | +59.1% | 0.41 |
| MC tune: ATR stops x2.0 | 28 | 46.4% | +59.1% | 0.41 |
| WF tune: conf=0.6 | 28 | 46.4% | +58.3% | 0.40 |
| WF tune: cooldown=3 | 30 | 50.0% | +80.2% | 0.39 |
| BS tune: silver_miner rules | 29 | 44.8% | +55.8% | 0.39 |
| BS tune: + volume_breakout | 29 | 44.8% | +55.8% | 0.39 |
| MC tune: max_loss=4.0% | 32 | 43.8% | +48.9% | 0.36 |
| BS tune: full mining rules (14) | 43 | 41.9% | +81.9% | 0.34 |
| WF tune: PT=15% | 27 | 37.0% | +42.1% | 0.33 |
| BS tune: full rules (10) | 41 | 43.9% | +69.4% | 0.32 |
| WF tune: cooldown=7 | 28 | 46.4% | +31.0% | 0.27 |
| WF tune: + commodity_breakout | 32 | 37.5% | +26.6% | 0.25 |
| WF tune: PT=8% | 30 | 46.7% | +13.1% | 0.14 |
| WF tune: conf=0.65 | 22 | 36.4% | +10.8% | 0.12 |
| MC tune: max_loss=3.0% [multi-TF] | 52 | 34.6% | +9.7% | 0.11 |

### Full Validation of Top Candidates

### MC tune: max_loss=3.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=+44.9%, Trades=33, WR=39.4%, Sharpe=0.41, PF=1.40, DD=-25.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.53, Test Sharpe=0.65, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1140, Sharpe CI=[-1.03, 3.91], WR CI=[27.3%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.4%, Median equity=$1,600, Survival=100.0% |
| Regime | **PASS** | bull:24t/+33.8%, bear:4t/+10.9%, chop:3t/+18.2%, volatile:2t/-4.6% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+59.1%, Trades=28, WR=46.4%, Sharpe=0.41, PF=1.64, DD=-24.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.78, Test Sharpe=0.81, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0921, Sharpe CI=[-0.87, 4.64], WR CI=[28.6%, 64.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.2%, Median equity=$1,744, Survival=100.0% |
| Regime | **PASS** | bull:22t/+51.2%, bear:1t/+13.2%, chop:3t/+13.8%, volatile:2t/-8.7% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+59.1%, Trades=28, WR=46.4%, Sharpe=0.41, PF=1.64, DD=-24.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.78, Test Sharpe=0.81, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0921, Sharpe CI=[-0.87, 4.64], WR CI=[28.6%, 64.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.2%, Median equity=$1,744, Survival=100.0% |
| Regime | **PASS** | bull:22t/+51.2%, bear:1t/+13.2%, chop:3t/+13.8%, volatile:2t/-8.7% |

**Result: 1/4 gates passed**

---

### MC tune: max_loss=3.0% [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=+9.7%, Trades=52, WR=34.6%, Sharpe=0.11, PF=1.10, DD=-32.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=1.08, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2583, Sharpe CI=[-1.58, 2.35], WR CI=[23.1%, 50.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.8%, Median equity=$1,228, Survival=100.0% |
| Regime | **PASS** | bull:38t/+3.3%, bear:8t/+9.8%, chop:3t/+21.5%, volatile:3t/-6.4% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **MC tune: max_loss=3.0%** | FAIL | FAIL | **PASS** | **PASS** | **0.41** | **+44.9%** | 33 |
| MC tune: max_loss=3.0% [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.11 | +9.7% | 52 |
| Alt C: Wider PT (3 rules, 12%/5%) | FAIL | FAIL | FAIL | **PASS** | 0.41 | +59.1% | 28 |
| WF tune: conf=0.45 | FAIL | FAIL | FAIL | **PASS** | 0.41 | +59.1% | 28 |
| WF tune: conf=0.55 | FAIL | FAIL | FAIL | **PASS** | 0.41 | +59.1% | 28 |

---

## 5. Final Recommendation

**SIL partially validates.** Best config: MC tune: max_loss=3.0% (2/4 gates).

### MC tune: max_loss=3.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=+44.9%, Trades=33, WR=39.4%, Sharpe=0.41, PF=1.40, DD=-25.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.53, Test Sharpe=0.65, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1140, Sharpe CI=[-1.03, 3.91], WR CI=[27.3%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.4%, Median equity=$1,600, Survival=100.0% |
| Regime | **PASS** | bull:24t/+33.8%, bear:4t/+10.9%, chop:3t/+18.2%, volatile:2t/-4.6% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

