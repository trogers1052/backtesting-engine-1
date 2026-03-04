# MA (Mastercard) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 64.7 minutes
**Category:** Payment network

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

MA — Global payment network — transaction processing, data analytics. Payment network.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 50.0% | +16.7% | 0.22 | 1.62 | -15.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 14 | 57.1% | +16.4% | 0.20 | 1.36 | -18.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 41.7% | +14.3% | 0.24 | 1.46 | -12.6% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 12 | 58.3% | +22.1% | 0.36 | 1.73 | -12.2% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 18 | 55.6% | +13.8% | 0.17 | 1.25 | -19.5% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 11 | 54.5% | +7.3% | 0.07 | 1.23 | -21.6% |

**Best baseline selected for validation: Alt C: Smaller PT (3 rules, 8%/5%)**

---

## 2. Full Validation

### Alt C: Smaller PT (3 rules, 8%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+22.1%, Trades=12, WR=58.3%, Sharpe=0.36, PF=1.73, DD=-12.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.98, Ratio=-176% (need >=50%) |
| Bootstrap | FAIL | p=0.1331, Sharpe CI=[-1.67, 8.23], WR CI=[33.3%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.3%, Median equity=$1,302, Survival=100.0% |
| Regime | FAIL | bull:8t/+37.6%, bear:1t/+9.1%, chop:3t/-16.7% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=12% | 8 | 62.5% | +35.2% | 0.53 |
| WF tune: conf=0.45 | 12 | 58.3% | +22.1% | 0.36 |
| BS tune: conf=0.4 | 12 | 58.3% | +22.1% | 0.36 |
| BS tune: + volume_breakout | 12 | 58.3% | +22.1% | 0.36 |
| BS tune: + financial_mean_reversion | 12 | 58.3% | +22.1% | 0.36 |
| WF tune: cooldown=7 | 12 | 58.3% | +20.8% | 0.26 |
| WF tune: cooldown=3 | 13 | 53.8% | +17.2% | 0.22 |
| WF tune: conf=0.55 | 12 | 50.0% | +12.7% | 0.19 |
| WF tune: conf=0.6 | 12 | 50.0% | +12.7% | 0.19 |
| BS tune: financial rules (12) | 20 | 60.0% | +14.4% | 0.17 |
| BS tune: full rules (10) | 19 | 52.6% | +13.2% | 0.16 |
| WF tune: conf=0.65 | 12 | 50.0% | +11.4% | 0.16 |
| Regime tune: tighter stop 4% | 13 | 46.2% | +12.2% | 0.14 |
| WF tune: PT=15% | 8 | 37.5% | +7.3% | 0.07 |
| Regime tune: + financial_seasonality | 16 | 56.2% | +7.3% | 0.07 |

### Full Validation of Top Candidates

### WF tune: PT=12%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+35.2%, Trades=8, WR=62.5%, Sharpe=0.53, PF=2.98, DD=-11.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.62, Test Sharpe=-0.98, Ratio=-156% (need >=50%) |
| Bootstrap | FAIL | p=0.0386, Sharpe CI=[-0.71, 12.06], WR CI=[25.0%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.2%, Median equity=$1,546, Survival=100.0% |
| Regime | FAIL | bull:5t/+45.4%, bear:1t/+13.2%, chop:2t/-10.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+22.1%, Trades=12, WR=58.3%, Sharpe=0.36, PF=1.73, DD=-12.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.98, Ratio=-176% (need >=50%) |
| Bootstrap | FAIL | p=0.1331, Sharpe CI=[-1.67, 8.23], WR CI=[33.3%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.3%, Median equity=$1,302, Survival=100.0% |
| Regime | FAIL | bull:8t/+37.6%, bear:1t/+9.1%, chop:3t/-16.7% |

**Result: 1/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+22.1%, Trades=12, WR=58.3%, Sharpe=0.36, PF=1.73, DD=-12.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.56, Test Sharpe=-0.98, Ratio=-176% (need >=50%) |
| Bootstrap | FAIL | p=0.1331, Sharpe CI=[-1.67, 8.23], WR CI=[33.3%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.3%, Median equity=$1,302, Survival=100.0% |
| Regime | FAIL | bull:8t/+37.6%, bear:1t/+9.1%, chop:3t/-16.7% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=12%** | FAIL | FAIL | **PASS** | FAIL | **0.53** | **+35.2%** | 8 |
| Alt C: Smaller PT (3 rules, 8%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.36 | +22.1% | 12 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.36 | +22.1% | 12 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | FAIL | 0.36 | +22.1% | 12 |

---

## 5. Final Recommendation

**MA partially validates.** Best config: WF tune: PT=12% (1/4 gates).

### WF tune: PT=12%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+35.2%, Trades=8, WR=62.5%, Sharpe=0.53, PF=2.98, DD=-11.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.62, Test Sharpe=-0.98, Ratio=-156% (need >=50%) |
| Bootstrap | FAIL | p=0.0386, Sharpe CI=[-0.71, 12.06], WR CI=[25.0%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.2%, Median equity=$1,546, Survival=100.0% |
| Regime | FAIL | bull:5t/+45.4%, bear:1t/+13.2%, chop:2t/-10.9% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

