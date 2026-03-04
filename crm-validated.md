# CRM (Salesforce) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 31.3 minutes
**Category:** Enterprise SaaS

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

CRM — Enterprise SaaS, CRM, AI (Agentforce) — momentum-driven growth. Enterprise SaaS.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 63.6% | +52.3% | 0.53 | 2.49 | -9.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 18 | 50.0% | +18.2% | 0.23 | 1.25 | -24.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 63.6% | +52.3% | 0.53 | 2.49 | -9.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 62.5% | +38.0% | 0.43 | 2.32 | -14.9% |
| Alt D: Tech rules (13 rules, 10%/5%) | 27 | 44.4% | +21.3% | 0.20 | 1.18 | -28.8% |
| Alt E: saas rules (3 rules, 10%/5%) | 17 | 47.1% | +22.3% | 0.21 | 1.35 | -28.1% |
| Alt F: SaaS momentum (12%/6%) | 15 | 46.7% | +12.7% | 0.14 | 1.17 | -29.9% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+52.3%, Trades=11, WR=63.6%, Sharpe=0.53, PF=2.49, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0391, Sharpe CI=[-0.43, 13.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,639, Survival=100.0% |
| Regime | FAIL | bull:10t/+45.1%, chop:1t/+9.9% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: conf=0.55 | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: conf=0.6 | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: conf=0.65 | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: ATR stops x2.5 | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: + tech_ema_pullback | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: + tech_mean_reversion | 11 | 63.6% | +52.3% | 0.53 |
| BS tune: conf=0.4 | 11 | 63.6% | +52.3% | 0.53 |
| Regime tune: tighter stop 4% | 11 | 63.6% | +52.3% | 0.53 |
| WF tune: cooldown=3 | 11 | 63.6% | +53.0% | 0.53 |
| WF tune: PT=8% | 11 | 63.6% | +32.4% | 0.44 |
| WF tune: PT=12% | 8 | 62.5% | +38.0% | 0.43 |
| WF tune: PT=15% | 7 | 57.1% | +35.0% | 0.43 |
| WF tune: cooldown=7 | 10 | 60.0% | +40.2% | 0.40 |
| BS tune: full rules (10) | 18 | 50.0% | +18.2% | 0.23 |
| BS tune: saas rules | 17 | 47.1% | +22.3% | 0.21 |
| BS tune: tech rules (13) | 27 | 44.4% | +21.3% | 0.20 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+52.3%, Trades=11, WR=63.6%, Sharpe=0.53, PF=2.49, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0391, Sharpe CI=[-0.43, 13.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,639, Survival=100.0% |
| Regime | FAIL | bull:10t/+45.1%, chop:1t/+9.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+52.3%, Trades=11, WR=63.6%, Sharpe=0.53, PF=2.49, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0391, Sharpe CI=[-0.43, 13.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,639, Survival=100.0% |
| Regime | FAIL | bull:10t/+45.1%, chop:1t/+9.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+52.3%, Trades=11, WR=63.6%, Sharpe=0.53, PF=2.49, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0391, Sharpe CI=[-0.43, 13.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,639, Survival=100.0% |
| Regime | FAIL | bull:10t/+45.1%, chop:1t/+9.9% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Lean 3 rules baseline (10%/5%, conf=0.50)** | FAIL | FAIL | **PASS** | FAIL | **0.53** | **+52.3%** | 11 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.53 | +52.3% | 11 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.53 | +52.3% | 11 |
| WF tune: conf=0.6 | FAIL | FAIL | **PASS** | FAIL | 0.53 | +52.3% | 11 |

---

## 5. Final Recommendation

**CRM partially validates.** Best config: Lean 3 rules baseline (10%/5%, conf=0.50) (1/4 gates).

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+52.3%, Trades=11, WR=63.6%, Sharpe=0.53, PF=2.49, DD=-9.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0391, Sharpe CI=[-0.43, 13.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.4%, Median equity=$1,639, Survival=100.0% |
| Regime | FAIL | bull:10t/+45.1%, chop:1t/+9.9% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

