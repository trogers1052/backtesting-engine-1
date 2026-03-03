# PANW (Palo Alto Networks) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 66.6 minutes
**Category:** Large-cap cybersecurity

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

PANW — Cybersecurity — firewalls, cloud security, SASE. Large-cap cybersecurity.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 16 | 37.5% | -4.3% | -0.13 | 0.94 | -28.8% |
| Alt A: Full general rules (10 rules, 10%/5%) | 34 | 44.1% | +19.9% | 0.23 | 1.14 | -24.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 19 | 42.1% | +26.6% | 0.27 | 1.39 | -24.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 14 | 28.6% | -21.9% | -0.41 | 0.67 | -33.6% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 16 | 37.5% | -4.3% | -0.13 | 0.94 | -28.8% |

**Best baseline selected for validation: Alt B: Tighter stops (3 rules, 10%/4%)**

---

## 2. Full Validation

### Alt B: Tighter stops (3 rules, 10%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.6%, Trades=19, WR=42.1%, Sharpe=0.27, PF=1.39, DD=-24.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-0.94, Ratio=-123% (need >=50%) |
| Bootstrap | FAIL | p=0.1820, Sharpe CI=[-2.03, 4.95], WR CI=[21.1%, 63.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.9%, Median equity=$1,321, Survival=100.0% |
| Regime | **PASS** | bull:16t/+15.0%, chop:2t/+6.8%, volatile:1t/+13.0% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=3 | 21 | 42.9% | +40.1% | 0.34 |
| WF tune: PT=8% | 23 | 47.8% | +26.4% | 0.28 |
| WF tune: conf=0.45 | 19 | 42.1% | +26.6% | 0.27 |
| BS tune: conf=0.4 | 19 | 42.1% | +26.6% | 0.27 |
| BS tune: + volume_breakout | 19 | 42.1% | +26.6% | 0.27 |
| WF tune: conf=0.55 | 18 | 38.9% | +14.8% | 0.18 |
| WF tune: conf=0.6 | 18 | 38.9% | +14.8% | 0.18 |
| WF tune: conf=0.65 | 18 | 38.9% | +14.8% | 0.18 |
| WF tune: cooldown=7 | 19 | 36.8% | +10.2% | 0.11 |
| WF tune: PT=12% | 16 | 31.2% | +3.0% | 0.04 |
| WF tune: PT=15% | 14 | 28.6% | +1.6% | 0.01 |
| BS tune: full rules (10) | 39 | 35.9% | -5.0% | -0.26 |
| WF tune: cooldown=3 [multi-TF] | 29 | 34.5% | +0.6% | -0.04 |

### Full Validation of Top Candidates

### WF tune: cooldown=3

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.1%, Trades=21, WR=42.9%, Sharpe=0.34, PF=1.54, DD=-25.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.87, Test Sharpe=-0.99, Ratio=-113% (need >=50%) |
| Bootstrap | FAIL | p=0.1202, Sharpe CI=[-1.32, 5.03], WR CI=[23.8%, 61.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.8%, Median equity=$1,486, Survival=100.0% |
| Regime | **PASS** | bull:18t/+27.1%, chop:2t/+6.8%, volatile:1t/+13.0% |

**Result: 2/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.4%, Trades=23, WR=47.8%, Sharpe=0.28, PF=1.38, DD=-23.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.38, Test Sharpe=-1.16, Ratio=-305% (need >=50%) |
| Bootstrap | FAIL | p=0.1497, Sharpe CI=[-1.47, 4.78], WR CI=[26.1%, 69.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.2%, Median equity=$1,370, Survival=100.0% |
| Regime | FAIL | bull:20t/+28.8%, chop:2t/+0.9%, volatile:1t/+8.1% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.6%, Trades=19, WR=42.1%, Sharpe=0.27, PF=1.39, DD=-24.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-0.94, Ratio=-123% (need >=50%) |
| Bootstrap | FAIL | p=0.1820, Sharpe CI=[-2.03, 4.95], WR CI=[21.1%, 63.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.9%, Median equity=$1,321, Survival=100.0% |
| Regime | **PASS** | bull:16t/+15.0%, chop:2t/+6.8%, volatile:1t/+13.0% |

**Result: 2/4 gates passed**

---

### WF tune: cooldown=3 [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.6%, Trades=29, WR=34.5%, Sharpe=-0.04, PF=1.01, DD=-22.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.05, Test Sharpe=-1.01, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3639, Sharpe CI=[-2.43, 3.10], WR CI=[17.2%, 51.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.9%, Median equity=$1,086, Survival=100.0% |
| Regime | FAIL | bull:26t/+14.9%, chop:3t/+0.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=3** | FAIL | FAIL | **PASS** | **PASS** | **0.34** | **+40.1%** | 21 |
| Alt B: Tighter stops (3 rules, 10%/4%) | FAIL | FAIL | **PASS** | **PASS** | 0.27 | +26.6% | 19 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | **PASS** | 0.27 | +26.6% | 19 |
| WF tune: PT=8% | FAIL | FAIL | **PASS** | FAIL | 0.28 | +26.4% | 23 |
| WF tune: cooldown=3 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.04 | +0.6% | 29 |

---

## 5. Final Recommendation

**PANW partially validates.** Best config: WF tune: cooldown=3 (2/4 gates).

### WF tune: cooldown=3

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+40.1%, Trades=21, WR=42.9%, Sharpe=0.34, PF=1.54, DD=-25.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.87, Test Sharpe=-0.99, Ratio=-113% (need >=50%) |
| Bootstrap | FAIL | p=0.1202, Sharpe CI=[-1.32, 5.03], WR CI=[23.8%, 61.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.8%, Median equity=$1,486, Survival=100.0% |
| Regime | **PASS** | bull:18t/+27.1%, chop:2t/+6.8%, volatile:1t/+13.0% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

