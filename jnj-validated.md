# JNJ (Johnson & Johnson) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 69.9 minutes
**Category:** Diversified healthcare (ultra-defensive)

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

JNJ — Diversified healthcare — beta 0.14-0.35, 54yr Dividend King, post-Kenvue spinoff. Diversified healthcare (ultra-defensive).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 8 | 37.5% | +3.3% | -0.01 | 1.16 | -21.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 17.6% | -13.0% | -0.27 | 0.50 | -36.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 37.5% | +6.9% | 0.05 | 1.41 | -18.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 8 | 37.5% | +7.9% | 0.08 | 1.39 | -21.9% |
| Alt D: Healthcare rules (13 rules, 10%/5%) | 33 | 45.5% | -7.1% | -0.15 | 0.80 | -38.0% |
| Alt E: diversified lean (3 rules, 10%/5%) | 23 | 65.2% | +17.6% | 0.26 | 1.58 | -19.7% |
| Alt F: Diversified tight (6%/3%, conf=0.55) | 21 | 57.1% | +7.6% | 0.07 | 1.26 | -17.2% |
| Alt G: Diversified moderate (8%/4%) | 26 | 57.7% | +1.1% | -0.07 | 0.98 | -26.0% |

**Best baseline selected for validation: Alt E: diversified lean (3 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: diversified lean (3 rules, 10%/5%)

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+17.6%, Trades=23, WR=65.2%, Sharpe=0.26, PF=1.58, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.72, Test Sharpe=1.18, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1376, Sharpe CI=[-1.51, 4.58], WR CI=[52.2%, 87.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.7%, Median equity=$1,269, Survival=100.0% |
| Regime | FAIL | bull:16t/+38.1%, bear:2t/-4.7%, chop:3t/-10.2%, volatile:2t/+3.5% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 12 | 66.7% | +30.6% | 0.62 |
| WF tune: cooldown=3 | 24 | 62.5% | +25.2% | 0.37 |
| WF tune: PT=12% | 23 | 65.2% | +27.3% | 0.35 |
| Regime tune: tighter stop 4% | 24 | 62.5% | +17.4% | 0.28 |
| WF tune: ATR stops x2.5 | 24 | 62.5% | +17.3% | 0.28 |
| WF tune: conf=0.6 | 23 | 60.9% | +18.1% | 0.27 |
| WF tune: conf=0.55 | 23 | 65.2% | +17.6% | 0.26 |
| WF tune: PT=15% | 22 | 63.6% | +18.4% | 0.25 |
| WF tune: PT=6% | 24 | 66.7% | +11.8% | 0.16 |
| WF tune: PT=8% | 23 | 65.2% | +11.6% | 0.15 |
| WF tune: cooldown=7 | 20 | 60.0% | +12.0% | 0.15 |
| WF tune: conf=0.45 | 23 | 60.9% | +10.5% | 0.12 |
| BS tune: conf=0.4 | 23 | 60.9% | +10.5% | 0.12 |
| WF tune: PT=7% | 23 | 65.2% | +8.5% | 0.09 |
| WF tune: + hc_pullback | 25 | 60.0% | +7.5% | 0.07 |
| BS tune: healthcare rules (13) | 33 | 45.5% | -7.1% | -0.15 |
| BS tune: full rules (10) | 17 | 17.6% | -13.0% | -0.27 |
| WF tune: conf=0.65 [multi-TF] | 19 | 57.9% | +27.5% | 0.29 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+30.6%, Trades=12, WR=66.7%, Sharpe=0.62, PF=2.70, DD=-13.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.67, Test Sharpe=1.05, Ratio=156% (need >=50%) |
| Bootstrap | FAIL | p=0.0554, Sharpe CI=[-0.83, 9.92], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-16.2%, Median equity=$1,396, Survival=100.0% |
| Regime | FAIL | bull:9t/+44.7%, bear:1t/-5.1%, chop:1t/-7.8%, volatile:1t/+4.5% |

**Result: 2/4 gates passed**

---

### WF tune: cooldown=3

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+25.2%, Trades=24, WR=62.5%, Sharpe=0.37, PF=1.83, DD=-19.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=1.18, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1117, Sharpe CI=[-1.17, 4.78], WR CI=[45.8%, 83.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.3%, Median equity=$1,312, Survival=100.0% |
| Regime | FAIL | bull:17t/+40.4%, bear:3t/-2.7%, chop:3t/-8.6%, volatile:1t/+1.0% |

**Result: 1/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+27.3%, Trades=23, WR=65.2%, Sharpe=0.35, PF=1.96, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.72, Test Sharpe=1.15, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1129, Sharpe CI=[-1.34, 4.52], WR CI=[52.2%, 87.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.7%, Median equity=$1,339, Survival=100.0% |
| Regime | FAIL | bull:16t/+44.1%, bear:2t/-4.7%, chop:3t/-10.2%, volatile:2t/+3.5% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.65 [multi-TF]

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+27.5%, Trades=19, WR=57.9%, Sharpe=0.29, PF=2.04, DD=-16.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.63, Test Sharpe=1.13, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0829, Sharpe CI=[-1.07, 5.64], WR CI=[42.1%, 84.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.1%, Median equity=$1,355, Survival=100.0% |
| Regime | FAIL | bull:16t/+39.0%, bear:1t/-5.0%, chop:1t/-5.0%, volatile:1t/+4.4% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | **PASS** | FAIL | **PASS** | FAIL | **0.62** | **+30.6%** | 12 |
| WF tune: cooldown=3 | FAIL | FAIL | **PASS** | FAIL | 0.37 | +25.2% | 24 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | FAIL | 0.35 | +27.3% | 23 |
| WF tune: conf=0.65 [multi-TF] | FAIL | FAIL | **PASS** | FAIL | 0.29 | +27.5% | 19 |
| Alt E: diversified lean (3 rules, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.26 | +17.6% | 23 |

---

## 5. Final Recommendation

**JNJ partially validates.** Best config: WF tune: conf=0.65 (2/4 gates).

### WF tune: conf=0.65

- **Rules:** `healthcare_mean_reversion, healthcare_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+30.6%, Trades=12, WR=66.7%, Sharpe=0.62, PF=2.70, DD=-13.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.67, Test Sharpe=1.05, Ratio=156% (need >=50%) |
| Bootstrap | FAIL | p=0.0554, Sharpe CI=[-0.83, 9.92], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-16.2%, Median equity=$1,396, Survival=100.0% |
| Regime | FAIL | bull:9t/+44.7%, bear:1t/-5.1%, chop:1t/-7.8%, volatile:1t/+4.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

