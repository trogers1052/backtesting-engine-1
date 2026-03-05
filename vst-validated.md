# VST (Vistra Corp) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 75.5 minutes
**Category:** Nuclear / AI power (MOMENTUM)

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

VST — Nuclear + natural gas power — AI data center proxy, beta 1.4+, extremely volatile. Nuclear / AI power (MOMENTUM).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 20 | 50.0% | +38.9% | 0.36 | 1.47 | -25.4% |
| Alt A: Full general rules (10 rules, 10%/5%) | 39 | 43.6% | +49.2% | 0.38 | 1.18 | -41.7% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 20 | 45.0% | +21.0% | 0.21 | 1.26 | -25.8% |
| Alt C: Wider PT (3 rules, 12%/5%) | 18 | 44.4% | +36.7% | 0.45 | 1.47 | -25.1% |
| Alt D: Utility rules (13 rules, 10%/5%) | 41 | 41.5% | +30.6% | 0.30 | 1.12 | -42.9% |
| Alt E: nuclear_power lean (3 rules, 10%/5%) | 20 | 50.0% | +38.9% | 0.36 | 1.47 | -25.4% |
| Alt F: Nuclear momentum (15%/8%) | 12 | 66.7% | +129.9% | 0.63 | 2.43 | -19.7% |
| Alt G: Nuclear energy rules (12%/7%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt H: Nuclear wide (20%/10%) | 11 | 63.6% | +135.4% | 0.62 | 2.33 | -21.3% |

**Best baseline selected for validation: Alt F: Nuclear momentum (15%/8%)**

---

## 2. Full Validation

### Alt F: Nuclear momentum (15%/8%)

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+129.9%, Trades=12, WR=66.7%, Sharpe=0.63, PF=2.43, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-0.99, Ratio=-147% (need >=50%) |
| Bootstrap | **PASS** | p=0.0174, Sharpe CI=[0.27, 12.66], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$2,445, Survival=100.0% |
| Regime | FAIL | bull:12t/+104.2% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 12 | 66.7% | +129.9% | 0.63 |
| WF tune: conf=0.55 | 12 | 66.7% | +129.9% | 0.63 |
| WF tune: conf=0.6 | 12 | 66.7% | +129.9% | 0.63 |
| WF tune: ATR stops x2.5 | 12 | 66.7% | +129.9% | 0.63 |
| WF tune: + utility_mean_reversion | 12 | 66.7% | +129.9% | 0.63 |
| WF tune: + utility_rate_reversion | 12 | 66.7% | +129.9% | 0.63 |
| WF tune: cooldown=7 | 12 | 66.7% | +126.9% | 0.63 |
| WF tune: PT=12% | 15 | 60.0% | +61.8% | 0.50 |
| WF tune: conf=0.65 | 12 | 58.3% | +83.8% | 0.46 |
| Regime tune: tighter stop 4% | 17 | 41.2% | +66.4% | 0.44 |
| WF tune: PT=8% | 18 | 66.7% | +66.2% | 0.42 |
| WF tune: PT=7% | 18 | 66.7% | +49.8% | 0.37 |
| Regime tune: full rules (10) | 27 | 40.7% | +12.2% | 0.16 |
| Regime tune: utility rules (13) | 29 | 37.9% | -3.1% | 0.06 |
| Alt F: Nuclear momentum (15%/8%) [multi-TF] | 24 | 41.7% | +21.9% | 0.21 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.45
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+129.9%, Trades=12, WR=66.7%, Sharpe=0.63, PF=2.43, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-0.99, Ratio=-147% (need >=50%) |
| Bootstrap | **PASS** | p=0.0174, Sharpe CI=[0.27, 12.66], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$2,445, Survival=100.0% |
| Regime | FAIL | bull:12t/+104.2% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.55
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+129.9%, Trades=12, WR=66.7%, Sharpe=0.63, PF=2.43, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-0.99, Ratio=-147% (need >=50%) |
| Bootstrap | **PASS** | p=0.0174, Sharpe CI=[0.27, 12.66], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$2,445, Survival=100.0% |
| Regime | FAIL | bull:12t/+104.2% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.6
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+129.9%, Trades=12, WR=66.7%, Sharpe=0.63, PF=2.43, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-0.99, Ratio=-147% (need >=50%) |
| Bootstrap | **PASS** | p=0.0174, Sharpe CI=[0.27, 12.66], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$2,445, Survival=100.0% |
| Regime | FAIL | bull:12t/+104.2% |

**Result: 2/4 gates passed**

---

### Alt F: Nuclear momentum (15%/8%) [multi-TF]

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+21.9%, Trades=24, WR=41.7%, Sharpe=0.21, PF=1.12, DD=-40.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.63, Test Sharpe=-0.80, Ratio=-127% (need >=50%) |
| Bootstrap | FAIL | p=0.2695, Sharpe CI=[-2.31, 3.95], WR CI=[20.8%, 62.5%] |
| Monte Carlo | FAIL | Ruin=0.8%, P95 DD=-50.3%, Median equity=$1,255, Survival=99.2% |
| Regime | **PASS** | bull:22t/+31.6%, chop:1t/+15.5%, volatile:1t/-8.1% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt F: Nuclear momentum (15%/8%)** | FAIL | **PASS** | **PASS** | FAIL | **0.63** | **+129.9%** | 12 |
| WF tune: conf=0.45 | FAIL | **PASS** | **PASS** | FAIL | 0.63 | +129.9% | 12 |
| WF tune: conf=0.55 | FAIL | **PASS** | **PASS** | FAIL | 0.63 | +129.9% | 12 |
| WF tune: conf=0.6 | FAIL | **PASS** | **PASS** | FAIL | 0.63 | +129.9% | 12 |
| Alt F: Nuclear momentum (15%/8%) [multi-TF] | FAIL | FAIL | FAIL | **PASS** | 0.21 | +21.9% | 24 |

---

## 5. Final Recommendation

**VST partially validates.** Best config: Alt F: Nuclear momentum (15%/8%) (2/4 gates).

### Alt F: Nuclear momentum (15%/8%)

- **Rules:** `trend_continuation, momentum_reversal, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 8.0%
- **Cooldown:** 3 bars

**Performance:** Return=+129.9%, Trades=12, WR=66.7%, Sharpe=0.63, PF=2.43, DD=-19.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.67, Test Sharpe=-0.99, Ratio=-147% (need >=50%) |
| Bootstrap | **PASS** | p=0.0174, Sharpe CI=[0.27, 12.66], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-28.6%, Median equity=$2,445, Survival=100.0% |
| Regime | FAIL | bull:12t/+104.2% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

