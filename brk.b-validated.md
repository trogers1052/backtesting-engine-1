# BRK.B (Berkshire Hathaway) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 79.6 minutes
**Category:** Insurance conglomerate

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

BRK.B — Insurance conglomerate — GEICO, BNSF, equity portfolio (Buffett). Insurance conglomerate.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 55.6% | +22.6% | 0.41 | 1.86 | -16.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 11 | 54.5% | +36.0% | 0.63 | 2.89 | -13.7% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 30.8% | -1.2% | -0.26 | 0.95 | -16.9% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 10 | 70.0% | +45.0% | 0.76 | 4.92 | -10.5% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 19 | 47.4% | +27.7% | 0.49 | 1.73 | -15.9% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 13 | 46.2% | +14.9% | 0.25 | 1.63 | -16.0% |

**Best baseline selected for validation: Alt C: Smaller PT (3 rules, 8%/5%)**

---

## 2. Full Validation

### Alt C: Smaller PT (3 rules, 8%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+45.0%, Trades=10, WR=70.0%, Sharpe=0.76, PF=4.92, DD=-10.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.82, Test Sharpe=-1.45, Ratio=-176% (need >=50%) |
| Bootstrap | **PASS** | p=0.0034, Sharpe CI=[1.79, 76.76], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.9%, Median equity=$1,678, Survival=100.0% |
| Regime | FAIL | bull:7t/+42.8%, bear:1t/+8.4%, volatile:2t/+3.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: full rules (10) | 12 | 58.3% | +30.5% | 0.80 |
| WF tune: conf=0.45 | 10 | 70.0% | +45.0% | 0.76 |
| Regime tune: + financial_mean_reversion | 10 | 70.0% | +45.0% | 0.76 |
| WF tune: cooldown=3 | 10 | 70.0% | +39.3% | 0.71 |
| WF tune: conf=0.55 | 11 | 63.6% | +39.3% | 0.62 |
| WF tune: conf=0.6 | 11 | 63.6% | +39.3% | 0.62 |
| WF tune: conf=0.65 | 11 | 63.6% | +38.7% | 0.61 |
| WF tune: PT=12% | 7 | 57.1% | +34.2% | 0.59 |
| WF tune: PT=15% | 8 | 50.0% | +31.7% | 0.48 |
| Regime tune: financial rules (12) | 21 | 52.4% | +27.1% | 0.42 |
| Regime tune: + financial_seasonality | 17 | 52.9% | +28.2% | 0.39 |
| WF tune: cooldown=7 | 10 | 60.0% | +22.9% | 0.34 |
| Regime tune: tighter stop 4% | 12 | 50.0% | +24.1% | 0.32 |
| Alt C: Smaller PT (3 rules, 8%/5%) [multi-TF] | 15 | 53.3% | +26.0% | 0.57 |

### Full Validation of Top Candidates

### Regime tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.5%, Trades=12, WR=58.3%, Sharpe=0.80, PF=2.35, DD=-13.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.93, Test Sharpe=-1.08, Ratio=-117% (need >=50%) |
| Bootstrap | FAIL | p=0.0498, Sharpe CI=[-0.64, 11.14], WR CI=[41.7%, 91.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.1%, Median equity=$1,413, Survival=100.0% |
| Regime | FAIL | bull:9t/+39.0%, bear:1t/+8.6%, chop:2t/-10.3% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+45.0%, Trades=10, WR=70.0%, Sharpe=0.76, PF=4.92, DD=-10.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.82, Test Sharpe=-1.45, Ratio=-176% (need >=50%) |
| Bootstrap | **PASS** | p=0.0034, Sharpe CI=[1.79, 76.76], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.9%, Median equity=$1,678, Survival=100.0% |
| Regime | FAIL | bull:7t/+42.8%, bear:1t/+8.4%, volatile:2t/+3.5% |

**Result: 2/4 gates passed**

---

### Regime tune: + financial_mean_reversion

- **Rules:** `trend_continuation, seasonality, death_cross, financial_mean_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+45.0%, Trades=10, WR=70.0%, Sharpe=0.76, PF=4.92, DD=-10.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.82, Test Sharpe=-1.45, Ratio=-176% (need >=50%) |
| Bootstrap | **PASS** | p=0.0034, Sharpe CI=[1.79, 76.76], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.9%, Median equity=$1,678, Survival=100.0% |
| Regime | FAIL | bull:7t/+42.8%, bear:1t/+8.4%, volatile:2t/+3.5% |

**Result: 2/4 gates passed**

---

### Alt C: Smaller PT (3 rules, 8%/5%) [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.0%, Trades=15, WR=53.3%, Sharpe=0.57, PF=1.77, DD=-12.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.79, Test Sharpe=-2.30, Ratio=-293% (need >=50%) |
| Bootstrap | FAIL | p=0.0883, Sharpe CI=[-1.17, 7.37], WR CI=[33.3%, 80.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.1%, Median equity=$1,385, Survival=100.0% |
| Regime | **PASS** | bull:12t/+25.1%, bear:1t/+8.0%, chop:1t/+8.1%, volatile:1t/-5.1% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt C: Smaller PT (3 rules, 8%/5%)** | FAIL | **PASS** | **PASS** | FAIL | **0.76** | **+45.0%** | 10 |
| WF tune: conf=0.45 | FAIL | **PASS** | **PASS** | FAIL | 0.76 | +45.0% | 10 |
| Regime tune: + financial_mean_reversion | FAIL | **PASS** | **PASS** | FAIL | 0.76 | +45.0% | 10 |
| Alt C: Smaller PT (3 rules, 8%/5%) [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | 0.57 | +26.0% | 15 |
| Regime tune: full rules (10) | FAIL | FAIL | **PASS** | FAIL | 0.80 | +30.5% | 12 |

---

## 5. Final Recommendation

**BRK.B partially validates.** Best config: Alt C: Smaller PT (3 rules, 8%/5%) (2/4 gates).

### Alt C: Smaller PT (3 rules, 8%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+45.0%, Trades=10, WR=70.0%, Sharpe=0.76, PF=4.92, DD=-10.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.82, Test Sharpe=-1.45, Ratio=-176% (need >=50%) |
| Bootstrap | **PASS** | p=0.0034, Sharpe CI=[1.79, 76.76], WR CI=[50.0%, 100.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-9.9%, Median equity=$1,678, Survival=100.0% |
| Regime | FAIL | bull:7t/+42.8%, bear:1t/+8.4%, volatile:2t/+3.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

