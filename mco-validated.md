# MCO (Moody's Corporation) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 64.8 minutes
**Category:** Financial data/ratings

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

MCO — Credit ratings, research, risk analytics — duopoly with S&P Global. Financial data/ratings.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 12 | 33.3% | -5.4% | -0.53 | 0.85 | -22.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 43.8% | +10.9% | 0.13 | 1.12 | -23.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 13 | 30.8% | -7.4% | -0.40 | 0.79 | -17.7% |
| Alt C: Smaller PT (3 rules, 8%/5%) | 11 | 63.6% | +26.0% | 0.44 | 2.05 | -10.8% |
| Alt D: Financial-specific rules (12 rules, 10%/5%) | 18 | 50.0% | +15.2% | 0.19 | 1.20 | -20.4% |
| Alt E: Financial lean rules (3 sector rules, 10%/5%) | 12 | 50.0% | +11.3% | 0.14 | 1.33 | -13.4% |

**Best baseline selected for validation: Alt C: Smaller PT (3 rules, 8%/5%)**

---

## 2. Full Validation

### Alt C: Smaller PT (3 rules, 8%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.0%, Trades=11, WR=63.6%, Sharpe=0.44, PF=2.05, DD=-10.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-1.49, Ratio=-195% (need >=50%) |
| Bootstrap | FAIL | p=0.0625, Sharpe CI=[-0.90, 12.44], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.0%, Median equity=$1,399, Survival=100.0% |
| Regime | FAIL | bull:11t/+36.9% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 11 | 63.6% | +26.1% | 0.44 |
| WF tune: conf=0.45 | 11 | 63.6% | +26.0% | 0.44 |
| WF tune: conf=0.55 | 11 | 63.6% | +26.0% | 0.44 |
| WF tune: conf=0.6 | 11 | 63.6% | +26.0% | 0.44 |
| BS tune: conf=0.4 | 11 | 63.6% | +26.0% | 0.44 |
| BS tune: + volume_breakout | 11 | 63.6% | +26.0% | 0.44 |
| BS tune: + financial_mean_reversion | 11 | 63.6% | +26.0% | 0.44 |
| Regime tune: + financial_seasonality | 13 | 61.5% | +22.1% | 0.33 |
| WF tune: cooldown=7 | 13 | 53.8% | +15.5% | 0.32 |
| WF tune: PT=15% | 8 | 37.5% | +12.2% | 0.16 |
| BS tune: financial rules (12) | 22 | 50.0% | +9.0% | 0.09 |
| BS tune: full rules (10) | 20 | 45.0% | +8.0% | 0.08 |
| WF tune: cooldown=3 | 13 | 46.2% | -0.5% | -0.17 |
| Regime tune: tighter stop 4% | 14 | 42.9% | -0.1% | -0.17 |
| WF tune: PT=12% | 10 | 30.0% | -3.6% | -0.18 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.1%, Trades=11, WR=63.6%, Sharpe=0.44, PF=2.07, DD=-10.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-1.45, Ratio=-190% (need >=50%) |
| Bootstrap | FAIL | p=0.0609, Sharpe CI=[-0.86, 12.62], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.7%, Median equity=$1,402, Survival=100.0% |
| Regime | FAIL | bull:11t/+37.1% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.45

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.0%, Trades=11, WR=63.6%, Sharpe=0.44, PF=2.05, DD=-10.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-1.49, Ratio=-195% (need >=50%) |
| Bootstrap | FAIL | p=0.0625, Sharpe CI=[-0.90, 12.44], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.0%, Median equity=$1,399, Survival=100.0% |
| Regime | FAIL | bull:11t/+36.9% |

**Result: 1/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.0%, Trades=11, WR=63.6%, Sharpe=0.44, PF=2.05, DD=-10.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-1.49, Ratio=-195% (need >=50%) |
| Bootstrap | FAIL | p=0.0625, Sharpe CI=[-0.90, 12.44], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-18.0%, Median equity=$1,399, Survival=100.0% |
| Regime | FAIL | bull:11t/+36.9% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | FAIL | FAIL | **PASS** | FAIL | **0.44** | **+26.1%** | 11 |
| Alt C: Smaller PT (3 rules, 8%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.44 | +26.0% | 11 |
| WF tune: conf=0.45 | FAIL | FAIL | **PASS** | FAIL | 0.44 | +26.0% | 11 |
| WF tune: conf=0.55 | FAIL | FAIL | **PASS** | FAIL | 0.44 | +26.0% | 11 |

---

## 5. Final Recommendation

**MCO partially validates.** Best config: WF tune: conf=0.65 (1/4 gates).

### WF tune: conf=0.65

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.1%, Trades=11, WR=63.6%, Sharpe=0.44, PF=2.07, DD=-10.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.76, Test Sharpe=-1.45, Ratio=-190% (need >=50%) |
| Bootstrap | FAIL | p=0.0609, Sharpe CI=[-0.86, 12.62], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-17.7%, Median equity=$1,402, Survival=100.0% |
| Regime | FAIL | bull:11t/+37.1% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

