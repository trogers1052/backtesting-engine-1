# KMB (Kimberly-Clark) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 23.4 minutes
**Category:** Household products (Dividend King)

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

KMB — Household products — beta 0.08-0.31, yield ~5%, Dividend King. Household products (Dividend King).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 14.3% | -19.7% | -1.11 | 0.27 | -20.3% |
| Alt A: Full general rules (10 rules, 10%/5%) | 8 | 37.5% | -11.8% | -0.52 | 0.42 | -15.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 7 | 14.3% | -15.3% | -0.96 | 0.36 | -15.9% |
| Alt C: Wider PT (3 rules, 12%/5%) | 6 | 0.0% | -23.4% | -1.26 | 0.00 | -24.0% |
| Alt D: Staples rules (13 rules, 10%/5%) | 19 | 31.6% | -19.8% | -0.77 | 0.43 | -21.8% |
| Alt E: household lean (3 rules, 10%/5%) | 12 | 25.0% | -12.3% | -0.57 | 0.48 | -22.1% |
| Alt F: Household tight (6%/3%, conf=0.55) | 14 | 28.6% | -8.4% | -0.46 | 0.62 | -15.8% |
| Alt G: Household moderate (7%/4%) | 15 | 26.7% | -12.2% | -0.50 | 0.55 | -18.9% |

**Best baseline selected for validation: Alt F: Household tight (6%/3%, conf=0.55)**

---

## 2. Full Validation

### Alt F: Household tight (6%/3%, conf=0.55)

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=-8.4%, Trades=14, WR=28.6%, Sharpe=-0.46, PF=0.62, DD=-15.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.68, Test Sharpe=-0.86, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6876, Sharpe CI=[-6.24, 2.89], WR CI=[14.3%, 64.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.1%, Median equity=$926, Survival=100.0% |
| Regime | FAIL | bull:11t/+7.4%, bear:1t/-4.4%, chop:1t/-3.0%, volatile:1t/-6.6% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 9 | 55.6% | +11.9% | 0.22 |
| BS tune: full rules (10) | 17 | 35.3% | -2.8% | -0.16 |
| WF tune: PT=15% | 13 | 23.1% | -6.2% | -0.33 |
| WF tune: conf=0.6 | 13 | 30.8% | -0.9% | -0.33 |
| WF tune: PT=7% | 14 | 28.6% | -7.7% | -0.40 |
| WF tune: cooldown=3 | 15 | 40.0% | -5.5% | -0.44 |
| WF tune: conf=0.45 | 14 | 28.6% | -8.4% | -0.46 |
| WF tune: ATR stops x2.5 | 14 | 28.6% | -8.4% | -0.46 |
| BS tune: conf=0.4 | 14 | 28.6% | -8.4% | -0.46 |
| BS tune: staples rules (13) | 26 | 34.6% | -15.9% | -0.48 |
| WF tune: PT=12% | 13 | 23.1% | -9.4% | -0.52 |
| WF tune: PT=8% | 13 | 23.1% | -11.4% | -0.63 |
| WF tune: + staples_pullback | 16 | 25.0% | -14.4% | -0.66 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.65
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+11.9%, Trades=9, WR=55.6%, Sharpe=0.22, PF=1.97, DD=-7.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=-0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1440, Sharpe CI=[-2.58, 9.01], WR CI=[33.3%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.6%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:7t/+16.7%, bear:1t/-4.4%, chop:1t/+2.2% |

**Result: 1/4 gates passed**

---

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 3 bars

**Performance:** Return=-2.8%, Trades=17, WR=35.3%, Sharpe=-0.16, PF=0.91, DD=-16.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.56, Test Sharpe=-3.08, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.4802, Sharpe CI=[-4.38, 3.45], WR CI=[11.8%, 58.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.2%, Median equity=$1,002, Survival=100.0% |
| Regime | FAIL | bull:11t/+1.2%, bear:2t/+1.2%, chop:2t/-5.0%, volatile:1t/-2.7%, crisis:1t/+7.3% |

**Result: 1/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.55
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=-6.2%, Trades=13, WR=23.1%, Sharpe=-0.33, PF=0.71, DD=-15.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.46, Test Sharpe=-0.86, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6207, Sharpe CI=[-8.64, 2.78], WR CI=[7.7%, 53.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.2%, Median equity=$940, Survival=100.0% |
| Regime | FAIL | bull:10t/+9.5%, bear:1t/-4.4%, chop:1t/-3.0%, volatile:1t/-6.6% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | FAIL | FAIL | **PASS** | FAIL | **0.22** | **+11.9%** | 9 |
| BS tune: full rules (10) | FAIL | FAIL | **PASS** | FAIL | -0.16 | -2.8% | 17 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | -0.33 | -6.2% | 13 |
| Alt F: Household tight (6%/3%, conf=0.55) | FAIL | FAIL | **PASS** | FAIL | -0.46 | -8.4% | 14 |

---

## 5. Final Recommendation

**KMB partially validates.** Best config: WF tune: conf=0.65 (1/4 gates).

### WF tune: conf=0.65

- **Rules:** `consumer_staples_mean_reversion, consumer_staples_seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.65
- **Max Loss:** 3.0%
- **Cooldown:** 7 bars

**Performance:** Return=+11.9%, Trades=9, WR=55.6%, Sharpe=0.22, PF=1.97, DD=-7.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.24, Test Sharpe=-0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1440, Sharpe CI=[-2.58, 9.01], WR CI=[33.3%, 88.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-10.6%, Median equity=$1,145, Survival=100.0% |
| Regime | FAIL | bull:7t/+16.7%, bear:1t/-4.4%, chop:1t/+2.2% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

