# APH (Amphenol) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 41.5 minutes
**Category:** Low-beta tech/industrial

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

APH — Electronic connectors/sensors — low-beta tech compounder, data center + defense. Low-beta tech/industrial.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 14 | 57.1% | +54.4% | 0.47 | 2.28 | -17.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 27 | 59.3% | +148.9% | 0.92 | 2.61 | -15.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 15 | 60.0% | +75.9% | 0.66 | 2.79 | -12.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 13 | 53.8% | +58.5% | 0.54 | 2.27 | -17.8% |
| Alt D: Tech rules (13 rules, 10%/5%) | 26 | 65.4% | +184.2% | 1.05 | 2.99 | -15.3% |
| Alt E: low_beta_tech rules (4 rules, 10%/5%) | 17 | 58.8% | +83.9% | 1.09 | 2.71 | -15.5% |
| Alt F: Low-beta tight (8%/4%) | 21 | 66.7% | +107.0% | 0.88 | 3.39 | -15.9% |

**Best baseline selected for validation: Alt E: low_beta_tech rules (4 rules, 10%/5%)**

---

## 2. Full Validation

### Alt E: low_beta_tech rules (4 rules, 10%/5%)

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+83.9%, Trades=17, WR=58.8%, Sharpe=1.09, PF=2.71, DD=-15.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.90, Test Sharpe=1.02, Ratio=114% (need >=50%) |
| Bootstrap | **PASS** | p=0.0160, Sharpe CI=[0.31, 8.50], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-21.8%, Median equity=$1,984, Survival=100.0% |
| Regime | FAIL | bull:15t/+61.2%, volatile:2t/+14.5% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: tighter stop 4% | 17 | 58.8% | +87.8% | 1.17 |
| Regime tune: tech rules (13) | 26 | 65.4% | +184.2% | 1.05 |
| Regime tune: full rules (10) | 27 | 59.3% | +148.9% | 0.92 |
| Regime tune: PT=12% | 16 | 56.2% | +75.7% | 0.76 |
| Regime tune: PT=15% | 13 | 53.8% | +65.7% | 0.63 |
| Regime tune: conf=0.65 | 13 | 53.8% | +46.7% | 0.53 |
| Regime tune: tech rules (13) [multi-TF] | 43 | 51.2% | +99.3% | 0.94 |

### Full Validation of Top Candidates

### Regime tune: tighter stop 4%

- **Rules:** `tech_ema_pullback, tech_mean_reversion, tech_seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+87.8%, Trades=17, WR=58.8%, Sharpe=1.17, PF=2.95, DD=-15.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.90, Test Sharpe=1.14, Ratio=127% (need >=50%) |
| Bootstrap | **PASS** | p=0.0124, Sharpe CI=[0.53, 8.84], WR CI=[35.3%, 82.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-20.1%, Median equity=$2,038, Survival=100.0% |
| Regime | FAIL | bull:15t/+63.7%, volatile:2t/+14.5% |

**Result: 3/4 gates passed**

---

### Regime tune: tech rules (13)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+184.2%, Trades=26, WR=65.4%, Sharpe=1.05, PF=2.99, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.11, Test Sharpe=0.87, Ratio=78% (need >=50%) |
| Bootstrap | **PASS** | p=0.0008, Sharpe CI=[1.63, 8.65], WR CI=[46.2%, 80.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.0%, Median equity=$3,215, Survival=100.0% |
| Regime | **PASS** | bull:18t/+71.0%, chop:5t/+37.2%, volatile:3t/+18.8% |

**Result: 4/4 gates passed**

---

### Regime tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+148.9%, Trades=27, WR=59.3%, Sharpe=0.92, PF=2.61, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=0.87, Ratio=104% (need >=50%) |
| Bootstrap | **PASS** | p=0.0038, Sharpe CI=[0.95, 7.36], WR CI=[40.7%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.6%, Median equity=$2,820, Survival=100.0% |
| Regime | **PASS** | bull:19t/+82.2%, bear:2t/-9.6%, chop:5t/+37.2%, volatile:1t/+4.3% |

**Result: 4/4 gates passed**

---

### Regime tune: tech rules (13) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+99.3%, Trades=43, WR=51.2%, Sharpe=0.94, PF=1.82, DD=-21.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.53, Test Sharpe=1.25, Ratio=236% (need >=50%) |
| Bootstrap | FAIL | p=0.0254, Sharpe CI=[-0.01, 4.37], WR CI=[37.2%, 67.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.5%, Median equity=$2,257, Survival=100.0% |
| Regime | **PASS** | bull:29t/+60.1%, bear:4t/-11.2%, chop:5t/+38.2%, volatile:5t/+5.6% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: tech rules (13)** | **PASS** | **PASS** | **PASS** | **PASS** | **1.05** | **+184.2%** | 26 |
| Regime tune: full rules (10) | **PASS** | **PASS** | **PASS** | **PASS** | 0.92 | +148.9% | 27 |
| Regime tune: tighter stop 4% | **PASS** | **PASS** | **PASS** | FAIL | 1.17 | +87.8% | 17 |
| Alt E: low_beta_tech rules (4 rules, 10%/5%) | **PASS** | **PASS** | **PASS** | FAIL | 1.09 | +83.9% | 17 |
| Regime tune: tech rules (13) [multi-TF] | **PASS** | FAIL | **PASS** | **PASS** | 0.94 | +99.3% | 43 |

---

## 5. Final Recommendation

**APH fully validates.** Best config: Regime tune: tech rules (13) (4/4 gates).

### Regime tune: tech rules (13)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, tech_ema_pullback, tech_mean_reversion, tech_seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+184.2%, Trades=26, WR=65.4%, Sharpe=1.05, PF=2.99, DD=-15.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.11, Test Sharpe=0.87, Ratio=78% (need >=50%) |
| Bootstrap | **PASS** | p=0.0008, Sharpe CI=[1.63, 8.65], WR CI=[46.2%, 80.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.0%, Median equity=$3,215, Survival=100.0% |
| Regime | **PASS** | bull:18t/+71.0%, chop:5t/+37.2%, volatile:3t/+18.8% |

**Result: 4/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

