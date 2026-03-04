# GDX (VanEck Gold Miners ETF) Validated Optimization Results

**Date:** 2026-03-04
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 59.7 minutes
**Category:** Gold miners ETF

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

GDX — Basket of ~50 senior gold miners — 2-3x leveraged to gold price. Gold miners ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 29 | 62.1% | +59.9% | 0.46 | 1.69 | -30.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 43 | 65.1% | +222.1% | 0.67 | 2.98 | -25.6% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 30 | 56.7% | +66.2% | 0.48 | 1.84 | -28.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 28 | 60.7% | +59.0% | 0.60 | 1.68 | -30.7% |
| Alt D: Full mining rules (14 rules, 10%/5%) | 38 | 65.8% | +263.9% | 0.70 | 3.45 | -20.6% |
| Alt E: gold_miner rules (4 rules, 10%/5%) | 29 | 62.1% | +59.9% | 0.46 | 1.69 | -30.7% |
| Alt F: Gold miner ratio (12%/5%) | 28 | 60.7% | +59.0% | 0.60 | 1.68 | -30.7% |

**Best baseline selected for validation: Alt D: Full mining rules (14 rules, 10%/5%)**

---

## 2. Full Validation

### Alt D: Full mining rules (14 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+263.9%, Trades=38, WR=65.8%, Sharpe=0.70, PF=3.45, DD=-20.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.63, Test Sharpe=1.10, Ratio=175% (need >=50%) |
| Bootstrap | **PASS** | p=0.0006, Sharpe CI=[1.36, 6.93], WR CI=[50.0%, 81.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.8%, Median equity=$4,244, Survival=100.0% |
| Regime | FAIL | bull:31t/+129.9%, bear:1t/+10.2%, chop:2t/+6.6%, volatile:3t/+1.8%, crisis:1t/+10.9% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: PT=12% | 36 | 55.6% | +234.2% | 0.73 |
| Regime tune: PT=15% | 30 | 53.3% | +169.3% | 0.72 |
| Regime tune: tighter stop 4% | 48 | 54.2% | +212.9% | 0.71 |
| Regime tune: full mining rules (14) | 38 | 65.8% | +263.9% | 0.70 |
| Regime tune: full rules (10) | 43 | 65.1% | +222.1% | 0.67 |
| Regime tune: conf=0.65 | 33 | 63.6% | +310.3% | 0.61 |
| Regime tune: PT=12% [multi-TF] | 83 | 53.0% | +128.4% | 0.55 |

### Full Validation of Top Candidates

### Regime tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+234.2%, Trades=36, WR=55.6%, Sharpe=0.73, PF=2.34, DD=-24.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.36, Test Sharpe=1.00, Ratio=74% (need >=50%) |
| Bootstrap | **PASS** | p=0.0031, Sharpe CI=[0.92, 6.02], WR CI=[41.7%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.6%, Median equity=$3,840, Survival=100.0% |
| Regime | FAIL | bull:28t/+116.6%, bear:2t/+7.7%, chop:3t/+20.2%, volatile:3t/+7.8% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+169.3%, Trades=30, WR=53.3%, Sharpe=0.72, PF=1.83, DD=-26.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.11, Test Sharpe=0.70, Ratio=63% (need >=50%) |
| Bootstrap | **PASS** | p=0.0163, Sharpe CI=[0.25, 5.64], WR CI=[40.0%, 73.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.1%, Median equity=$3,021, Survival=100.0% |
| Regime | FAIL | bull:24t/+116.5%, bear:2t/-13.4%, chop:1t/+15.3%, volatile:3t/+11.5% |

**Result: 3/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=+212.9%, Trades=48, WR=54.2%, Sharpe=0.71, PF=2.73, DD=-27.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.41, Test Sharpe=1.10, Ratio=267% (need >=50%) |
| Bootstrap | **PASS** | p=0.0035, Sharpe CI=[0.78, 5.06], WR CI=[39.6%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.8%, Median equity=$3,679, Survival=100.0% |
| Regime | FAIL | bull:38t/+115.6%, bear:2t/+5.8%, chop:4t/+6.3%, volatile:3t/+24.5%, crisis:1t/-6.5% |

**Result: 3/4 gates passed**

---

### Regime tune: PT=12% [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+128.4%, Trades=83, WR=53.0%, Sharpe=0.55, PF=1.96, DD=-27.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.06, Test Sharpe=0.84, Ratio=1367% (need >=50%) |
| Bootstrap | **PASS** | p=0.0117, Sharpe CI=[0.26, 3.10], WR CI=[44.6%, 66.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.9%, Median equity=$2,834, Survival=100.0% |
| Regime | FAIL | bull:59t/+118.4%, bear:10t/-8.0%, chop:5t/-0.7%, volatile:8t/-4.0%, crisis:1t/+12.1% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: PT=12%** | **PASS** | **PASS** | **PASS** | FAIL | **0.73** | **+234.2%** | 36 |
| Regime tune: PT=15% | **PASS** | **PASS** | **PASS** | FAIL | 0.72 | +169.3% | 30 |
| Regime tune: tighter stop 4% | **PASS** | **PASS** | **PASS** | FAIL | 0.71 | +212.9% | 48 |
| Alt D: Full mining rules (14 rules, 10%/5%) | **PASS** | **PASS** | **PASS** | FAIL | 0.70 | +263.9% | 38 |
| Regime tune: PT=12% [multi-TF] | **PASS** | **PASS** | **PASS** | FAIL | 0.55 | +128.4% | 83 |

---

## 5. Final Recommendation

**GDX partially validates.** Best config: Regime tune: PT=12% (3/4 gates).

### Regime tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, miner_metal_ratio, volume_breakout, dollar_weakness`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+234.2%, Trades=36, WR=55.6%, Sharpe=0.73, PF=2.34, DD=-24.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=1.36, Test Sharpe=1.00, Ratio=74% (need >=50%) |
| Bootstrap | **PASS** | p=0.0031, Sharpe CI=[0.92, 6.02], WR CI=[41.7%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.6%, Median equity=$3,840, Survival=100.0% |
| Regime | FAIL | bull:28t/+116.6%, bear:2t/+7.7%, chop:3t/+20.2%, volatile:3t/+7.8% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

