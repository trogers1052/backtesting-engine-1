# AMD (Advanced Micro Devices) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 83.8 minutes
**Category:** Large-cap semiconductor

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

AMD — Semiconductor — CPUs, GPUs, data center AI chips. Large-cap semiconductor.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 15 | 46.7% | +23.9% | 0.24 | 1.37 | -21.7% |
| Alt A: Full general rules (10 rules, 10%/5%) | 35 | 42.9% | +44.3% | 0.34 | 1.29 | -32.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 16 | 43.8% | +27.2% | 0.27 | 1.43 | -20.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 14 | 42.9% | +27.9% | 0.27 | 1.42 | -21.7% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 15 | 46.7% | +17.5% | 0.17 | 1.29 | -20.8% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+44.3%, Trades=35, WR=42.9%, Sharpe=0.34, PF=1.29, DD=-32.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.12, Test Sharpe=0.60, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1487, Sharpe CI=[-1.19, 3.51], WR CI=[25.7%, 60.0%] |
| Monte Carlo | FAIL | Ruin=0.4%, P95 DD=-48.8%, Median equity=$1,619, Survival=99.6% |
| Regime | FAIL | bull:35t/+66.9% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 26 | 46.2% | +57.0% | 0.52 |
| WF tune: conf=0.55 | 24 | 45.8% | +47.1% | 0.48 |
| WF tune: conf=0.65 | 24 | 45.8% | +45.6% | 0.47 |
| WF tune: conf=0.6 | 24 | 45.8% | +39.7% | 0.41 |
| WF tune: PT=8% | 37 | 45.9% | +50.8% | 0.37 |
| MC tune: max_loss=3.0% | 41 | 34.1% | +59.5% | 0.37 |
| WF tune: conf=0.45 | 35 | 42.9% | +44.3% | 0.34 |
| BS tune: conf=0.4 | 35 | 42.9% | +44.3% | 0.34 |
| BS tune: full rules (10) | 35 | 42.9% | +44.3% | 0.34 |
| BS tune: + volume_breakout | 35 | 42.9% | +44.3% | 0.34 |
| MC tune: max_loss=4.0% | 36 | 41.7% | +42.7% | 0.34 |
| WF tune: PT=12% | 31 | 35.5% | +16.6% | 0.18 |
| WF tune: PT=15% | 29 | 31.0% | +9.2% | 0.10 |
| WF tune: conf=0.55 [multi-TF] | 104 | 45.2% | +59.6% | 0.43 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+57.0%, Trades=26, WR=46.2%, Sharpe=0.52, PF=1.46, DD=-29.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.82, Test Sharpe=0.33, Ratio=40% (need >=50%) |
| Bootstrap | FAIL | p=0.1187, Sharpe CI=[-1.30, 4.43], WR CI=[26.9%, 65.4%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-42.0%, Median equity=$1,676, Survival=100.0% |
| Regime | FAIL | bull:26t/+67.0% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+47.1%, Trades=24, WR=45.8%, Sharpe=0.48, PF=1.49, DD=-20.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.21, Test Sharpe=0.79, Ratio=377% (need >=50%) |
| Bootstrap | FAIL | p=0.1154, Sharpe CI=[-1.13, 4.88], WR CI=[29.2%, 70.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.1%, Median equity=$1,605, Survival=100.0% |
| Regime | FAIL | bull:24t/+59.1% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+45.6%, Trades=24, WR=45.8%, Sharpe=0.47, PF=1.48, DD=-20.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.19, Test Sharpe=0.79, Ratio=423% (need >=50%) |
| Bootstrap | FAIL | p=0.1216, Sharpe CI=[-1.19, 4.85], WR CI=[29.2%, 70.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.2%, Median equity=$1,575, Survival=100.0% |
| Regime | FAIL | bull:24t/+56.9% |

**Result: 2/4 gates passed**

---

### WF tune: conf=0.55 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+59.6%, Trades=104, WR=45.2%, Sharpe=0.43, PF=1.31, DD=-37.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.66, Ratio=145% (need >=50%) |
| Bootstrap | FAIL | p=0.0470, Sharpe CI=[-0.22, 2.41], WR CI=[38.5%, 57.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.2%, Median equity=$2,047, Survival=100.0% |
| Regime | FAIL | bull:80t/+65.1%, bear:7t/+7.1%, chop:13t/+8.5%, volatile:4t/+3.0% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.55** | **PASS** | FAIL | **PASS** | FAIL | **0.48** | **+47.1%** | 24 |
| WF tune: conf=0.65 | **PASS** | FAIL | **PASS** | FAIL | 0.47 | +45.6% | 24 |
| WF tune: conf=0.55 [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | 0.43 | +59.6% | 104 |
| WF tune: cooldown=7 | FAIL | FAIL | FAIL | FAIL | 0.52 | +57.0% | 26 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | 0.34 | +44.3% | 35 |

---

## 5. Final Recommendation

**AMD partially validates.** Best config: WF tune: conf=0.55 (2/4 gates).

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+47.1%, Trades=24, WR=45.8%, Sharpe=0.48, PF=1.49, DD=-20.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.21, Test Sharpe=0.79, Ratio=377% (need >=50%) |
| Bootstrap | FAIL | p=0.1154, Sharpe CI=[-1.13, 4.88], WR CI=[29.2%, 70.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.1%, Median equity=$1,605, Survival=100.0% |
| Regime | FAIL | bull:24t/+59.1% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

