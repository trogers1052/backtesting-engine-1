# XLK (Technology Select Sector SPDR ETF) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 67.2 minutes
**Category:** Tech sector ETF

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

XLK — Broad tech sector basket — AAPL, MSFT, NVDA top holdings. Tech sector ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 10 | 60.0% | +36.5% | 0.48 | 2.20 | -10.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 14 | 57.1% | +68.8% | 0.56 | 3.07 | -15.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 11 | 45.5% | +17.6% | 0.27 | 1.54 | -13.1% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 55.6% | +44.7% | 0.57 | 3.37 | -12.5% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 10 | 60.0% | +36.5% | 0.48 | 2.20 | -10.2% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+68.8%, Trades=14, WR=57.1%, Sharpe=0.56, PF=3.07, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.74, 10.45], WR CI=[28.6%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.6%, Median equity=$1,808, Survival=100.0% |
| Regime | FAIL | bull:12t/+56.7%, chop:2t/+7.5% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.45 | 14 | 57.1% | +68.8% | 0.56 |
| Regime tune: + volume_breakout | 14 | 57.1% | +68.8% | 0.56 |
| Regime tune: full rules (10) | 14 | 57.1% | +68.8% | 0.56 |
| WF tune: PT=8% | 19 | 52.6% | +39.0% | 0.42 |
| Regime tune: tighter stop 4% | 18 | 44.4% | +45.7% | 0.41 |
| WF tune: cooldown=7 | 17 | 47.1% | +34.1% | 0.36 |
| WF tune: PT=12% | 14 | 42.9% | +31.3% | 0.34 |
| WF tune: PT=15% | 10 | 40.0% | +27.6% | 0.28 |
| WF tune: conf=0.65 | 12 | 50.0% | +24.3% | 0.25 |
| WF tune: conf=0.55 | 13 | 53.8% | +22.7% | 0.23 |
| WF tune: conf=0.6 | 13 | 53.8% | +21.2% | 0.21 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | 42 | 50.0% | +44.9% | 0.48 |

### Full Validation of Top Candidates

### WF tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+68.8%, Trades=14, WR=57.1%, Sharpe=0.56, PF=3.07, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.74, 10.45], WR CI=[28.6%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.6%, Median equity=$1,808, Survival=100.0% |
| Regime | FAIL | bull:12t/+56.7%, chop:2t/+7.5% |

**Result: 2/4 gates passed**

---

### Regime tune: + volume_breakout

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, volume_breakout`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+68.8%, Trades=14, WR=57.1%, Sharpe=0.56, PF=3.07, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.74, 10.45], WR CI=[28.6%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.6%, Median equity=$1,808, Survival=100.0% |
| Regime | FAIL | bull:12t/+56.7%, chop:2t/+7.5% |

**Result: 2/4 gates passed**

---

### Regime tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+68.8%, Trades=14, WR=57.1%, Sharpe=0.56, PF=3.07, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.74, 10.45], WR CI=[28.6%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.6%, Median equity=$1,808, Survival=100.0% |
| Regime | FAIL | bull:12t/+56.7%, chop:2t/+7.5% |

**Result: 2/4 gates passed**

---

### Alt A: Full general rules (10 rules, 10%/5%) [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+44.9%, Trades=42, WR=50.0%, Sharpe=0.48, PF=1.93, DD=-15.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.50, Test Sharpe=0.30, Ratio=60% (need >=50%) |
| Bootstrap | FAIL | p=0.0381, Sharpe CI=[-0.24, 3.78], WR CI=[38.1%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.9%, Median equity=$1,630, Survival=100.0% |
| Regime | FAIL | bull:15t/+46.8%, bear:8t/+1.6%, chop:11t/+8.6%, volatile:8t/-3.5% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt A: Full general rules (10 rules, 10%/5%)** | FAIL | **PASS** | **PASS** | FAIL | **0.56** | **+68.8%** | 14 |
| WF tune: conf=0.45 | FAIL | **PASS** | **PASS** | FAIL | 0.56 | +68.8% | 14 |
| Regime tune: + volume_breakout | FAIL | **PASS** | **PASS** | FAIL | 0.56 | +68.8% | 14 |
| Regime tune: full rules (10) | FAIL | **PASS** | **PASS** | FAIL | 0.56 | +68.8% | 14 |
| Alt A: Full general rules (10 rules, 10%/5%) [multi-TF] | **PASS** | FAIL | **PASS** | FAIL | 0.48 | +44.9% | 42 |

---

## 5. Final Recommendation

**XLK partially validates.** Best config: Alt A: Full general rules (10 rules, 10%/5%) (2/4 gates).

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+68.8%, Trades=14, WR=57.1%, Sharpe=0.56, PF=3.07, DD=-15.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=0.78, Test Sharpe=0.27, Ratio=35% (need >=50%) |
| Bootstrap | **PASS** | p=0.0086, Sharpe CI=[0.74, 10.45], WR CI=[28.6%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-15.6%, Median equity=$1,808, Survival=100.0% |
| Regime | FAIL | bull:12t/+56.7%, chop:2t/+7.5% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

