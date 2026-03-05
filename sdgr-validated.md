# SDGR (Schrödinger) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 21.6 minutes
**Category:** Tech / Biotech (computational platform)

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

SDGR — Computational drug discovery — beta ~1.8, physics-based molecular simulation. Tech / Biotech (computational platform).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 5 | 20.0% | -22.9% | -0.81 | 0.34 | -31.3% |
| Alt A: Full general rules (10 rules, 10%/5%) | 16 | 43.8% | +30.1% | 0.31 | 1.30 | -36.4% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 5 | 20.0% | -11.9% | -0.43 | 0.50 | -21.6% |
| Alt C: Wider PT (3 rules, 12%/5%) | 5 | 20.0% | -22.9% | -0.81 | 0.34 | -31.3% |
| Alt D: Recommended rules (4 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt E: Tech full (13 rules, 10%/5%) | 16 | 43.8% | +30.1% | 0.31 | 1.30 | -36.4% |
| Alt F: Tech + healthcare seasonality (10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt G: Wide momentum (15%/8%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.1%, Trades=16, WR=43.8%, Sharpe=0.31, PF=1.30, DD=-36.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.51, Test Sharpe=0.30, Ratio=59% (need >=50%) |
| Bootstrap | FAIL | p=0.1936, Sharpe CI=[-2.20, 5.44], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.9%, Median equity=$1,381, Survival=100.0% |
| Regime | **PASS** | bull:11t/+34.8%, chop:3t/-4.6%, volatile:1t/-5.8%, crisis:1t/+20.0% |

**Result: 3/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: cooldown=7 | 11 | 45.5% | +26.0% | 0.44 |
| BS tune: conf=0.4 | 16 | 43.8% | +30.1% | 0.31 |
| BS tune: conf=0.45 | 16 | 43.8% | +30.1% | 0.31 |
| BS tune: full rules (10) | 16 | 43.8% | +30.1% | 0.31 |
| BS tune: conf=0.55 | 12 | 41.7% | +11.5% | 0.13 |
| BS tune: recommended rules | 0 | 0.0% | +0.0% | 0.00 |
| BS tune: cooldown=7 [multi-TF] | 60 | 41.7% | +1.6% | -0.02 |

### Full Validation of Top Candidates

### BS tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+26.0%, Trades=11, WR=45.5%, Sharpe=0.44, PF=1.63, DD=-22.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.30, Ratio=66% (need >=50%) |
| Bootstrap | FAIL | p=0.1667, Sharpe CI=[-2.70, 6.56], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.5%, Median equity=$1,309, Survival=100.0% |
| Regime | **PASS** | bull:7t/+12.1%, bear:1t/-6.4%, chop:2t/+6.5%, crisis:1t/+20.0% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.1%, Trades=16, WR=43.8%, Sharpe=0.31, PF=1.30, DD=-36.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.51, Test Sharpe=0.30, Ratio=59% (need >=50%) |
| Bootstrap | FAIL | p=0.1936, Sharpe CI=[-2.20, 5.44], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.9%, Median equity=$1,381, Survival=100.0% |
| Regime | **PASS** | bull:11t/+34.8%, chop:3t/-4.6%, volatile:1t/-5.8%, crisis:1t/+20.0% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.45

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.45
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.1%, Trades=16, WR=43.8%, Sharpe=0.31, PF=1.30, DD=-36.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.51, Test Sharpe=0.30, Ratio=59% (need >=50%) |
| Bootstrap | FAIL | p=0.1936, Sharpe CI=[-2.20, 5.44], WR CI=[18.8%, 68.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.9%, Median equity=$1,381, Survival=100.0% |
| Regime | **PASS** | bull:11t/+34.8%, chop:3t/-4.6%, volatile:1t/-5.8%, crisis:1t/+20.0% |

**Result: 3/4 gates passed**

---

### BS tune: cooldown=7 [multi-TF]

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+1.6%, Trades=60, WR=41.7%, Sharpe=-0.02, PF=1.02, DD=-37.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.11, Test Sharpe=0.55, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.3224, Sharpe CI=[-1.51, 2.13], WR CI=[31.7%, 58.3%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.4%, Median equity=$1,139, Survival=100.0% |
| Regime | **PASS** | bull:49t/+3.3%, bear:4t/+6.5%, chop:1t/+10.4%, volatile:5t/-11.4%, crisis:1t/+9.4% |

**Result: 2/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: cooldown=7** | **PASS** | FAIL | **PASS** | **PASS** | **0.44** | **+26.0%** | 11 |
| Alt A: Full general rules (10 rules, 10%/5%) | **PASS** | FAIL | **PASS** | **PASS** | 0.31 | +30.1% | 16 |
| BS tune: conf=0.4 | **PASS** | FAIL | **PASS** | **PASS** | 0.31 | +30.1% | 16 |
| BS tune: conf=0.45 | **PASS** | FAIL | **PASS** | **PASS** | 0.31 | +30.1% | 16 |
| BS tune: cooldown=7 [multi-TF] | FAIL | FAIL | **PASS** | **PASS** | -0.02 | +1.6% | 60 |

---

## 5. Final Recommendation

**SDGR partially validates.** Best config: BS tune: cooldown=7 (3/4 gates).

### BS tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+26.0%, Trades=11, WR=45.5%, Sharpe=0.44, PF=1.63, DD=-22.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.46, Test Sharpe=0.30, Ratio=66% (need >=50%) |
| Bootstrap | FAIL | p=0.1667, Sharpe CI=[-2.70, 6.56], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-24.5%, Median equity=$1,309, Survival=100.0% |
| Regime | **PASS** | bull:7t/+12.1%, bear:1t/-6.4%, chop:2t/+6.5%, crisis:1t/+20.0% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

