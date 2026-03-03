# COP (ConocoPhillips) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 18.1 minutes
**Category:** Large-cap E&P

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

COP — Independent E&P — Permian, Eagle Ford, Bakken, Alaska, global LNG. Large-cap E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 9 | 22.2% | -20.9% | -0.54 | 0.38 | -31.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 13 | 23.1% | -31.7% | -0.60 | 0.29 | -41.5% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 10 | 20.0% | -21.3% | -0.52 | 0.38 | -32.2% |
| Alt C: Wider PT (3 rules, 12%/5%) | 9 | 22.2% | -19.7% | -0.48 | 0.42 | -31.8% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 13 | 23.1% | -31.7% | -0.60 | 0.29 | -41.5% |

**Best baseline selected for validation: Alt B: Tighter stops (3 rules, 10%/4%)**

---

## 2. Full Validation

### Alt B: Tighter stops (3 rules, 10%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-21.3%, Trades=10, WR=20.0%, Sharpe=-0.52, PF=0.38, DD=-32.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.59, Test Sharpe=0.53, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8183, Sharpe CI=[-13.78, 2.29], WR CI=[0.0%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.6%, Median equity=$789, Survival=100.0% |
| Regime | **PASS** | bull:6t/-6.8%, bear:1t/-5.0%, chop:1t/-8.0%, volatile:2t/-1.1% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=15% | 10 | 20.0% | -10.5% | -0.24 |
| BS tune: full rules (10) | 14 | 21.4% | -17.6% | -0.36 |
| BS tune: energy rules (12) | 14 | 21.4% | -17.6% | -0.36 |
| WF tune: PT=12% | 10 | 20.0% | -20.1% | -0.46 |
| WF tune: conf=0.45 | 10 | 20.0% | -21.3% | -0.52 |
| WF tune: conf=0.55 | 10 | 20.0% | -21.3% | -0.52 |
| WF tune: conf=0.6 | 10 | 20.0% | -21.3% | -0.52 |
| WF tune: conf=0.65 | 10 | 20.0% | -21.3% | -0.52 |
| WF tune: cooldown=3 | 10 | 20.0% | -21.3% | -0.52 |
| BS tune: conf=0.4 | 10 | 20.0% | -21.3% | -0.52 |
| BS tune: + volume_breakout | 10 | 20.0% | -21.3% | -0.52 |
| BS tune: + commodity_breakout | 10 | 20.0% | -21.3% | -0.52 |
| WF tune: cooldown=7 | 9 | 11.1% | -26.6% | -0.57 |
| WF tune: PT=8% | 12 | 25.0% | -23.6% | -0.72 |
| Alt B: Tighter stops (3 rules, 10%/4%) [multi-TF] | 11 | 18.2% | -11.0% | -0.32 |

### Full Validation of Top Candidates

### WF tune: PT=15%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-10.5%, Trades=10, WR=20.0%, Sharpe=-0.24, PF=0.67, DD=-28.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.94, Test Sharpe=0.57, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.5975, Sharpe CI=[-13.16, 3.51], WR CI=[0.0%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.0%, Median equity=$902, Survival=100.0% |
| Regime | FAIL | bull:6t/-3.1%, bear:1t/-4.1%, chop:1t/-8.0%, volatile:2t/+9.6% |

**Result: 1/4 gates passed**

---

### BS tune: full rules (10)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=-17.6%, Trades=14, WR=21.4%, Sharpe=-0.36, PF=0.45, DD=-35.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.03, Test Sharpe=0.55, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7940, Sharpe CI=[-11.75, 2.00], WR CI=[7.1%, 57.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.5%, Median equity=$827, Survival=100.0% |
| Regime | FAIL | bull:10t/-13.0%, bear:1t/-4.9%, volatile:3t/+1.2% |

**Result: 1/4 gates passed**

---

### BS tune: energy rules (12)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, commodity_breakout, dollar_weakness`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=-17.6%, Trades=14, WR=21.4%, Sharpe=-0.36, PF=0.45, DD=-35.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.03, Test Sharpe=0.55, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7940, Sharpe CI=[-11.75, 2.00], WR CI=[7.1%, 57.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-32.5%, Median equity=$827, Survival=100.0% |
| Regime | FAIL | bull:10t/-13.0%, bear:1t/-4.9%, volatile:3t/+1.2% |

**Result: 1/4 gates passed**

---

### Alt B: Tighter stops (3 rules, 10%/4%) [multi-TF]

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-11.0%, Trades=11, WR=18.2%, Sharpe=-0.32, PF=0.52, DD=-27.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.01, Test Sharpe=0.53, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7309, Sharpe CI=[-88.16, 2.82], WR CI=[0.0%, 54.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-25.4%, Median equity=$896, Survival=100.0% |
| Regime | FAIL | bull:7t/-6.6%, bear:1t/-3.6%, chop:1t/-4.5%, volatile:2t/+5.3% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt B: Tighter stops (3 rules, 10%/4%)** | FAIL | FAIL | **PASS** | **PASS** | **-0.52** | **-21.3%** | 10 |
| WF tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | -0.24 | -10.5% | 10 |
| Alt B: Tighter stops (3 rules, 10%/4%) [multi-TF] | FAIL | FAIL | **PASS** | FAIL | -0.32 | -11.0% | 11 |
| BS tune: full rules (10) | FAIL | FAIL | **PASS** | FAIL | -0.36 | -17.6% | 14 |
| BS tune: energy rules (12) | FAIL | FAIL | **PASS** | FAIL | -0.36 | -17.6% | 14 |

---

## 5. Final Recommendation

**COP partially validates.** Best config: Alt B: Tighter stops (3 rules, 10%/4%) (2/4 gates).

### Alt B: Tighter stops (3 rules, 10%/4%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-21.3%, Trades=10, WR=20.0%, Sharpe=-0.52, PF=0.38, DD=-32.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.59, Test Sharpe=0.53, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8183, Sharpe CI=[-13.78, 2.29], WR CI=[0.0%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-35.6%, Median equity=$789, Survival=100.0% |
| Regime | **PASS** | bull:6t/-6.8%, bear:1t/-5.0%, chop:1t/-8.0%, volatile:2t/-1.1% |

**Result: 2/4 gates passed**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions and/or reduced sizing
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

