# EOG (EOG Resources) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 12.5 minutes
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

EOG — Premium US shale E&P — Eagle Ford, Permian, Powder River Basin. Large-cap E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 11 | 9.1% | -41.5% | -0.97 | 0.18 | -47.2% |
| Alt A: Full general rules (10 rules, 10%/5%) | 22 | 27.3% | -26.9% | -0.63 | 0.61 | -42.1% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 12 | 8.3% | -31.7% | -0.75 | 0.22 | -39.4% |
| Alt C: Wider PT (3 rules, 12%/5%) | 11 | 9.1% | -37.9% | -0.76 | 0.29 | -46.9% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 22 | 27.3% | -26.9% | -0.63 | 0.61 | -42.1% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-26.9%, Trades=22, WR=27.3%, Sharpe=-0.63, PF=0.61, DD=-42.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.69, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.7925, Sharpe CI=[-5.90, 1.67], WR CI=[9.1%, 45.5%] |
| Monte Carlo | FAIL | Ruin=0.8%, P95 DD=-48.0%, Median equity=$724, Survival=99.2% |
| Regime | FAIL | bull:15t/-22.4%, bear:3t/-15.7%, chop:2t/-10.8%, volatile:2t/+22.2% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=8% | 21 | 28.6% | -25.9% | -0.53 |
| WF tune: PT=12% | 13 | 15.4% | -27.6% | -0.58 |
| WF tune: PT=15% | 11 | 9.1% | -28.3% | -0.59 |
| MC tune: max_loss=3.0% | 31 | 19.4% | -30.3% | -0.60 |
| MC tune: max_loss=4.0% | 22 | 18.2% | -35.2% | -0.63 |
| WF tune: conf=0.45 | 22 | 27.3% | -26.9% | -0.63 |
| BS tune: conf=0.4 | 22 | 27.3% | -26.9% | -0.63 |
| BS tune: full rules (10) | 22 | 27.3% | -26.9% | -0.63 |
| BS tune: energy rules (12) | 22 | 27.3% | -26.9% | -0.63 |
| BS tune: + volume_breakout | 22 | 27.3% | -26.9% | -0.63 |
| BS tune: + commodity_breakout | 22 | 27.3% | -26.9% | -0.63 |
| Regime tune: + dollar_weakness | 22 | 27.3% | -26.9% | -0.63 |
| WF tune: cooldown=7 | 17 | 11.8% | -46.3% | -0.81 |
| WF tune: conf=0.65 | 10 | 10.0% | -29.7% | -1.11 |
| WF tune: conf=0.6 | 15 | 13.3% | -37.2% | -1.13 |
| WF tune: conf=0.55 | 17 | 17.6% | -39.3% | -1.15 |

### Full Validation of Top Candidates

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-25.9%, Trades=21, WR=28.6%, Sharpe=-0.53, PF=0.60, DD=-38.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.50, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8082, Sharpe CI=[-6.20, 1.70], WR CI=[9.5%, 47.6%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-45.2%, Median equity=$737, Survival=99.9% |
| Regime | FAIL | bull:15t/-26.3%, bear:2t/-8.4%, chop:2t/-10.8%, volatile:2t/+19.8% |

**Result: 0/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-27.6%, Trades=13, WR=15.4%, Sharpe=-0.58, PF=0.44, DD=-39.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.59, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8815, Sharpe CI=[-30.69, 1.35], WR CI=[0.0%, 38.5%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-43.9%, Median equity=$696, Survival=100.0% |
| Regime | FAIL | bull:9t/-35.0%, bear:2t/-8.3%, chop:1t/-7.0%, volatile:1t/+18.1% |

**Result: 0/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-28.3%, Trades=11, WR=9.1%, Sharpe=-0.59, PF=0.35, DD=-39.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.59, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9252, Sharpe CI=[-28.86, 1.02], WR CI=[0.0%, 27.3%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.9%, Median equity=$686, Survival=100.0% |
| Regime | FAIL | bull:7t/-37.3%, bear:2t/-8.3%, chop:1t/-7.0%, volatile:1t/+18.1% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: PT=8%** | FAIL | FAIL | FAIL | FAIL | **-0.53** | **-25.9%** | 21 |
| WF tune: PT=12% | FAIL | FAIL | FAIL | FAIL | -0.58 | -27.6% | 13 |
| WF tune: PT=15% | FAIL | FAIL | FAIL | FAIL | -0.59 | -28.3% | 11 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | -0.63 | -26.9% | 22 |

---

## 5. Final Recommendation

**EOG partially validates.** Best config: WF tune: PT=8% (0/4 gates).

### WF tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-25.9%, Trades=21, WR=28.6%, Sharpe=-0.53, PF=0.60, DD=-38.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.50, Test Sharpe=0.00, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8082, Sharpe CI=[-6.20, 1.70], WR CI=[9.5%, 47.6%] |
| Monte Carlo | FAIL | Ruin=0.1%, P95 DD=-45.2%, Median equity=$737, Survival=99.9% |
| Regime | FAIL | bull:15t/-26.3%, bear:2t/-8.4%, chop:2t/-10.8%, volatile:2t/+19.8% |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

