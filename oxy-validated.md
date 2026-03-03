# OXY (Occidental Petroleum) Validated Optimization Results

**Date:** 2026-03-02
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 16.7 minutes
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

OXY — E&P focused, Permian Basin, Buffett-backed, carbon capture. Large-cap E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 5 | 0.0% | -25.9% | -1.17 | 0.00 | -25.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 17 | 11.8% | -45.2% | -1.35 | 0.20 | -47.8% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 5 | 0.0% | -23.5% | -1.19 | 0.00 | -23.5% |
| Alt C: Wider PT (3 rules, 12%/5%) | 5 | 0.0% | -25.9% | -1.17 | 0.00 | -25.9% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 17 | 11.8% | -45.2% | -1.35 | 0.20 | -47.8% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-45.2%, Trades=17, WR=11.8%, Sharpe=-1.35, PF=0.20, DD=-47.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.83, Test Sharpe=-0.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9958, Sharpe CI=[-24.67, -1.08], WR CI=[0.0%, 35.3%] |
| Monte Carlo | FAIL | Ruin=9.4%, P95 DD=-51.8%, Median equity=$536, Survival=90.5% |
| Regime | FAIL | bull:12t/-53.9%, bear:1t/+4.2%, volatile:4t/-9.6% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: conf=0.65 | 12 | 8.3% | -39.1% | -1.04 |
| WF tune: cooldown=7 | 13 | 15.4% | -32.7% | -1.18 |
| WF tune: conf=0.6 | 17 | 11.8% | -43.2% | -1.19 |
| WF tune: conf=0.55 | 19 | 10.5% | -45.7% | -1.22 |
| MC tune: max_loss=3.0% | 23 | 8.7% | -48.5% | -1.29 |
| MC tune: max_loss=4.0% | 21 | 9.5% | -49.8% | -1.31 |
| WF tune: PT=15% | 18 | 11.1% | -44.3% | -1.32 |
| WF tune: PT=12% | 19 | 10.5% | -51.3% | -1.34 |
| WF tune: conf=0.45 | 17 | 11.8% | -45.2% | -1.35 |
| BS tune: conf=0.4 | 17 | 11.8% | -45.2% | -1.35 |
| BS tune: full rules (10) | 17 | 11.8% | -45.2% | -1.35 |
| BS tune: energy rules (12) | 17 | 11.8% | -45.2% | -1.35 |
| BS tune: + volume_breakout | 17 | 11.8% | -45.2% | -1.35 |
| BS tune: + commodity_breakout | 17 | 11.8% | -45.2% | -1.35 |
| Regime tune: + dollar_weakness | 17 | 11.8% | -45.2% | -1.35 |
| WF tune: PT=8% | 20 | 15.0% | -48.1% | -1.36 |

### Full Validation of Top Candidates

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-39.1%, Trades=12, WR=8.3%, Sharpe=-1.04, PF=0.05, DD=-41.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.01, Test Sharpe=-0.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9999, Sharpe CI=[-46.14, -3.27], WR CI=[0.0%, 41.7%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-42.2%, Median equity=$595, Survival=100.0% |
| Regime | FAIL | bull:8t/-31.6%, bear:1t/+3.0%, volatile:3t/-21.3% |

**Result: 0/4 gates passed**

---

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=-32.7%, Trades=13, WR=15.4%, Sharpe=-1.18, PF=0.27, DD=-40.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.77, Test Sharpe=-0.63, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9665, Sharpe CI=[-45.29, 0.28], WR CI=[0.0%, 46.2%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.7%, Median equity=$669, Survival=100.0% |
| Regime | FAIL | bull:9t/-39.8%, bear:1t/+4.2%, volatile:3t/-2.3% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.6

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-43.2%, Trades=17, WR=11.8%, Sharpe=-1.19, PF=0.07, DD=-45.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.44, Test Sharpe=-0.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9999, Sharpe CI=[-13.35, -2.84], WR CI=[5.9%, 47.1%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-46.6%, Median equity=$561, Survival=100.0% |
| Regime | FAIL | bull:11t/-39.4%, bear:3t/+5.0%, volatile:3t/-21.3% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: conf=0.65** | FAIL | FAIL | FAIL | FAIL | **-1.04** | **-39.1%** | 12 |
| WF tune: cooldown=7 | FAIL | FAIL | FAIL | FAIL | -1.18 | -32.7% | 13 |
| WF tune: conf=0.6 | FAIL | FAIL | FAIL | FAIL | -1.19 | -43.2% | 17 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | -1.35 | -45.2% | 17 |

---

## 5. Final Recommendation

**OXY partially validates.** Best config: WF tune: conf=0.65 (0/4 gates).

### WF tune: conf=0.65

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-39.1%, Trades=12, WR=8.3%, Sharpe=-1.04, PF=0.05, DD=-41.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.01, Test Sharpe=-0.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9999, Sharpe CI=[-46.14, -3.27], WR CI=[0.0%, 41.7%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-42.2%, Median equity=$595, Survival=100.0% |
| Regime | FAIL | bull:8t/-31.6%, bear:1t/+3.0%, volatile:3t/-21.3% |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

