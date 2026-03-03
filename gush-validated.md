# GUSH (Direxion Daily S&P Oil & Gas Bull 2X) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 12.5 minutes
**Category:** Leveraged energy ETF

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

GUSH — 2x leveraged oil & gas ETF — daily rebalancing, volatility decay. Leveraged energy ETF.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 7 | 14.3% | -26.7% | -1.43 | 0.29 | -30.9% |
| Alt A: Full general rules (10 rules, 10%/5%) | 29 | 24.1% | -56.6% | -0.59 | 0.41 | -73.3% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 8 | 12.5% | -26.8% | -1.21 | 0.29 | -31.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 7 | 14.3% | -25.8% | -1.33 | 0.32 | -30.9% |
| Alt D: Energy-extended rules (12 rules, 10%/5%) | 29 | 24.1% | -56.6% | -0.59 | 0.41 | -73.3% |

**Best baseline selected for validation: Alt A: Full general rules (10 rules, 10%/5%)**

---

## 2. Full Validation

### Alt A: Full general rules (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-56.6%, Trades=29, WR=24.1%, Sharpe=-0.59, PF=0.41, DD=-73.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.69, Test Sharpe=0.33, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.9595, Sharpe CI=[-6.58, 0.27], WR CI=[10.3%, 41.4%] |
| Monte Carlo | FAIL | Ruin=100.0%, P95 DD=-69.6%, Median equity=$419, Survival=0.0% |
| Regime | FAIL | bull:21t/-55.8%, bear:5t/-31.0%, volatile:3t/+10.3% |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 22 | 31.8% | -19.5% | -0.19 |
| WF tune: conf=0.55 | 16 | 25.0% | -16.3% | -0.29 |
| WF tune: PT=15% | 24 | 20.8% | -38.4% | -0.29 |
| MC tune: max_loss=4.0% | 32 | 25.0% | -44.3% | -0.35 |
| WF tune: PT=12% | 28 | 21.4% | -45.2% | -0.37 |
| WF tune: conf=0.65 | 16 | 18.8% | -28.3% | -0.42 |
| MC tune: max_loss=3.0% | 34 | 17.6% | -44.3% | -0.46 |
| WF tune: PT=8% | 31 | 29.0% | -43.5% | -0.56 |
| WF tune: conf=0.45 | 29 | 24.1% | -56.6% | -0.59 |
| BS tune: conf=0.4 | 29 | 24.1% | -56.6% | -0.59 |
| BS tune: full rules (10) | 29 | 24.1% | -56.6% | -0.59 |
| BS tune: energy rules (12) | 29 | 24.1% | -56.6% | -0.59 |
| BS tune: + volume_breakout | 29 | 24.1% | -56.6% | -0.59 |
| BS tune: + commodity_breakout | 29 | 24.1% | -56.6% | -0.59 |
| Regime tune: + dollar_weakness | 29 | 24.1% | -56.6% | -0.59 |
| WF tune: conf=0.6 | 16 | 18.8% | -23.9% | -0.69 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=-19.5%, Trades=22, WR=31.8%, Sharpe=-0.19, PF=0.67, DD=-49.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.95, Test Sharpe=1.13, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6368, Sharpe CI=[-4.37, 2.37], WR CI=[18.2%, 54.5%] |
| Monte Carlo | FAIL | Ruin=0.4%, P95 DD=-47.6%, Median equity=$822, Survival=99.7% |
| Regime | FAIL | bull:16t/-16.6%, bear:4t/-19.1%, volatile:2t/+23.6% |

**Result: 0/4 gates passed**

---

### WF tune: conf=0.55

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-16.3%, Trades=16, WR=25.0%, Sharpe=-0.29, PF=0.59, DD=-39.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.59, Test Sharpe=0.72, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6479, Sharpe CI=[-5.31, 2.84], WR CI=[12.5%, 56.2%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.8%, Median equity=$836, Survival=100.0% |
| Regime | FAIL | bull:11t/+1.1%, bear:4t/-24.9%, volatile:1t/+11.5% |

**Result: 0/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-38.4%, Trades=24, WR=20.8%, Sharpe=-0.29, PF=0.55, DD=-65.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-1.05, Test Sharpe=0.90, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.8049, Sharpe CI=[-5.57, 1.59], WR CI=[8.3%, 41.7%] |
| Monte Carlo | FAIL | Ruin=20.1%, P95 DD=-60.7%, Median equity=$611, Survival=79.9% |
| Regime | FAIL | bull:17t/-42.8%, bear:5t/-29.4%, volatile:2t/+33.9% |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=7** | FAIL | FAIL | FAIL | FAIL | **-0.19** | **-19.5%** | 22 |
| WF tune: conf=0.55 | FAIL | FAIL | FAIL | FAIL | -0.29 | -16.3% | 16 |
| WF tune: PT=15% | FAIL | FAIL | FAIL | FAIL | -0.29 | -38.4% | 24 |
| Alt A: Full general rules (10 rules, 10%/5%) | FAIL | FAIL | FAIL | FAIL | -0.59 | -56.6% | 29 |

---

## 5. Final Recommendation

**GUSH partially validates.** Best config: WF tune: cooldown=7 (0/4 gates).

### WF tune: cooldown=7

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=-19.5%, Trades=22, WR=31.8%, Sharpe=-0.19, PF=0.67, DD=-49.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.95, Test Sharpe=1.13, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.6368, Sharpe CI=[-4.37, 2.37], WR CI=[18.2%, 54.5%] |
| Monte Carlo | FAIL | Ruin=0.4%, P95 DD=-47.6%, Median equity=$822, Survival=99.7% |
| Regime | FAIL | bull:16t/-16.6%, bear:4t/-19.1%, volatile:2t/+23.6% |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

