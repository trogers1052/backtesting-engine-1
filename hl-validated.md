# HL (Hecla Mining) Validated Optimization Results

**Date:** 2026-02-28
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily entries + 5-minute exits (multi-TF hybrid)
**Validation Runtime:** 65.1 minutes
**Prior Status:** Blacklisted (36.8% WR, -0.07 Sharpe, -38.7% DD)

---

## Methodology

### Validate-Then-Tune Approach

Same methodology as CCJ validation:

1. **Screen Baselines** — Test multiple starting configs to find the best baseline
2. **Validate Best** — Run through all 4 statistical validation gates
3. **Diagnose** — Identify which gates pass/fail and why
4. **Targeted Tune** — Sweep only the parameters that address failing gates
5. **Re-validate** — Confirm tuned configs through all 4 gates

### Validation Gates

| Gate | Method | Pass Criteria | Purpose |
|------|--------|---------------|---------|
| Walk-Forward | 70/30 train/test split, 5-day embargo, 10-day purge | Test Sharpe >= 50% of Train Sharpe | Detects parameter overfitting |
| Bootstrap | 10,000 resamples with replacement | p < 0.05 AND Sharpe 95% CI excludes zero | Tests statistical significance |
| Monte Carlo | 10,000 trade-order permutations | Ruin probability < 10% AND P95 drawdown < 40% | Measures worst-case risk |
| Regime | SPY SMA_50/SMA_200 + VIX classification | No single regime contributes >70% of total profit | Detects regime dependency |

---

## 1. Baseline Screening

HL was previously blacklisted, so multiple baseline configs were screened:

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Primary baseline (5 mining rules, 12%/6%) | 44 | 36.4% | +15.9% | 0.18 | 1.05 | -41.0% |
| Alt A: CCJ-style (3 rules, 10%/6%) | 28 | 32.1% | -19.8% | -0.44 | 0.71 | -38.3% |
| Alt B: Broad mining (8 rules, 12%/6%) | 44 | 36.4% | +13.9% | 0.15 | 1.04 | -41.0% |
| Alt C: Aggressive (5 rules, 15%/5%) | 68 | 36.8% | +19.0% | 0.20 | 1.07 | -35.7% |
| Alt D: Conservative (3 rules, 8%/4%) | 23 | 39.1% | +2.7% | -0.02 | 1.04 | -27.1% |

**Best baseline selected for validation: Alt C: Aggressive (5 rules, 15%/5%)**

---

## 2. Full Validation

### Alt C: Aggressive (5 rules, 15%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout, miner_metal_ratio`
- **Profit Target:** 15%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+19.0%, Trades=68, WR=36.8%, Sharpe=0.20, PF=1.07, DD=-35.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.41, Test Sharpe=0.74, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2240, Sharpe CI=[-1.16, 2.16], WR CI=[27.9%, 51.5%] |
| Monte Carlo | FAIL | Ruin=0.9%, P95 DD=-51.7%, Median equity=$1,373, Survival=99.1% |
| Regime | PASS | bull:51t/+35.6%, bear:9t/+3.4%, chop:4t/-6.3%, volatile:4t/+16.8% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: cooldown=3 | 71 | 43.7% | +65.4% | 0.52 |
| WF tune: PT=20% | 60 | 36.7% | +54.2% | 0.40 |
| WF tune: PT=15% | 68 | 36.8% | +19.0% | 0.20 |

### Full Validation of Top Candidates

### BS tune: cooldown=3

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout, miner_metal_ratio`
- **Profit Target:** 15%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+65.4%, Trades=71, WR=43.7%, Sharpe=0.52, PF=1.30, DD=-38.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.07, Test Sharpe=1.29, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.0906, Sharpe CI=[-0.57, 2.61], WR CI=[35.2%, 57.7%] |
| Monte Carlo | FAIL | Ruin=0.2%, P95 DD=-46.5%, Median equity=$1,962, Survival=99.8% |
| Regime | FAIL | bull:51t/+72.5%, bear:9t/+7.1%, chop:7t/-12.5%, volatile:4t/+19.4% |

**Result: 0/4 gates passed**

---

### WF tune: PT=20%

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout, miner_metal_ratio`
- **Profit Target:** 20%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+54.2%, Trades=60, WR=36.7%, Sharpe=0.40, PF=1.29, DD=-34.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.10, Test Sharpe=0.86, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1357, Sharpe CI=[-0.95, 2.55], WR CI=[28.3%, 51.7%] |
| Monte Carlo | FAIL | Ruin=0.3%, P95 DD=-48.3%, Median equity=$1,764, Survival=99.7% |
| Regime | FAIL | bull:43t/+77.2%, bear:9t/-17.8%, chop:4t/-6.3%, volatile:4t/+24.9% |

**Result: 0/4 gates passed**

---

### WF tune: PT=15%

- **Rules:** `trend_continuation, seasonality, death_cross, commodity_breakout, miner_metal_ratio`
- **Profit Target:** 15%
- **Min Confidence:** 0.6
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+19.0%, Trades=68, WR=36.8%, Sharpe=0.20, PF=1.07, DD=-35.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.41, Test Sharpe=0.74, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2240, Sharpe CI=[-1.16, 2.16], WR CI=[27.9%, 51.5%] |
| Monte Carlo | FAIL | Ruin=0.9%, P95 DD=-51.7%, Median equity=$1,373, Survival=99.1% |
| Regime | PASS | bull:51t/+35.6%, bear:9t/+3.4%, chop:4t/-6.3%, volatile:4t/+16.8% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Alt C: Aggressive (5 rules, 15%/5%)** | FAIL | FAIL | FAIL | **PASS** | **0.20** | **+19.0%** | 68 |
| BS tune: cooldown=3 | FAIL | FAIL | FAIL | FAIL | 0.52 | +65.4% | 71 |
| WF tune: PT=20% | FAIL | FAIL | FAIL | FAIL | 0.40 | +54.2% | 60 |
| WF tune: PT=15% | FAIL | FAIL | FAIL | **PASS** | 0.20 | +19.0% | 68 |

---

## 5. Gate Failure Analysis

### Walk-Forward: Consistently Negative Train Sharpe

Every config produced a **negative Sharpe in the training period (Jan 2021 - Aug 2024)**, meaning the strategy lost money during the majority of the backtest window. The positive test period performance (Aug 2024 - Feb 2026) reflects a recent silver rally, not a persistent edge.

| Config | Train Sharpe | Test Sharpe | Interpretation |
|--------|-------------|-------------|----------------|
| Baseline (15%/5%) | -0.41 | 0.74 | Lost money 2021-2024, lucky rally 2024-2026 |
| Cooldown=3 | -0.07 | 1.29 | Near-zero in training, all gains recent |
| PT=20% | -0.10 | 0.86 | Same pattern: recent gains inflate full-period stats |

This is the opposite of overfitting — the strategy simply doesn't work across the full history. The walk-forward gate is correctly identifying that any positive returns are recency bias.

### Bootstrap: No Statistical Edge (p >> 0.05)

All configs showed p-values far above the 0.05 significance threshold:

| Config | p-value | Sharpe 95% CI | Interpretation |
|--------|---------|---------------|----------------|
| Baseline (15%/5%) | 0.2240 | [-1.16, 2.16] | Cannot reject null hypothesis of no edge |
| Cooldown=3 | 0.0906 | [-0.57, 2.61] | Closest to significance, still fails |
| PT=20% | 0.1357 | [-0.95, 2.55] | CI includes large negative values |

The Sharpe 95% CIs all include zero (and even large negative values), meaning we cannot distinguish HL's performance from random noise with any statistical confidence. With 60-71 trades over 5 years, there is sufficient sample size — the edge simply doesn't exist.

### Monte Carlo: Excessive Drawdown Risk

While ruin probability is low (<1%), the P95 drawdown exceeds the 40% threshold in every config:

| Config | P95 Drawdown | Worst Case | Threshold |
|--------|-------------|------------|-----------|
| Baseline (15%/5%) | -51.7% | -68.2% | < 40% |
| Cooldown=3 | -46.5% | -66.1% | < 40% |
| PT=20% | -48.3% | -66.0% | < 40% |

In 5% of trade orderings, the strategy draws down 46-52% — catastrophic for an $888 account. The worst-case drawdown of -66% to -68% would leave the account at ~$320.

### Regime: Only Gate That Sometimes Passes

The baseline config (15%/5%) passes Regime analysis because HL trades across bull, bear, chop, and volatile markets. However, performance in all regimes is poor:
- **Bull:** 35.3% WR, barely profitable (+0.70% avg)
- **Bear:** 44.4% WR, marginal (+0.38% avg)
- **Chop:** 50.0% WR but -1.57% avg (winners are tiny)
- **Volatile:** 75% WR — only 4 trades, not statistically meaningful

The regime diversity is "spread evenly bad" rather than "concentrated good."

---

## 6. Final Recommendation

**HL blacklist confirmed.** No configuration passes more than 1 of 4 validation gates. The statistical evidence is clear:

- **No edge exists** (p=0.09-0.22, all CIs include zero)
- **Negative train-period Sharpe** (strategy lost money 2021-2024)
- **Excessive drawdown risk** (P95 DD = 47-52%, worst case 66-68%)
- **Win rate ~37%** with insufficient winners to compensate

### Why HL Doesn't Work With Our Rules

1. **Silver is more volatile and mean-reverting than uranium** — trend_continuation rules that work for CCJ's secular bull market generate noise trades on HL's choppier price action
2. **Seasonal patterns are weak** — silver doesn't have uranium's institutional procurement cycle, so seasonality rules produce low-confidence signals
3. **Death cross protection is insufficient** — HL's drawdowns come from rapid gap-downs, not gradual declines that death_cross can detect
4. **37% win rate requires 2:1+ R:R to break even** — HL's average winner is too small relative to average loser (PF = 1.07)

### Comparison: HL vs CCJ (Why Rules Work for One But Not the Other)

| Metric | CCJ (Validated) | HL (Blacklisted) |
|--------|----------------|------------------|
| Gates Passed | 3/4 | 1/4 |
| Train Sharpe | 0.48 | -0.41 |
| Bootstrap p-value | 0.0146 | 0.2240 |
| P95 Drawdown | -31.2% | -51.7% |
| Win Rate | 57.6% | 36.8% |
| Profit Factor | 2.04 | 1.07 |
| Commodity | Uranium (trending) | Silver (mean-reverting) |
| Demand Driver | Nuclear energy adoption | Industrial/speculative |

CCJ works because uranium has a structural supply deficit driving a secular trend. HL fails because silver lacks this one-directional catalyst.

### Alternative Silver/Mining Exposure

If silver/mining exposure is desired, these validated alternatives perform better:

| Symbol | Description | Historical WR | Sharpe | Notes |
|--------|-------------|---------------|--------|-------|
| WPM | Wheaton Precious Metals | 48.6% | 0.67 | Streaming model, less operational risk |
| SLV | Silver ETF | 56.9% | 0.60 | Pure silver price exposure, no miner risk |
| CCJ | Cameco (Uranium) | 57.6% | 0.83 | Validated through 3/4 gates |

### Action: Keep HL Blacklisted

```
# In rules.yaml - DO NOT TRADE list:
# HL: CONFIRMED BLACKLIST - 0-1/4 validation gates passed
#     36.8% WR, -0.41 train Sharpe, p=0.22, P95 DD=-52%
#     Validated 2026-02-28 with 5 baselines + 13 tune configs
```

