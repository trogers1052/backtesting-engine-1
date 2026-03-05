# USAR (USA Rare Earth) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 8.2 minutes
**Category:** Mining / Rare Earth (speculative)

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

USAR — Rare earth mining/processing — micro-cap, speculative, critical minerals play. Mining / Rare Earth (speculative).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt A: Full general rules (10 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt C: Wider PT (3 rules, 12%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt D: Recommended rules (5 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt E: Mining full (5 rules, 10%/5%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt F: Mining momentum (15%/8%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Alt G: Mining wide (20%/10%) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline selected for validation: Lean 3 rules baseline (10%/5%, conf=0.50)**

---

## 2. Full Validation

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+0.0%, Trades=0, WR=0.0%, Sharpe=0.00, PF=0.00, DD=-0.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: PT=6% | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: PT=7% | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: PT=8% | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: PT=12% | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: PT=15% | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: conf=0.45 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: conf=0.55 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: conf=0.6 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: conf=0.65 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: cooldown=3 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: cooldown=7 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: ATR stops x2.5 | 0 | 0.0% | +0.0% | 0.00 |
| WF tune: recommended rules | 0 | 0.0% | +0.0% | 0.00 |
| BS tune: conf=0.4 | 0 | 0.0% | +0.0% | 0.00 |
| BS tune: full rules (10) | 0 | 0.0% | +0.0% | 0.00 |
| MC tune: max_loss=3.0% | 0 | 0.0% | +0.0% | 0.00 |
| MC tune: max_loss=4.0% | 0 | 0.0% | +0.0% | 0.00 |
| MC tune: ATR stops x2.0 | 0 | 0.0% | +0.0% | 0.00 |

### Full Validation of Top Candidates

### WF tune: PT=6%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 6%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+0.0%, Trades=0, WR=0.0%, Sharpe=0.00, PF=0.00, DD=-0.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

### WF tune: PT=7%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 7%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+0.0%, Trades=0, WR=0.0%, Sharpe=0.00, PF=0.00, DD=-0.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

### WF tune: PT=8%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+0.0%, Trades=0, WR=0.0%, Sharpe=0.00, PF=0.00, DD=-0.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Lean 3 rules baseline (10%/5%, conf=0.50)** | FAIL | FAIL | FAIL | FAIL | **0.00** | **+0.0%** | 0 |
| WF tune: PT=6% | FAIL | FAIL | FAIL | FAIL | 0.00 | +0.0% | 0 |
| WF tune: PT=7% | FAIL | FAIL | FAIL | FAIL | 0.00 | +0.0% | 0 |
| WF tune: PT=8% | FAIL | FAIL | FAIL | FAIL | 0.00 | +0.0% | 0 |

---

## 5. Final Recommendation

**USAR partially validates.** Best config: Lean 3 rules baseline (10%/5%, conf=0.50) (0/4 gates).

### Lean 3 rules baseline (10%/5%, conf=0.50)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+0.0%, Trades=0, WR=0.0%, Sharpe=0.00, PF=0.00, DD=-0.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

