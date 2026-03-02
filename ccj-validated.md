# CCJ Validated Optimization Results

**Date:** 2026-02-28
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily entries + 5-minute exits (multi-TF hybrid)
**Validation Runtime:** 34.2 minutes

---

## Methodology

### Validate-Then-Tune Approach

Rather than a brute-force grid search (which risks overfitting via multiple comparisons bias), we used a quant-correct validate-then-tune methodology:

1. **Validate Baseline** — Run the current CCJ ruleset through all 4 statistical validation gates
2. **Diagnose** — Identify which gates pass/fail and why
3. **Targeted Tune** — Based on diagnosis, sweep only the parameters that address the failing gate
4. **Re-validate** — Confirm tuned configs through all 4 gates (not just the failing one)
5. **Report** — Document findings with full statistical evidence

### Validation Gates

| Gate | Method | Pass Criteria | Purpose |
|------|--------|---------------|---------|
| Walk-Forward | 70/30 train/test split, 5-day embargo, 10-day purge | Test Sharpe >= 50% of Train Sharpe | Detects parameter overfitting to in-sample data |
| Bootstrap | 10,000 resamples with replacement | p < 0.05 AND Sharpe 95% CI excludes zero | Tests statistical significance of the trading edge |
| Monte Carlo | 10,000 trade-order permutations | Ruin probability < 10% AND P95 drawdown < 40% | Measures path dependency and worst-case risk |
| Regime | SPY SMA_50/SMA_200 + VIX classification | No single regime contributes >70% of total profit | Detects strategies that only work in one market condition |

### Bug Fix Applied Before Validation

A critical 100x scale error was discovered and fixed in the validation modules during this work. The backtesting strategy stores `profit_pct` as decimal fractions (0.10 = 10%), but `bootstrap.py` and `monte_carlo.py` were dividing by 100 again, interpreting 10% gains as 0.1%. This caused Bootstrap to always report "no edge" and Monte Carlo to show nearly flat equity curves regardless of actual performance. The fix was applied to `bootstrap.py`, `monte_carlo.py`, `report.py`, and all corresponding test files. All 381 tests pass after the fix.

---

## 1. Baseline Validation

### Configuration

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 6.0%
- **Cooldown:** 7 bars
- **Stop Loss:** 100% (disabled — max_loss_pct is the effective stop)
- **Timeframe:** daily entries, 5min exits

### Backtest Performance

| Metric | Value |
|--------|-------|
| Total Return | +112.8% |
| Total Trades | 33 |
| Win Rate | 57.6% |
| Sharpe Ratio | 0.83 |
| Profit Factor | 2.04 |
| Max Drawdown | -23.5% |

### Gate Results

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.48, Test Sharpe=0.72, Ratio=148% (need >=50%) |
| Bootstrap | **PASS** | p=0.0146, Sharpe CI=[0.26, 5.72], WR CI=[42.4%, 75.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.2%, Median equity=$2,391, Survival=100.0% |
| Regime | **FAIL** | bull:30t/+100.5%, chop:1t/+10.2%, volatile:2t/-12.4% |

**Result: 3/4 gates passed**

### Gate Details

**Walk-Forward (PASS):** The test period (Aug 2024 - Feb 2026) actually produced a *higher* Sharpe (0.72) than the training period (Jan 2021 - Aug 2024, Sharpe=0.48). The 148% ratio is well above the 50% threshold. This is strong evidence the strategy is not overfit — it performs better out-of-sample than in-sample.

**Bootstrap (PASS):** With p=0.0146 (well below 0.05), the null hypothesis that this strategy has no positive risk-adjusted edge is rejected. The Sharpe ratio 95% confidence interval [0.26, 5.72] excludes zero, confirming a statistically significant positive edge. The win rate CI [42.4%, 75.8%] includes 50%, which is a minor caveat — the edge comes from trade magnitude (big winners, small losers) rather than win frequency.

**Monte Carlo (PASS):** Across 10,000 random trade orderings, the strategy never hits ruin (equity below $500 on $1,000 initial). The median drawdown is -20.2%, P95 drawdown is -31.2%, and worst-case drawdown is -50.9%. Final equity is always $2,391 (multiplication is commutative — path order doesn't affect final value, only drawdown severity). 100% survival rate.

**Regime (FAIL):** 30 of 33 trades (91%) occurred during SPY bull markets. Bull regime contributes +100.5% vs chop +10.2% and volatile -12.4%. With 100.5 / (100.5 + 10.2) = 91% of positive profit from bull, this exceeds the 70% regime-dependency threshold. See analysis in Section 4.

---

## 2. Diagnosis

**Only Regime gate failed.** The baseline passes Walk-Forward, Bootstrap, and Monte Carlo convincingly. The regime dependency is the sole concern.

**Root cause:** CCJ (Cameco, uranium) is a trend/momentum commodity stock. The `trend_continuation` rule explicitly buys pullbacks in uptrends, and `seasonality` targets Q1-Q2 uranium procurement cycles. Both rules naturally generate more signals during bull markets when CCJ is trending up. The `death_cross` rule provides downside protection but generates few standalone signals.

**Tuning strategy:** Generate 4 targeted configs aimed at distributing trades across regimes:
1. Add `trend_alignment` rule (align with SPY trend)
2. Add `golden_cross` rule (require SMA_50 > SMA_200)
3. Both `trend_alignment` + `golden_cross`
4. Raise confidence threshold to 0.70 (filter weaker signals)

---

## 3. Tuning Results

### Quick Screen (4 configs)

All 4 tuning configs were backtested against the full period to filter viable candidates:

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Baseline (reference) | 33 | 57.6% | +112.8% | 0.83 |
| + trend_alignment | 36 | 47.2% | +30.6% | 0.42 |
| + golden_cross | 19 | 42.1% | -1.0% | -0.07 |
| + trend_align + golden_cross | 38 | 44.7% | +21.8% | 0.27 |
| Higher confidence=0.70 | 29 | 44.8% | +28.7% | 0.36 |

All tuning configs dramatically underperform the baseline. Adding rules destroys the edge rather than distributing it across regimes.

### Full Validation of Top 3 Candidates

The top 3 candidates by Sharpe were run through all 4 validation gates:

#### Regime tune: + trend_alignment (0/4 gates)

- **Rules:** `trend_continuation, seasonality, death_cross, trend_alignment`
- **Performance:** Return=+30.6%, Trades=36, WR=47.2%, Sharpe=0.42, PF=1.24, DD=-26.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.02, Test Sharpe=0.48, Ratio=N/A (train negative) |
| Bootstrap | FAIL | p=0.1658, Sharpe CI=[-1.26, 3.55] — CI includes zero |
| Monte Carlo | FAIL | P95 DD=-41.2% (exceeds 40% threshold) |
| Regime | FAIL | bull:31t/+44.2%, chop:3t/+14.3%, volatile:2t/-11.0% — still >70% bull |

Adding `trend_alignment` introduced 3 more trades but dropped win rate from 57.6% to 47.2% and destroyed the statistical edge (p jumped from 0.0146 to 0.1658).

#### Regime tune: higher confidence=0.70 (2/4 gates)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Performance:** Return=+28.7%, Trades=29, WR=44.8%, Sharpe=0.36, PF=1.30, DD=-36.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | PASS | Train Sharpe=0.07, Test Sharpe=0.22, Ratio=325% |
| Bootstrap | FAIL | p=0.1759, Sharpe CI=[-1.43, 4.01] — CI includes zero |
| Monte Carlo | PASS | Ruin=0.0%, P95 DD=-38.2%, Survival=100.0% |
| Regime | FAIL | bull:26t/+60.0%, bear:1t/-5.3%, volatile:2t/-12.4% — still >70% bull |

Higher confidence reduced trades from 33 to 29 and cut return from +112.8% to +28.7%. Lost Bootstrap significance entirely.

#### Regime tune: + trend_align + golden_cross (0/4 gates)

- **Rules:** `trend_continuation, seasonality, death_cross, trend_alignment, golden_cross`
- **Performance:** Return=+21.8%, Trades=38, WR=44.7%, Sharpe=0.27, PF=1.17, DD=-26.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.07, Test Sharpe=0.52, Ratio=N/A (train negative) |
| Bootstrap | FAIL | p=0.2083, Sharpe CI=[-1.40, 3.23] — CI includes zero |
| Monte Carlo | FAIL | P95 DD=-42.8% (exceeds 40% threshold) |
| Regime | FAIL | bull:32t/+42.0%, chop:3t/+14.3%, volatile:2t/-11.0% — still >70% bull |

Adding both rules created the worst config. More trades (38) but lower quality — the additional rules generated noise trades that diluted the edge.

---

## 4. Regime Analysis Deep Dive

### Why CCJ Is Regime-Dependent

The regime dependency is **structural, not a flaw in the rules**:

1. **Uranium is a trending commodity.** CCJ tracks uranium prices, which have been in a secular bull market since 2021 driven by nuclear energy adoption. The `trend_continuation` rule correctly identifies this.

2. **Seasonality is real but regime-correlated.** Uranium procurement cycles (Q1-Q2) create genuine seasonal patterns, but these are strongest when the overall market is bullish.

3. **Bear/volatile periods had very few trades.** Only 3 of 33 trades occurred outside bull markets (1 chop, 2 volatile). The 2 volatile trades both lost (-6.21% avg), suggesting the `death_cross` rule correctly prevented most bear-market entries.

4. **Adding regime-diversifying rules destroyed the edge.** Every attempt to generate non-bull trades (trend_alignment, golden_cross) produced noise trades with ~45% win rate and negative expected value.

### Implication for Live Trading

The regime dependency means:
- **This strategy should be sized conservatively** — it will underperform or lose money in bear/volatile/crisis markets
- **Consider reducing or pausing CCJ positions when SPY regime shifts to bear/volatile/crisis** (context-service already detects this)
- **Do not add rules just to pass the regime gate** — the data shows this destroys the genuine edge

---

## 5. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Baseline** | **PASS** | **PASS** | **PASS** | FAIL | **0.83** | **+112.8%** | 33 |
| + trend_alignment | FAIL | FAIL | FAIL | FAIL | 0.42 | +30.6% | 36 |
| conf=0.70 | PASS | FAIL | PASS | FAIL | 0.36 | +28.7% | 29 |
| + trend_align + golden_cross | FAIL | FAIL | FAIL | FAIL | 0.27 | +21.8% | 38 |
| + golden_cross (screened only) | - | - | - | - | -0.07 | -1.0% | 19 |

---

## 6. Final Recommendation

**Use the baseline configuration as-is.** It passes 3/4 validation gates with strong evidence of a genuine statistical edge:

- **Statistically significant edge** (p=0.0146, Sharpe CI excludes zero)
- **Not overfit** (out-of-sample Sharpe 148% of in-sample)
- **Survivable risk** (0% ruin, 100% survival, P95 DD=-31.2%)
- **Regime-aware deployment** (reduce size in non-bull markets)

### Production Configuration

```
Rules: trend_continuation, seasonality, death_cross
Profit Target: 10%
Min Confidence: 0.65
Max Loss: 6.0%
Cooldown: 7 bars
Timeframe: daily (entries), 5min (exits)
```

### Reproduce with CLI

```
python -m backtesting --symbol CCJ --start 2021-01-01 --end 2026-02-28 \
  --rules trend_continuation,seasonality,death_cross \
  --profit-target 0.10 --min-confidence 0.65 --max-loss 6.0 \
  --cooldown-bars 7 --timeframe daily --exit-timeframe 5min \
  --walk-forward --bootstrap --monte-carlo --regime-analysis
```

---

## 7. Validation Infrastructure Notes

### Files Modified During This Work

| File | Change |
|------|--------|
| `backtesting/validation/bootstrap.py` | Fixed profit_pct scale: removed erroneous `/100.0` division |
| `backtesting/validation/monte_carlo.py` | Fixed profit_pct scale: `multipliers = 1.0 + trade_pnl` (was `/100`) |
| `backtesting/validation/report.py` | Fixed regime display: multiply by 100 for percentage display |
| `tests/test_bootstrap.py` | Updated test data to use decimal fractions (0.05 not 5.0) |
| `tests/test_monte_carlo.py` | Updated test data to use decimal fractions; fixed commutative product test |
| `tests/test_regime.py` | Updated test data to use decimal fractions |
| `validate_ccj.py` | New: validate-then-tune optimization script |

### Test Results

All **381 tests pass** after the scale fix (76 validation tests + 305 other tests).
