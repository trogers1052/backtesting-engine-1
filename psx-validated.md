# PSX (Phillips 66) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 14.3 minutes
**Category:** C Tier — Refining
**Bear Beta:** 0.80

---

## Methodology

**Hybrid Walk-Backward Validation** — tune rules on recent market structure, then validate they survive historical regimes.

### Validation Gates

| Gate | Method | Pass Criteria | Purpose |
|------|--------|---------------|---------|
| Walk-Backward | Tune on last 6mo, holdout 2mo, walk back through 2020-2024 | Holdout profitable + 3/4 regimes pass | Rules work in current AND historical markets |
| Bootstrap | 10,000 resamples | p < 0.05 AND Sharpe CI excludes zero | Statistical significance |
| Monte Carlo | 10,000 trade-order permutations | Ruin < 10% AND P95 DD < 40% | Worst-case risk |
| Regime | SPY SMA_50/SMA_200 + VIX | No regime >70% of profit | Not regime-dependent |

### Historical Regime Windows

| Window | Period | Expected |
|--------|--------|----------|
| 2020 Crash + Recovery | Feb 2020 - Dec 2020 | Crisis/Bull |
| 2021 Bull | Jan 2021 - Dec 2021 | Bull |
| 2022 Bear (Rate Hike) | Jan 2022 - Oct 2022 | Bear |
| 2023-2024 Chop | Jan 2023 - Jun 2024 | Chop |

---

## 1. Baseline Screening

PSX -- Refining + midstream + chemicals, margins can widen in volatility. C Tier — Refining.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 16 | 43.8% | +25.2% | 0.32 | 1.39 | -23.9% |
| Full general (10 rules, 10%/5%) | 18 | 44.4% | +30.9% | 0.59 | 1.48 | -25.7% |
| Tighter stops (3 rules, 10%/4%) | 16 | 37.5% | +11.5% | 0.14 | 1.11 | -19.1% |
| Wider PT (3 rules, 12%/5%) | 12 | 33.3% | +11.5% | 0.13 | 1.13 | -25.6% |
| Energy rules (14, 10%/5%) | 18 | 44.4% | +30.9% | 0.59 | 1.48 | -25.7% |
| Sector rules (refining) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Sector rules + 8% PT | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Sector + wider stops 6% | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline: Full general (10 rules, 10%/5%)**

---

## 2. Full Validation (Walk-Backward)

### Full general (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.9%, Trades=18, WR=44.4%, Sharpe=0.59, PF=1.48, DD=-25.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1451, Sharpe CI=[-1.66, 5.47], WR CI=[27.8%, 72.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.7%, Median=$1,368, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, bear:1t/-5.3%, volatile:2t/+2.6% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +20.3% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.45 | 18 | 44.4% | +30.9% | 0.59 |
| BS tune: +energy_momentum | 18 | 44.4% | +30.9% | 0.59 |
| Reg tune: +mean_rev | 18 | 44.4% | +30.9% | 0.59 |
| WB tune: stop=6.0% | 15 | 46.7% | +32.8% | 0.51 |
| WB tune: stop=4.0% | 21 | 38.1% | +25.6% | 0.44 |
| WB tune: PT=8% | 23 | 47.8% | +23.4% | 0.39 |
| WB tune: PT=15% | 18 | 33.3% | +25.0% | 0.34 |
| WB tune: PT=12% | 17 | 35.3% | +19.6% | 0.27 |
| WB tune: stop=3.0% | 26 | 30.8% | +12.2% | 0.16 |
| BS tune: conf=0.55 | 21 | 38.1% | -3.5% | -0.16 |
| WB tune: refining sector rules | 0 | 0.0% | +0.0% | 0.00 |

### Full Validation of Top Candidates

### WB tune: ATR x2.5

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+30.9%, Trades=18, WR=44.4%, Sharpe=0.59, PF=1.48, DD=-25.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1451, Sharpe CI=[-1.66, 5.47], WR CI=[27.8%, 72.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.7%, Median=$1,368, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, bear:1t/-5.3%, volatile:2t/+2.6% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +20.3% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### WB tune: energy 14 rules

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.9%, Trades=18, WR=44.4%, Sharpe=0.59, PF=1.48, DD=-25.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1451, Sharpe CI=[-1.66, 5.47], WR CI=[27.8%, 72.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.7%, Median=$1,368, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, bear:1t/-5.3%, volatile:2t/+2.6% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +20.3% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.9%, Trades=18, WR=44.4%, Sharpe=0.59, PF=1.48, DD=-25.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1451, Sharpe CI=[-1.66, 5.47], WR CI=[27.8%, 72.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.7%, Median=$1,368, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, bear:1t/-5.3%, volatile:2t/+2.6% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +20.3% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Full general (10 rules, 10%/5%)** | FAIL | FAIL | **PASS** | FAIL | **0.59** | **+30.9%** | 18 |
| WB tune: ATR x2.5 | FAIL | FAIL | **PASS** | FAIL | 0.59 | +30.9% | 18 |
| WB tune: energy 14 rules | FAIL | FAIL | **PASS** | FAIL | 0.59 | +30.9% | 18 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | FAIL | 0.59 | +30.9% | 18 |

---

## 5. Final Recommendation

**PSX partially validates.** Best config: Full general (10 rules, 10%/5%) (1/4 gates).

### Full general (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+30.9%, Trades=18, WR=44.4%, Sharpe=0.59, PF=1.48, DD=-25.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1451, Sharpe CI=[-1.66, 5.47], WR CI=[27.8%, 72.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-30.7%, Median=$1,368, Survival=100.0% |
| Regime | FAIL | bull:15t/+40.6%, bear:1t/-5.3%, volatile:2t/+2.6% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +20.3% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

