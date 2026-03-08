# CVX (Chevron) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 18.2 minutes
**Category:** B Tier — Integrated
**Bear Beta:** 0.70

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

CVX -- Integrated oil major — Permian, LNG, refining, strongest balance sheet. B Tier — Integrated.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 9 | 11.1% | -29.4% | -0.81 | 0.14 | -39.5% |
| Full general (10 rules, 10%/5%) | 15 | 33.3% | -14.1% | -0.24 | 0.68 | -34.3% |
| Tighter stops (3 rules, 10%/4%) | 9 | 11.1% | -24.6% | -0.65 | 0.20 | -36.9% |
| Wider PT (3 rules, 12%/5%) | 9 | 11.1% | -28.7% | -0.75 | 0.16 | -39.5% |
| Energy rules (14, 10%/5%) | 22 | 40.9% | -9.6% | -0.15 | 0.76 | -33.9% |
| Sector rules (integrated) | 16 | 37.5% | -19.6% | -0.31 | 0.50 | -42.0% |
| Sector rules + 8% PT | 18 | 38.9% | -11.7% | -0.22 | 0.63 | -35.9% |

**Best baseline: Energy rules (14, 10%/5%)**

---

## 2. Full Validation (Walk-Backward)

### Energy rules (14, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.6%, Trades=22, WR=40.9%, Sharpe=-0.15, PF=0.76, DD=-33.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +3.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.5385, Sharpe CI=[-3.60, 2.88], WR CI=[27.3%, 68.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median=$935, Survival=100.0% |
| Regime | FAIL | bull:16t/-1.8%, bear:1t/-1.8%, chop:2t/+6.1%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +3.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -11.6% | 5 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WB tune: stop=4.0% | 23 | 39.1% | -10.4% | -0.15 |
| WB tune: stop=6.0% | 21 | 38.1% | -10.6% | -0.16 |
| BS tune: conf=0.45 | 22 | 36.4% | -11.0% | -0.16 |
| WB tune: ATR x2.5 | 23 | 39.1% | -11.4% | -0.16 |
| WB tune: stop=3.0% | 29 | 31.0% | -16.5% | -0.23 |
| WB tune: PT=15% | 22 | 36.4% | -14.4% | -0.23 |
| WB tune: PT=12% | 22 | 36.4% | -17.0% | -0.27 |
| WB tune: integrated sector rules | 15 | 40.0% | -15.6% | -0.28 |
| BS tune: conf=0.55 | 23 | 30.4% | -21.0% | -0.43 |

### Full Validation of Top Candidates

### WB tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.6%, Trades=27, WR=44.4%, Sharpe=-0.00, PF=0.92, DD=-31.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +3.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.4005, Sharpe CI=[-2.47, 3.17], WR CI=[29.6%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.8%, Median=$1,056, Survival=100.0% |
| Regime | FAIL | bull:18t/+1.4%, bear:3t/+1.9%, chop:3t/-0.8%, volatile:3t/+7.9% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +3.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -11.6% | 5 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

### BS tune: energy 14 rules

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.6%, Trades=22, WR=40.9%, Sharpe=-0.15, PF=0.76, DD=-33.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +3.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.5385, Sharpe CI=[-3.60, 2.88], WR CI=[27.3%, 68.2%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.5%, Median=$935, Survival=100.0% |
| Regime | FAIL | bull:16t/-1.8%, bear:1t/-1.8%, chop:2t/+6.1%, volatile:3t/-4.2% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +3.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -11.6% | 5 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=-12.3%, Trades=21, WR=38.1%, Sharpe=-0.15, PF=0.73, DD=-36.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +3.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.5574, Sharpe CI=[-3.89, 2.91], WR CI=[23.8%, 61.9%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-40.1%, Median=$910, Survival=100.0% |
| Regime | FAIL | bull:15t/-4.0%, bear:1t/-1.8%, chop:2t/+6.1%, volatile:3t/-4.2% |

**Result: 0/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +3.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -7.8% | 5 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WB tune: PT=8%** | FAIL | FAIL | **PASS** | FAIL | **-0.00** | **+0.6%** | 27 |
| Energy rules (14, 10%/5%) | FAIL | FAIL | **PASS** | FAIL | -0.15 | -9.6% | 22 |
| BS tune: energy 14 rules | FAIL | FAIL | **PASS** | FAIL | -0.15 | -9.6% | 22 |
| BS tune: conf=0.4 | FAIL | FAIL | FAIL | FAIL | -0.15 | -12.3% | 21 |

---

## 5. Final Recommendation

**CVX partially validates.** Best config: WB tune: PT=8% (1/4 gates).

### WB tune: PT=8%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+0.6%, Trades=27, WR=44.4%, Sharpe=-0.00, PF=0.92, DD=-31.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +3.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.4005, Sharpe CI=[-2.47, 3.17], WR CI=[29.6%, 66.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-34.8%, Median=$1,056, Survival=100.0% |
| Regime | FAIL | bull:18t/+1.4%, bear:3t/+1.9%, chop:3t/-0.8%, volatile:3t/+7.9% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +3.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -11.6% | 5 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

