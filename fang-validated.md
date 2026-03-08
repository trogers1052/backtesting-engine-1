# FANG (Diamondback Energy) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 17.7 minutes
**Category:** C Tier — Upstream E&P
**Bear Beta:** 1.20-1.40

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

FANG -- Lowest-cost Permian E&P, recovery play, survives anything. C Tier — Upstream E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 11 | 9.1% | -29.4% | -0.85 | 0.16 | -35.3% |
| Full general (10 rules, 10%/5%) | 22 | 40.9% | +26.2% | 0.85 | 1.28 | -16.4% |
| Tighter stops (3 rules, 10%/4%) | 12 | 8.3% | -30.6% | -0.80 | 0.15 | -36.5% |
| Wider PT (3 rules, 12%/5%) | 11 | 9.1% | -27.2% | -0.79 | 0.24 | -33.1% |
| Energy rules (14, 10%/5%) | 22 | 40.9% | +26.2% | 0.85 | 1.28 | -16.4% |
| Sector rules (upstream) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
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

**Performance:** Return=+26.2%, Trades=22, WR=40.9%, Sharpe=0.85, PF=1.28, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1663, Sharpe CI=[-1.60, 4.56], WR CI=[27.3%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.3%, Median=$1,339, Survival=100.0% |
| Regime | **PASS** | bull:19t/+19.4%, bear:1t/+10.7%, volatile:2t/+5.5% |

**Result: 2/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +3.6% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.45 | 22 | 40.9% | +26.2% | 0.85 |
| BS tune: +energy_momentum | 22 | 40.9% | +26.2% | 0.85 |
| WB tune: stop=6.0% | 18 | 50.0% | +43.5% | 0.80 |
| WB tune: ATR x2.5 | 23 | 39.1% | +24.7% | 0.69 |
| WB tune: PT=12% | 17 | 41.2% | +23.5% | 0.41 |
| WB tune: PT=8% | 26 | 42.3% | +19.8% | 0.32 |
| WB tune: stop=4.0% | 22 | 36.4% | +14.8% | 0.29 |
| WB tune: stop=3.0% | 28 | 28.6% | +5.6% | 0.02 |
| BS tune: conf=0.55 | 25 | 40.0% | +5.1% | 0.00 |
| WB tune: upstream sector rules | 0 | 0.0% | +0.0% | 0.00 |

### Full Validation of Top Candidates

### WB tune: PT=15%

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+36.8%, Trades=16, WR=43.8%, Sharpe=0.89, PF=1.70, DD=-17.3%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1315, Sharpe CI=[-1.90, 5.71], WR CI=[25.0%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-29.3%, Median=$1,456, Survival=100.0% |
| Regime | FAIL | bull:12t/+52.4%, bear:2t/-17.5%, chop:1t/-5.0%, volatile:1t/+15.7% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +17.7% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### WB tune: energy 14 rules

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+26.2%, Trades=22, WR=40.9%, Sharpe=0.85, PF=1.28, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1663, Sharpe CI=[-1.60, 4.56], WR CI=[27.3%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.3%, Median=$1,339, Survival=100.0% |
| Regime | **PASS** | bull:19t/+19.4%, bear:1t/+10.7%, volatile:2t/+5.5% |

**Result: 2/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +3.6% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### BS tune: conf=0.4

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+26.2%, Trades=22, WR=40.9%, Sharpe=0.85, PF=1.28, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1663, Sharpe CI=[-1.60, 4.56], WR CI=[27.3%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.3%, Median=$1,339, Survival=100.0% |
| Regime | **PASS** | bull:19t/+19.4%, bear:1t/+10.7%, volatile:2t/+5.5% |

**Result: 2/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +3.6% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Full general (10 rules, 10%/5%)** | FAIL | FAIL | **PASS** | **PASS** | **0.85** | **+26.2%** | 22 |
| WB tune: energy 14 rules | FAIL | FAIL | **PASS** | **PASS** | 0.85 | +26.2% | 22 |
| BS tune: conf=0.4 | FAIL | FAIL | **PASS** | **PASS** | 0.85 | +26.2% | 22 |
| WB tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.89 | +36.8% | 16 |

---

## 5. Final Recommendation

**FANG partially validates.** Best config: Full general (10 rules, 10%/5%) (2/4 gates).

### Full general (10 rules, 10%/5%)

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 3 bars

**Performance:** Return=+26.2%, Trades=22, WR=40.9%, Sharpe=0.85, PF=1.28, DD=-16.4%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1663, Sharpe CI=[-1.60, 4.56], WR CI=[27.3%, 63.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.3%, Median=$1,339, Survival=100.0% |
| Regime | **PASS** | bull:19t/+19.4%, bear:1t/+10.7%, volatile:2t/+5.5% |

**Result: 2/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +3.6% | 6 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions or reduced sizing
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

