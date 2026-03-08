# COP (ConocoPhillips) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 82.6 minutes
**Category:** B Tier — Upstream E&P
**Bear Beta:** 0.90-1.00

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

COP -- Lowest-cost independent E&P — Permian, Eagle Ford, Bakken, Alaska. B Tier — Upstream E&P.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 9 | 22.2% | -18.7% | -0.43 | 0.38 | -31.9% |
| Full general (10 rules, 10%/5%) | 13 | 23.1% | -29.8% | -0.51 | 0.29 | -41.5% |
| Tighter stops (3 rules, 10%/4%) | 10 | 20.0% | -19.0% | -0.41 | 0.38 | -32.2% |
| Wider PT (3 rules, 12%/5%) | 9 | 22.2% | -17.5% | -0.38 | 0.42 | -31.8% |
| Energy rules (14, 10%/5%) | 22 | 36.4% | -23.9% | -0.34 | 0.61 | -42.6% |
| Sector rules (upstream) | 15 | 33.3% | -21.4% | -0.31 | 0.57 | -40.7% |
| Sector rules + 8% PT | 20 | 30.0% | -17.6% | -0.26 | 0.57 | -40.5% |
| Sector + wider stops 6% | 15 | 33.3% | -21.5% | -0.31 | 0.57 | -40.8% |

**Best baseline: Sector rules + 8% PT**

---

## 2. Full Validation (Walk-Backward)

### Sector rules + 8% PT

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=-17.6%, Trades=20, WR=30.0%, Sharpe=-0.26, PF=0.57, DD=-40.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=FRAGILE, Holdout=PASS (+0.0%), Regimes=0/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull (Low Rat... |
| Bootstrap | FAIL | p=0.6970, Sharpe CI=[-5.20, 2.31], WR CI=[15.0%, 55.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-38.4%, Median=$840, Survival=100.0% |
| Regime | FAIL | bull:16t/-6.3%, chop:2t/+0.3%, volatile:2t/-7.5% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 0/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -13.1% | 3 | bull | FAIL |

**Verdict: FRAGILE**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WB tune: ATR x2.5 | 20 | 30.0% | -17.6% | -0.26 |
| BS tune: conf=0.4 | 20 | 30.0% | -17.6% | -0.26 |
| BS tune: conf=0.45 | 20 | 30.0% | -17.6% | -0.26 |
| BS tune: conf=0.55 | 20 | 30.0% | -17.6% | -0.26 |
| Reg tune: +mean_rev | 20 | 30.0% | -17.6% | -0.26 |
| WB tune: PT=12% | 15 | 20.0% | -19.4% | -0.37 |
| WB tune: PT=15% | 15 | 20.0% | -20.1% | -0.40 |

### Full Validation of Top Candidates

### WB tune: energy 14 rules

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.2%, Trades=33, WR=36.4%, Sharpe=-0.11, PF=0.88, DD=-41.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +4.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.5297, Sharpe CI=[-2.91, 2.28], WR CI=[21.2%, 51.5%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.0%, Median=$941, Survival=100.0% |
| Regime | **PASS** | bull:26t/-8.2%, bear:3t/-15.4%, chop:1t/+8.3%, volatile:3t/+15.3% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +4.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -13.2% | 9 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

### WB tune: stop=3.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=-5.6%, Trades=20, WR=35.0%, Sharpe=-0.12, PF=0.76, DD=-32.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=FRAGILE, Holdout=PASS (+0.0%), Regimes=0/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull (Low Rat... |
| Bootstrap | FAIL | p=0.5025, Sharpe CI=[-3.93, 3.08], WR CI=[20.0%, 60.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-31.1%, Median=$976, Survival=100.0% |
| Regime | FAIL | bull:16t/+7.7%, chop:2t/+0.3%, volatile:2t/-7.0% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 0/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -11.9% | 3 | bull | FAIL |

**Verdict: FRAGILE**

---

### WB tune: stop=6.0%

- **Rules:** `energy_momentum, energy_seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 6.0%
- **Cooldown:** 5 bars

**Performance:** Return=-17.8%, Trades=17, WR=35.3%, Sharpe=-0.25, PF=0.57, DD=-40.7%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=FRAGILE, Holdout=PASS (+0.0%), Regimes=0/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull (Low Rat... |
| Bootstrap | FAIL | p=0.6754, Sharpe CI=[-4.92, 2.67], WR CI=[17.6%, 64.7%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.6%, Median=$824, Survival=100.0% |
| Regime | FAIL | bull:13t/+5.9%, chop:2t/+0.3%, volatile:2t/-20.6% |

**Result: 0/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 0/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +0.0% | 0 | volatile | FAIL |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -16.1% | 3 | bull | FAIL |

**Verdict: FRAGILE**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WB tune: energy 14 rules** | FAIL | FAIL | FAIL | **PASS** | **-0.11** | **-9.2%** | 33 |
| WB tune: stop=3.0% | FAIL | FAIL | **PASS** | FAIL | -0.12 | -5.6% | 20 |
| Sector rules + 8% PT | FAIL | FAIL | **PASS** | FAIL | -0.26 | -17.6% | 20 |
| WB tune: stop=6.0% | FAIL | FAIL | FAIL | FAIL | -0.25 | -17.8% | 17 |

---

## 5. Final Recommendation

**COP partially validates.** Best config: WB tune: energy 14 rules (1/4 gates).

### WB tune: energy 14 rules

- **Rules:** `enhanced_buy_dip, momentum_reversal, trend_continuation, rsi_oversold, macd_bearish_crossover, trend_alignment, golden_cross, trend_break_warning, death_cross, seasonality, energy_momentum, energy_mean_reversion, energy_seasonality, midstream_yield_reversion`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 3 bars

**Performance:** Return=-9.2%, Trades=33, WR=36.4%, Sharpe=-0.11, PF=0.88, DD=-41.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +4.0% [PASS], 2021 Bull... |
| Bootstrap | FAIL | p=0.5297, Sharpe CI=[-2.91, 2.28], WR CI=[21.2%, 51.5%] |
| Monte Carlo | FAIL | Ruin=0.0%, P95 DD=-41.0%, Median=$941, Survival=100.0% |
| Regime | **PASS** | bull:26t/-8.2%, bear:3t/-15.4%, chop:1t/+8.3%, volatile:3t/+15.3% |

**Result: 1/4 gates passed**

#### Walk-Backward Detail

- **Tune period:** 2025-07-09 to 2026-01-05 (+0.0%, 0 trades)
- **Holdout (forward):** 2026-01-06 to 2026-03-07 (+0.0%, 0 trades) **PASS**
- **Regimes passed:** 1/4 (need 3)

| Regime Window | Period | Return | Trades | Dominant | Status |
|---------------|--------|--------|--------|----------|--------|
| 2022 Bear (Rate Hike) | 2022-01-03 to 2022-10-14 | +4.0% | 1 | volatile | **PASS** |
| 2021 Bull (Low Rate Momentum) | 2021-01-04 to 2021-12-31 | +0.0% | 0 | bull | FAIL |
| 2020 Crash + Recovery | 2020-02-19 to 2020-12-31 | -100.0% | 0 | error | FAIL |
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | -13.2% | 9 | bull | FAIL |

**Verdict: REGIME_DEPENDENT**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

