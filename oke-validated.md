# OKE (ONEOK) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 15.5 minutes
**Category:** B Tier — Midstream
**Bear Beta:** 0.60-0.70

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

OKE -- NGL pipelines and processing, post-Magellan diversification. B Tier — Midstream.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 9 | 33.3% | -1.6% | -0.03 | 0.81 | -31.1% |
| Full general (10 rules, 10%/5%) | 15 | 40.0% | -6.5% | -0.04 | 0.86 | -33.9% |
| Tighter stops (3 rules, 10%/4%) | 10 | 30.0% | +4.9% | 0.04 | 1.02 | -26.8% |
| Wider PT (3 rules, 12%/5%) | 9 | 33.3% | +3.5% | 0.04 | 0.97 | -31.1% |
| Energy rules (14, 10%/5%) | 15 | 40.0% | -6.5% | -0.04 | 0.86 | -33.9% |
| Sector rules (midstream) | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |
| Sector rules + 8% PT | 0 | 0.0% | +0.0% | 0.00 | 0.00 | -0.0% |

**Best baseline: Wider PT (3 rules, 12%/5%)**

---

## 2. Full Validation (Walk-Backward)

### Wider PT (3 rules, 12%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+3.5%, Trades=9, WR=33.3%, Sharpe=0.04, PF=0.97, DD=-31.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.3607, Sharpe CI=[-5.29, 6.11], WR CI=[11.1%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.7%, Median=$1,069, Survival=100.0% |
| Regime | FAIL | bull:8t/+19.0%, bear:1t/-8.4% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +9.6% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WB tune: stop=6.0% | 8 | 37.5% | +5.8% | 0.06 |
| WB tune: ATR x2.5 | 9 | 33.3% | +3.5% | 0.04 |
| BS tune: conf=0.4 | 9 | 33.3% | +3.5% | 0.04 |
| BS tune: conf=0.45 | 9 | 33.3% | +3.5% | 0.04 |
| BS tune: +energy_momentum | 9 | 33.3% | +3.5% | 0.04 |
| Reg tune: +mean_rev | 9 | 33.3% | +3.5% | 0.04 |
| WB tune: stop=3.0% | 14 | 21.4% | -1.2% | -0.02 |
| WB tune: PT=8% | 10 | 40.0% | -0.8% | -0.02 |
| WB tune: energy 14 rules | 12 | 33.3% | -9.6% | -0.08 |
| WB tune: midstream sector rules | 0 | 0.0% | +0.0% | 0.00 |

### Full Validation of Top Candidates

### WB tune: stop=4.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+11.6%, Trades=10, WR=30.0%, Sharpe=0.13, PF=1.27, DD=-26.8%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.2836, Sharpe CI=[-4.20, 5.88], WR CI=[10.0%, 70.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.0%, Median=$1,138, Survival=100.0% |
| Regime | FAIL | bull:9t/+20.7%, bear:1t/-4.4% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +9.6% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### WB tune: PT=15%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+10.9%, Trades=9, WR=33.3%, Sharpe=0.12, PF=1.20, DD=-31.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.2928, Sharpe CI=[-4.68, 6.39], WR CI=[11.1%, 77.8%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.7%, Median=$1,136, Survival=100.0% |
| Regime | FAIL | bull:8t/+25.9%, bear:1t/-8.4% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +12.4% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### BS tune: conf=0.55

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+10.6%, Trades=8, WR=37.5%, Sharpe=0.11, PF=1.22, DD=-29.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.2864, Sharpe CI=[-4.74, 8.28], WR CI=[12.5%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.7%, Median=$1,132, Survival=100.0% |
| Regime | **PASS** | bull:6t/+11.3%, bear:1t/-8.4%, chop:1t/+13.5% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +16.0% | 2 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **BS tune: conf=0.55** | FAIL | FAIL | **PASS** | **PASS** | **0.11** | **+10.6%** | 8 |
| WB tune: stop=4.0% | FAIL | FAIL | **PASS** | FAIL | 0.13 | +11.6% | 10 |
| WB tune: PT=15% | FAIL | FAIL | **PASS** | FAIL | 0.12 | +10.9% | 9 |
| Wider PT (3 rules, 12%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.04 | +3.5% | 9 |

---

## 5. Final Recommendation

**OKE partially validates.** Best config: BS tune: conf=0.55 (2/4 gates).

### BS tune: conf=0.55

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.55
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+10.6%, Trades=8, WR=37.5%, Sharpe=0.11, PF=1.22, DD=-29.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.2864, Sharpe CI=[-4.74, 8.28], WR CI=[12.5%, 87.5%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.7%, Median=$1,132, Survival=100.0% |
| Regime | **PASS** | bull:6t/+11.3%, bear:1t/-8.4%, chop:1t/+13.5% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +16.0% | 2 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### Deployment Recommendation

- Conditional deployment with regime restrictions or reduced sizing
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

