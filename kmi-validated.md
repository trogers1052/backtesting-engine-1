# KMI (Kinder Morgan) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 13.0 minutes
**Category:** B Tier — Midstream
**Bear Beta:** 0.50-0.60

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

KMI -- Largest US natural gas pipeline network, 90% fee-based revenue. B Tier — Midstream.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 11 | 45.5% | +10.0% | 0.10 | 1.27 | -20.5% |
| Full general (10 rules, 10%/5%) | 17 | 35.3% | +12.8% | 0.13 | 1.21 | -26.4% |
| Tighter stops (3 rules, 10%/4%) | 11 | 45.5% | +18.0% | 0.19 | 1.55 | -17.2% |
| Wider PT (3 rules, 12%/5%) | 11 | 45.5% | +21.3% | 0.21 | 1.54 | -20.5% |
| Energy rules (14, 10%/5%) | 17 | 35.3% | +12.8% | 0.13 | 1.21 | -26.4% |
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

**Performance:** Return=+21.3%, Trades=11, WR=45.5%, Sharpe=0.21, PF=1.54, DD=-20.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.2008, Sharpe CI=[-2.87, 7.03], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-27.5%, Median=$1,248, Survival=100.0% |
| Regime | FAIL | bull:7t/+53.3%, bear:1t/-6.0%, volatile:2t/-10.9%, crisis:1t/-9.0% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +10.1% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| BS tune: conf=0.4 | 11 | 45.5% | +21.3% | 0.21 |
| BS tune: conf=0.45 | 11 | 45.5% | +21.3% | 0.21 |
| BS tune: conf=0.55 | 11 | 45.5% | +21.3% | 0.21 |
| BS tune: +energy_momentum | 11 | 45.5% | +21.3% | 0.21 |
| Reg tune: +mean_rev | 11 | 45.5% | +21.3% | 0.21 |
| WB tune: stop=6.0% | 11 | 45.5% | +20.8% | 0.21 |
| WB tune: PT=15% | 10 | 40.0% | +19.7% | 0.20 |
| WB tune: energy 14 rules | 18 | 33.3% | +18.4% | 0.19 |
| WB tune: PT=8% | 12 | 50.0% | +6.8% | 0.06 |
| WB tune: midstream sector rules | 0 | 0.0% | +0.0% | 0.00 |

### Full Validation of Top Candidates

### WB tune: stop=4.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+29.4%, Trades=11, WR=45.5%, Sharpe=0.27, PF=1.86, DD=-17.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1369, Sharpe CI=[-2.13, 7.47], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.8%, Median=$1,341, Survival=100.0% |
| Regime | FAIL | bull:7t/+56.1%, bear:1t/-6.0%, volatile:2t/-6.8%, crisis:1t/-9.0% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +10.1% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### WB tune: stop=3.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 3.0%
- **Cooldown:** 5 bars

**Performance:** Return=+26.8%, Trades=12, WR=41.7%, Sharpe=0.24, PF=1.67, DD=-15.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1621, Sharpe CI=[-2.20, 6.56], WR CI=[16.7%, 75.0%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.6%, Median=$1,309, Survival=100.0% |
| Regime | FAIL | bull:7t/+57.3%, bear:1t/-3.7%, chop:1t/-3.3%, volatile:2t/-9.5%, crisis:1t/-9.0% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +11.6% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### WB tune: ATR x2.5

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars
- **Stop Mode:** ATR x2.5

**Performance:** Return=+23.1%, Trades=11, WR=45.5%, Sharpe=0.22, PF=1.60, DD=-19.5%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1900, Sharpe CI=[-2.74, 7.07], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-26.6%, Median=$1,265, Survival=100.0% |
| Regime | FAIL | bull:7t/+54.6%, bear:1t/-6.0%, volatile:2t/-10.9%, crisis:1t/-9.0% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +11.6% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WB tune: stop=4.0%** | FAIL | FAIL | **PASS** | FAIL | **0.27** | **+29.4%** | 11 |
| WB tune: stop=3.0% | FAIL | FAIL | **PASS** | FAIL | 0.24 | +26.8% | 12 |
| WB tune: ATR x2.5 | FAIL | FAIL | **PASS** | FAIL | 0.22 | +23.1% | 11 |
| Wider PT (3 rules, 12%/5%) | FAIL | FAIL | **PASS** | FAIL | 0.21 | +21.3% | 11 |

---

## 5. Final Recommendation

**KMI partially validates.** Best config: WB tune: stop=4.0% (1/4 gates).

### WB tune: stop=4.0%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+29.4%, Trades=11, WR=45.5%, Sharpe=0.27, PF=1.86, DD=-17.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Verdict=REGIME_DEPENDENT, Holdout=PASS (+0.0%), Regimes=1/4 (need 3) | 2022 Bear (Rate Hike): +0.0% [FAIL], 2021 Bull... |
| Bootstrap | FAIL | p=0.1369, Sharpe CI=[-2.13, 7.47], WR CI=[18.2%, 72.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.8%, Median=$1,341, Survival=100.0% |
| Regime | FAIL | bull:7t/+56.1%, bear:1t/-6.0%, volatile:2t/-6.8%, crisis:1t/-9.0% |

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
| 2023-2024 Choppy Transition | 2023-01-03 to 2024-06-30 | +10.1% | 3 | bull | **PASS** |

**Verdict: REGIME_DEPENDENT**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

