# NEP (NextEra Energy Partners) Validated Optimization Results

**Date:** 2026-03-07
**Period:** 2019-01-01 to 2026-03-07
**Initial Cash:** $1,000
**Validation:** Hybrid Walk-Backward (tune 6mo + holdout 2mo + historical regimes)
**Runtime:** 5.1 minutes
**Category:** C Tier — Renewables
**Bear Beta:** 0.40-0.50

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

NEP -- Contracted renewables portfolio, contrarian deep-value play. C Tier — Renewables.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules (10%/5%) | 1 | 0.0% | -5.4% | -0.97 | 0.00 | -7.1% |
| Full general (10 rules, 10%/5%) | 4 | 25.0% | -5.2% | -0.37 | 0.58 | -14.2% |
| Tighter stops (3 rules, 10%/4%) | 1 | 0.0% | -4.8% | -1.02 | 0.00 | -5.3% |
| Wider PT (3 rules, 12%/5%) | 1 | 0.0% | -5.4% | -0.97 | 0.00 | -7.1% |
| Energy rules (14, 10%/5%) | 4 | 25.0% | -5.2% | -0.37 | 0.58 | -14.2% |
| Sector rules (renewables) | 1 | 0.0% | -5.4% | -0.97 | 0.00 | -7.1% |
| Sector rules + 8% PT | 1 | 0.0% | -4.8% | -1.02 | 0.00 | -5.3% |
---

## 2. Full Validation (Walk-Backward)

### Lean 3 rules (10%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-5.4%, Trades=1, WR=0.0%, Sharpe=-0.97, PF=0.00, DD=-7.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WB tune: stop=3.0% | 1 | 0.0% | -4.8% | -1.02 |
| WB tune: stop=4.0% | 1 | 0.0% | -4.8% | -1.02 |
| WB tune: stop=6.0% | 1 | 0.0% | -5.4% | -0.97 |
| WB tune: ATR x2.5 | 1 | 0.0% | -5.4% | -0.97 |
| WB tune: energy 14 rules | 4 | 25.0% | -5.2% | -0.37 |
| BS tune: conf=0.4 | 1 | 0.0% | -5.4% | -0.97 |
| BS tune: conf=0.45 | 1 | 0.0% | -5.4% | -0.97 |
| BS tune: conf=0.55 | 1 | 0.0% | -5.4% | -0.97 |
| BS tune: +energy_momentum | 1 | 0.0% | -5.4% | -0.97 |
| MC tune: ATR x2.0 | 1 | 0.0% | -5.4% | -0.97 |
| Reg tune: +mean_rev | 1 | 0.0% | -5.4% | -0.97 |

### Full Validation of Top Candidates

### WB tune: PT=8%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 8%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-5.4%, Trades=1, WR=0.0%, Sharpe=-0.97, PF=0.00, DD=-7.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

### WB tune: PT=12%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-5.4%, Trades=1, WR=0.0%, Sharpe=-0.97, PF=0.00, DD=-7.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

### WB tune: PT=15%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 15%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-5.4%, Trades=1, WR=0.0%, Sharpe=-0.97, PF=0.00, DD=-7.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

## 4. Summary Table

| Config | WB | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Lean 3 rules (10%/5%)** | FAIL | FAIL | FAIL | FAIL | **-0.97** | **-5.4%** | 1 |
| WB tune: PT=8% | FAIL | FAIL | FAIL | FAIL | -0.97 | -5.4% | 1 |
| WB tune: PT=12% | FAIL | FAIL | FAIL | FAIL | -0.97 | -5.4% | 1 |
| WB tune: PT=15% | FAIL | FAIL | FAIL | FAIL | -0.97 | -5.4% | 1 |

---

## 5. Final Recommendation

**NEP partially validates.** Best config: Lean 3 rules (10%/5%) (0/4 gates).

### Lean 3 rules (10%/5%)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=-5.4%, Trades=1, WR=0.0%, Sharpe=-0.97, PF=0.00, DD=-7.1%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Backward | FAIL | Too few trades |
| Bootstrap | FAIL | Too few trades |
| Monte Carlo | FAIL | Too few trades |
| Regime | FAIL | Too few trades |

**Result: 0/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor failing gate(s) in live trading
- Re-validate after 6 months of additional data

