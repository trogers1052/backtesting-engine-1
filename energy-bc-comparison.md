# Energy B/C Tier -- Hybrid Walk-Backward Comparison Matrix

**Date:** 2026-03-07
**Validation:** Walk-Backward (tune 6mo, holdout 2mo, 4 historical regimes)

---

## Comparison Matrix

| Symbol | Tier | Bear Beta | Best Config | Gates | WB | BS | MC | Reg | Sharpe | Return | Trades | WR |
|--------|------|-----------|-------------|-------|----|----|----|-----|--------|--------|--------|----|
| **CVX** | B Tier — Integrated | 0.70 | WB tune: PT=8% | 1/4 | FAIL | FAIL | PASS | FAIL | -0.00 | +0.6% | 27 | 44% |
| **KMI** | B Tier — Midstream | 0.50-0.60 | WB tune: stop=4.0% | 1/4 | FAIL | FAIL | PASS | FAIL | 0.27 | +29.4% | 11 | 45% |
| **OKE** | B Tier — Midstream | 0.60-0.70 | BS tune: conf=0.55 | 2/4 | FAIL | FAIL | PASS | PASS | 0.11 | +10.6% | 8 | 38% |
| **COP** | B Tier — Upstream E&P | 0.90-1.00 | WB tune: energy 14 rules | 1/4 | FAIL | FAIL | FAIL | PASS | -0.11 | -9.2% | 33 | 36% |
| **PSX** | C Tier — Refining | 0.80 | Full general (10 rules, 10%/5%) | 1/4 | FAIL | FAIL | PASS | FAIL | 0.59 | +30.9% | 18 | 44% |
| **FANG** | C Tier — Upstream E&P | 1.20-1.40 | Full general (10 rules, 10%/5%) | 2/4 | FAIL | FAIL | PASS | PASS | 0.85 | +26.2% | 22 | 41% |
| **NEP** | C Tier — Renewables | 0.40-0.50 | Lean 3 rules (10%/5%) | 0/4 | FAIL | FAIL | FAIL | FAIL | -0.97 | -5.4% | 1 | 0% |

---

## Walk-Backward Detail

### CVX (Chevron)

- Tune: +0.0% (0t)
- Holdout: +0.0% (0t) **PASS**
- Regimes: 1/4
- Verdict: **REGIME_DEPENDENT**

| Regime | Return | Trades | Status |
|--------|--------|--------|--------|
| 2022 Bear (Rate Hike) | +3.0% | 1 | **PASS** |
| 2021 Bull (Low Rate Momentum) | +0.0% | 0 | FAIL |
| 2020 Crash + Recovery | -100.0% | 0 | FAIL |
| 2023-2024 Choppy Transition | -11.6% | 5 | FAIL |

### KMI (Kinder Morgan)

- Tune: +0.0% (0t)
- Holdout: +0.0% (0t) **PASS**
- Regimes: 1/4
- Verdict: **REGIME_DEPENDENT**

| Regime | Return | Trades | Status |
|--------|--------|--------|--------|
| 2022 Bear (Rate Hike) | +0.0% | 0 | FAIL |
| 2021 Bull (Low Rate Momentum) | +0.0% | 0 | FAIL |
| 2020 Crash + Recovery | -100.0% | 0 | FAIL |
| 2023-2024 Choppy Transition | +10.1% | 3 | **PASS** |

### OKE (ONEOK)

- Tune: +0.0% (0t)
- Holdout: +0.0% (0t) **PASS**
- Regimes: 1/4
- Verdict: **REGIME_DEPENDENT**

| Regime | Return | Trades | Status |
|--------|--------|--------|--------|
| 2022 Bear (Rate Hike) | +0.0% | 0 | FAIL |
| 2021 Bull (Low Rate Momentum) | +0.0% | 0 | FAIL |
| 2020 Crash + Recovery | -100.0% | 0 | FAIL |
| 2023-2024 Choppy Transition | +16.0% | 2 | **PASS** |

### COP (ConocoPhillips)

- Tune: +0.0% (0t)
- Holdout: +0.0% (0t) **PASS**
- Regimes: 1/4
- Verdict: **REGIME_DEPENDENT**

| Regime | Return | Trades | Status |
|--------|--------|--------|--------|
| 2022 Bear (Rate Hike) | +4.0% | 1 | **PASS** |
| 2021 Bull (Low Rate Momentum) | +0.0% | 0 | FAIL |
| 2020 Crash + Recovery | -100.0% | 0 | FAIL |
| 2023-2024 Choppy Transition | -13.2% | 9 | FAIL |

### PSX (Phillips 66)

- Tune: +0.0% (0t)
- Holdout: +0.0% (0t) **PASS**
- Regimes: 1/4
- Verdict: **REGIME_DEPENDENT**

| Regime | Return | Trades | Status |
|--------|--------|--------|--------|
| 2022 Bear (Rate Hike) | +0.0% | 0 | FAIL |
| 2021 Bull (Low Rate Momentum) | +0.0% | 0 | FAIL |
| 2020 Crash + Recovery | -100.0% | 0 | FAIL |
| 2023-2024 Choppy Transition | +20.3% | 6 | **PASS** |

### FANG (Diamondback Energy)

- Tune: +0.0% (0t)
- Holdout: +0.0% (0t) **PASS**
- Regimes: 1/4
- Verdict: **REGIME_DEPENDENT**

| Regime | Return | Trades | Status |
|--------|--------|--------|--------|
| 2022 Bear (Rate Hike) | +0.0% | 0 | FAIL |
| 2021 Bull (Low Rate Momentum) | +0.0% | 0 | FAIL |
| 2020 Crash + Recovery | -100.0% | 0 | FAIL |
| 2023-2024 Choppy Transition | +3.6% | 6 | **PASS** |

---

### NEP (NextEra Energy Partners)

- Too few trades (1) — all gates auto-fail
- Verdict: **FRAGILE** — untradeable with current rules

---

## Key Findings

### 1. Walk-Backward Gate: Universal Failure

**Every single energy stock failed the Walk-Backward gate.** The pattern is consistent:
- **Tune period (last 6 months):** 0 trades for all 7 stocks — rules don't fire on recent energy price action
- **Holdout (last 2 months):** 0 trades, but passes by default (no loss = breakeven)
- **2020 Crash + Recovery:** Missing data for most stocks (data starts 2021+)
- **2021 Bull:** 0 trades — rules don't generate signals during low-volatility bull runs
- **2022 Bear:** Only CVX (+3.0%, 1 trade) and COP (+4.0%, 1 trade) generated any trades
- **2023-2024 Chop:** Best regime — KMI (+10.1%), OKE (+16.0%), PSX (+20.3%), FANG (+3.6%), COP (-13.2%)

**Root cause:** The rules are tuned for mean-reversion/momentum patterns that only fire during choppy/transitional markets. They're silent during strong trends (bull or bear) and on the recent 6-month tune window.

### 2. Data Coverage Gap

Most energy stocks only have data from 2021+, making the 2020 Crash regime window untestable. This automatically costs 1 of 4 regime windows, making the 3/4 pass threshold nearly impossible.

### 3. Bootstrap Gate: Universal Failure

No energy stock achieved statistical significance (p < 0.05). All have Sharpe CIs spanning zero. Root cause: too few trades (1-33 range) for statistical power.

### 4. Best Performers (by Gates Passed)

| Rank | Symbol | Gates | Sharpe | Return | Why |
|------|--------|-------|--------|--------|-----|
| 1 | **FANG** | 2/4 | 0.85 | +26.2% | Best Sharpe, passes Monte Carlo + Regime |
| 2 | **OKE** | 2/4 | 0.11 | +10.6% | Passes Monte Carlo + Regime |
| 3 | PSX | 1/4 | 0.59 | +30.9% | High return but regime-dependent |
| 4 | KMI | 1/4 | 0.27 | +29.4% | Good return, few trades |
| 5 | CVX | 1/4 | -0.00 | +0.6% | Barely positive |
| 6 | COP | 1/4 | -0.11 | -9.2% | Negative return, most trades (33) |
| 7 | NEP | 0/4 | -0.97 | -5.4% | Untradeable (1 trade total) |

### 5. Hybrid Walk-Backward vs Old Walk-Forward

The hybrid approach is **significantly more demanding** than the old walk-forward validation:
- Walk-forward often passed because it tested on arbitrary time slices
- Walk-backward tests against **labeled historical regimes** — exposing that energy stocks only generate signals in specific market conditions
- The tune-on-recent + validate-on-historical structure reveals that current rules don't fire on recent energy price action at all (0 trades in tune window)

### 6. Actionable Conclusions

- **FANG and OKE** are the only conditionally deployable energy stocks (2/4 gates)
- **NEP** should be removed from the tradeable universe entirely
- **All energy stocks need rule refinement** — current rules are too restrictive for energy sector dynamics
- **Data gap:** Need to backfill 2019-2020 data for proper regime coverage
- **Rule development needed:** Energy-specific rules that fire during strong trends, not just choppy reversions
