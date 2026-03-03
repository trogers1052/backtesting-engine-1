# NET (Cloudflare) Validated Optimization Results

**Date:** 2026-03-03
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 36.0 minutes
**Category:** Mid-cap cloud/security

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

NET — Cloud infrastructure — CDN, security, edge computing. Mid-cap cloud/security.

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 15 | 46.7% | +16.0% | 0.17 | 1.29 | -33.6% |
| Alt A: Full general rules (10 rules, 10%/5%) | 44 | 43.2% | +33.2% | 0.27 | 1.17 | -36.6% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 16 | 43.8% | +23.9% | 0.23 | 1.43 | -31.8% |
| Alt C: Wider PT (3 rules, 12%/5%) | 15 | 46.7% | +23.7% | 0.22 | 1.42 | -32.8% |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | 14 | 50.0% | +31.7% | 0.28 | 1.61 | -27.9% |

**Best baseline selected for validation: Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65)**

---

## 2. Full Validation

### Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65)

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+31.7%, Trades=14, WR=50.0%, Sharpe=0.28, PF=1.61, DD=-27.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.67, Test Sharpe=0.65, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1846, Sharpe CI=[-2.25, 5.84], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.2%, Median equity=$1,389, Survival=100.0% |
| Regime | FAIL | bull:14t/+45.6% |

**Result: 1/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| WF tune: cooldown=7 | 13 | 53.8% | +40.7% | 0.35 |
| WF tune: PT=12% | 14 | 50.0% | +41.8% | 0.32 |
| Regime tune: tighter stop 4% | 14 | 42.9% | +27.5% | 0.30 |
| WF tune: cooldown=3 | 14 | 50.0% | +33.5% | 0.29 |
| BS tune: + volume_breakout | 14 | 50.0% | +31.7% | 0.28 |
| Regime tune: conf=0.65 | 14 | 50.0% | +31.7% | 0.28 |
| BS tune: full rules (10) | 44 | 43.2% | +33.2% | 0.27 |
| WF tune: PT=15% | 12 | 41.7% | +20.8% | 0.20 |
| WF tune: PT=8% | 14 | 50.0% | +18.1% | 0.20 |
| WF tune: conf=0.45 | 15 | 46.7% | +16.0% | 0.17 |
| WF tune: conf=0.55 | 15 | 46.7% | +16.0% | 0.17 |
| WF tune: conf=0.6 | 15 | 46.7% | +16.0% | 0.17 |
| BS tune: conf=0.4 | 15 | 46.7% | +16.0% | 0.17 |

### Full Validation of Top Candidates

### WF tune: cooldown=7

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+40.7%, Trades=13, WR=53.8%, Sharpe=0.35, PF=1.86, DD=-25.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.33, Test Sharpe=0.65, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1507, Sharpe CI=[-1.89, 6.70], WR CI=[30.8%, 76.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.7%, Median equity=$1,498, Survival=100.0% |
| Regime | FAIL | bull:13t/+52.9% |

**Result: 1/4 gates passed**

---

### WF tune: PT=12%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+41.8%, Trades=14, WR=50.0%, Sharpe=0.32, PF=1.80, DD=-27.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.68, Test Sharpe=0.66, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1534, Sharpe CI=[-1.86, 6.26], WR CI=[21.4%, 78.6%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-39.3%, Median equity=$1,520, Survival=100.0% |
| Regime | FAIL | bull:14t/+55.8% |

**Result: 1/4 gates passed**

---

### Regime tune: tighter stop 4%

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 4.0%
- **Cooldown:** 5 bars

**Performance:** Return=+27.5%, Trades=14, WR=42.9%, Sharpe=0.30, PF=1.54, DD=-25.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.39, Test Sharpe=0.60, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.2123, Sharpe CI=[-2.56, 5.37], WR CI=[21.4%, 71.4%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-37.8%, Median equity=$1,320, Survival=100.0% |
| Regime | FAIL | bull:14t/+39.7% |

**Result: 1/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **WF tune: cooldown=7** | FAIL | FAIL | **PASS** | FAIL | **0.35** | **+40.7%** | 13 |
| WF tune: PT=12% | FAIL | FAIL | **PASS** | FAIL | 0.32 | +41.8% | 14 |
| Regime tune: tighter stop 4% | FAIL | FAIL | **PASS** | FAIL | 0.30 | +27.5% | 14 |
| Alt D: Higher confidence (3 rules, 10%/5%, conf=0.65) | FAIL | FAIL | **PASS** | FAIL | 0.28 | +31.7% | 14 |

---

## 5. Final Recommendation

**NET partially validates.** Best config: WF tune: cooldown=7 (1/4 gates).

### WF tune: cooldown=7

- **Rules:** `trend_continuation, seasonality, death_cross`
- **Profit Target:** 10%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+40.7%, Trades=13, WR=53.8%, Sharpe=0.35, PF=1.86, DD=-25.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | FAIL | Train Sharpe=-0.33, Test Sharpe=0.65, Ratio=0% (need >=50%) |
| Bootstrap | FAIL | p=0.1507, Sharpe CI=[-1.89, 6.70], WR CI=[30.8%, 76.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-36.7%, Median equity=$1,498, Survival=100.0% |
| Regime | FAIL | bull:13t/+52.9% |

**Result: 1/4 gates passed**

---

### Deployment Recommendation

- Consider blacklisting or significant restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

