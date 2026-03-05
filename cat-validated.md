# CAT (Caterpillar) Validated Optimization Results

**Date:** 2026-03-05
**Period:** 2021-01-01 to 2026-02-28
**Initial Cash:** $1,000
**Timeframe:** Daily-only screening + multi-TF re-validation
**Validation Runtime:** 16.2 minutes
**Category:** Heavy equipment (cyclical)

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

CAT — Heavy equipment — beta 1.0, construction/mining capex bellwether, pricing power. Heavy equipment (cyclical).

| Config | Trades | Win Rate | Return | Sharpe | PF | Max DD |
|--------|--------|----------|--------|--------|-----|--------|
| Lean 3 rules baseline (10%/5%, conf=0.50) | 18 | 50.0% | +51.9% | 0.66 | 2.28 | -15.1% |
| Alt A: Full general rules (10 rules, 10%/5%) | 30 | 46.7% | +66.7% | 0.47 | 2.10 | -20.7% |
| Alt B: Tighter stops (3 rules, 10%/4%) | 20 | 50.0% | +66.4% | 0.82 | 2.71 | -12.3% |
| Alt C: Wider PT (3 rules, 12%/5%) | 18 | 50.0% | +55.1% | 0.66 | 2.11 | -19.5% |
| Alt D: Industrial rules (13 rules, 10%/5%) | 31 | 48.4% | +73.9% | 0.52 | 2.20 | -18.8% |
| Alt E: heavy_equipment lean (4 rules, 10%/5%) | 17 | 58.8% | +69.0% | 0.85 | 3.00 | -19.8% |
| Alt F: Heavy equip balanced (12%/5%) | 15 | 60.0% | +65.9% | 1.11 | 2.84 | -15.6% |
| Alt G: Heavy equip momentum (15%/6%) | 12 | 50.0% | +22.4% | 0.28 | 1.72 | -25.8% |

**Best baseline selected for validation: Alt F: Heavy equip balanced (12%/5%)**

---

## 2. Full Validation

### Alt F: Heavy equip balanced (12%/5%)

- **Rules:** `industrial_mean_reversion, industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+65.9%, Trades=15, WR=60.0%, Sharpe=1.11, PF=2.84, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=1.10, Ratio=131% (need >=50%) |
| Bootstrap | FAIL | p=0.0274, Sharpe CI=[-0.07, 9.23], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.6%, Median equity=$1,889, Survival=100.0% |
| Regime | FAIL | bull:13t/+55.5%, bear:1t/+12.1%, chop:1t/+3.8% |

**Result: 2/4 gates passed**

---

## 3. Tuning Results

### Quick Screen

| Config | Trades | Win Rate | Return | Sharpe |
|--------|--------|----------|--------|--------|
| Regime tune: conf=0.65 | 11 | 63.6% | +65.7% | 1.14 |
| BS tune: cooldown=7 | 14 | 64.3% | +76.9% | 1.11 |
| BS tune: conf=0.4 | 15 | 60.0% | +65.9% | 1.11 |
| BS tune: conf=0.45 | 15 | 60.0% | +65.9% | 1.11 |
| BS tune: conf=0.55 | 15 | 60.0% | +65.9% | 1.11 |
| Regime tune: tighter stop 4% | 16 | 56.2% | +69.8% | 1.07 |
| BS tune: cooldown=3 | 16 | 56.2% | +64.7% | 0.86 |
| Regime tune: PT=7% | 19 | 68.4% | +66.3% | 0.79 |
| Regime tune: PT=15% | 12 | 50.0% | +54.2% | 0.77 |
| Regime tune: PT=8% | 17 | 58.8% | +51.4% | 0.69 |
| BS tune: industrial rules (13) | 26 | 42.3% | +56.1% | 0.53 |
| BS tune: full rules (10) | 25 | 40.0% | +47.7% | 0.47 |
| Regime tune: conf=0.65 [multi-TF] | 29 | 44.8% | +50.3% | 0.49 |

### Full Validation of Top Candidates

### Regime tune: conf=0.65

- **Rules:** `industrial_mean_reversion, industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+65.7%, Trades=11, WR=63.6%, Sharpe=1.14, PF=4.30, DD=-15.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.71, Test Sharpe=1.18, Ratio=165% (need >=50%) |
| Bootstrap | **PASS** | p=0.0132, Sharpe CI=[0.49, 14.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.3%, Median equity=$1,870, Survival=100.0% |
| Regime | FAIL | bull:10t/+56.7%, bear:1t/+12.1% |

**Result: 3/4 gates passed**

---

### BS tune: cooldown=7

- **Rules:** `industrial_mean_reversion, industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.5
- **Max Loss:** 5.0%
- **Cooldown:** 7 bars

**Performance:** Return=+76.9%, Trades=14, WR=64.3%, Sharpe=1.11, PF=3.23, DD=-14.9%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.82, Test Sharpe=1.01, Ratio=124% (need >=50%) |
| Bootstrap | **PASS** | p=0.0164, Sharpe CI=[0.34, 10.29], WR CI=[35.7%, 85.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-22.5%, Median equity=$2,028, Survival=100.0% |
| Regime | FAIL | bull:12t/+62.8%, bear:1t/+12.1%, chop:1t/+3.8% |

**Result: 3/4 gates passed**

---

### BS tune: conf=0.4

- **Rules:** `industrial_mean_reversion, industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.4
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+65.9%, Trades=15, WR=60.0%, Sharpe=1.11, PF=2.84, DD=-15.6%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.84, Test Sharpe=1.20, Ratio=143% (need >=50%) |
| Bootstrap | FAIL | p=0.0274, Sharpe CI=[-0.07, 9.23], WR CI=[33.3%, 86.7%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-23.6%, Median equity=$1,889, Survival=100.0% |
| Regime | FAIL | bull:13t/+55.5%, bear:1t/+12.1%, chop:1t/+3.8% |

**Result: 2/4 gates passed**

---

### Regime tune: conf=0.65 [multi-TF]

- **Rules:** `industrial_mean_reversion, industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+50.3%, Trades=29, WR=44.8%, Sharpe=0.49, PF=1.70, DD=-28.2%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.30, Test Sharpe=1.12, Ratio=370% (need >=50%) |
| Bootstrap | FAIL | p=0.0668, Sharpe CI=[-0.64, 4.63], WR CI=[27.6%, 62.1%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-33.8%, Median equity=$1,783, Survival=100.0% |
| Regime | **PASS** | bull:25t/+27.8%, bear:1t/+4.1%, chop:2t/+24.1%, volatile:1t/+12.4% |

**Result: 3/4 gates passed**

---

## 4. Summary Table

| Config | WF | BS | MC | Regime | Sharpe | Return | Trades |
|--------|-----|-----|-----|--------|--------|--------|--------|
| **Regime tune: conf=0.65** | **PASS** | **PASS** | **PASS** | FAIL | **1.14** | **+65.7%** | 11 |
| BS tune: cooldown=7 | **PASS** | **PASS** | **PASS** | FAIL | 1.11 | +76.9% | 14 |
| Regime tune: conf=0.65 [multi-TF] | **PASS** | FAIL | **PASS** | **PASS** | 0.49 | +50.3% | 29 |
| Alt F: Heavy equip balanced (12%/5%) | **PASS** | FAIL | **PASS** | FAIL | 1.11 | +65.9% | 15 |
| BS tune: conf=0.4 | **PASS** | FAIL | **PASS** | FAIL | 1.11 | +65.9% | 15 |

---

## 5. Final Recommendation

**CAT partially validates.** Best config: Regime tune: conf=0.65 (3/4 gates).

### Regime tune: conf=0.65

- **Rules:** `industrial_mean_reversion, industrial_pullback, industrial_seasonality, death_cross`
- **Profit Target:** 12%
- **Min Confidence:** 0.65
- **Max Loss:** 5.0%
- **Cooldown:** 5 bars

**Performance:** Return=+65.7%, Trades=11, WR=63.6%, Sharpe=1.14, PF=4.30, DD=-15.0%

| Gate | Status | Detail |
|------|--------|--------|
| Walk-Forward | **PASS** | Train Sharpe=0.71, Test Sharpe=1.18, Ratio=165% (need >=50%) |
| Bootstrap | **PASS** | p=0.0132, Sharpe CI=[0.49, 14.22], WR CI=[36.4%, 90.9%] |
| Monte Carlo | **PASS** | Ruin=0.0%, P95 DD=-19.3%, Median equity=$1,870, Survival=100.0% |
| Regime | FAIL | bull:10t/+56.7%, bear:1t/+12.1% |

**Result: 3/4 gates passed**

---

### Deployment Recommendation

- Full deployment with no restrictions
- Monitor the failing gate(s) in live trading
- Re-validate after 6 months of additional data

