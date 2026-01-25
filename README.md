# Backtesting Service

Test decision-engine trading rules against 5 years of historical data from TimescaleDB using **backtrader2**.

## Overview

The backtesting service validates your trading strategies before risking real money. It:

1. Loads historical OHLCV data from TimescaleDB
2. Calculates technical indicators using pandas-ta (matching analytics-service)
3. Evaluates decision-engine rules on each bar
4. Tracks trades with entry/exit reasoning
5. Reports comprehensive performance metrics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List available rules
python -m backtesting --list-rules

# List symbols in database
python -m backtesting --list-symbols

# Run backtest with your primary rule
python -m backtesting --symbol WPM --start 2021-01-01

# Run with multiple rules and show trades
python -m backtesting --symbol WPM \
    --rules buy_dip_in_uptrend,strong_buy_signal \
    --show-trades

# Test multiple symbols
python -m backtesting --symbol WPM,GOLD,NEM --start 2021-01-01

# Export results to JSON
python -m backtesting --symbol WPM --output results.json
```

## Available Rules

Your trading rules from decision-engine:

| Rule | Description |
|------|-------------|
| `buy_dip_in_uptrend` | **Your primary rule**: Buy when RSI dips AND weekly uptrend intact |
| `strong_buy_signal` | Buy when RSI dips in strong triple-SMA uptrend |
| `rsi_oversold` | Buy when RSI is oversold |
| `rsi_overbought` | Sell when RSI is overbought |
| `macd_bullish_crossover` | Buy on MACD bullish crossover |
| `macd_bearish_crossover` | Sell on MACD bearish crossover |
| `weekly_uptrend` | Weekly uptrend (SMA_20 > SMA_50) |
| `monthly_uptrend` | Monthly uptrend (SMA_50 > SMA_200) |
| `golden_cross` | Full trend alignment (SMA_20 > SMA_50 > SMA_200) |

## CLI Options

```
usage: python -m backtesting [options]

Symbol and Date:
  --symbol, -s     Stock symbol(s) to test (comma-separated)
  --start          Start date (YYYY-MM-DD, default: 2021-01-01)
  --end            End date (YYYY-MM-DD, default: today)

Strategy:
  --rules, -r      Rule names to use (comma-separated)
  --min-confidence Minimum confidence threshold (default: 0.6)
  --profit-target  Profit target percentage (default: 0.07)
  --stop-loss      Stop loss percentage (default: 0.05)
  --require-consensus  Require multiple rules to agree

Capital:
  --cash           Initial cash (default: $100,000)
  --commission     Commission rate (default: 0.001)

Output:
  --show-trades, -t  Show individual trade details
  --trade-limit      Max trades to show (default: 10)
  --output, -o       Export results to JSON file
  --quiet, -q        Minimal output

Info:
  --list-rules     List available trading rules
  --list-symbols   List symbols in database
```

## Example Output

```
╭───────────────────────────────────╮
│ Backtest Results: WPM             │
╰───────────────────────────────────╯

Strategy: buy_dip_in_uptrend
Period: 2021-01-01 to 2026-01-24

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric         ┃ Value       ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Initial Capital│ $100,000.00 │
│ Final Value    │ $189,500.00 │
│ Total Return   │ +89.5%      │
│                │             │
│ Total Trades   │ 47          │
│ Winning Trades │ 32          │
│ Losing Trades  │ 15          │
│ Win Rate       │ 68.1%       │
│ Avg Win        │ +8.2%       │
│ Avg Loss       │ -4.1%       │
│                │             │
│ Profit Factor  │ 2.80        │
│ Sharpe Ratio   │ 1.60        │
│ Max Drawdown   │ -12.3%      │
└────────────────┴─────────────┘

Recent Trades:
1. 2025-12-15: BUY @ $52.30 (confidence: 0.82)
   Reason: BUY DIP: solid dip (RSI: 32) in strong uptrend
   Rules: Buy Dip in Uptrend
   2025-12-18: SELL @ $55.95 (+7.0%) - Profit target

2. 2025-11-20: BUY @ $48.10 (confidence: 0.78)
   Reason: BUY DIP: moderate dip (RSI: 38) in uptrend
   Rules: Buy Dip in Uptrend
   2025-11-25: SELL @ $49.50 (+2.9%) - Profit target
```

## Configuration

Copy `.env.example` to `.env`:

```env
# Market Data Database (TimescaleDB)
MARKET_DATA_DB_HOST=localhost
MARKET_DATA_DB_PORT=5433
MARKET_DATA_DB_USER=ingestor
MARKET_DATA_DB_PASSWORD=ingestor
MARKET_DATA_DB_NAME=stock_db

# Default backtest parameters
DEFAULT_START_DATE=2021-01-01
DEFAULT_INITIAL_CASH=100000.0
DEFAULT_PROFIT_TARGET=0.07
DEFAULT_STOP_LOSS=0.05
DEFAULT_MIN_CONFIDENCE=0.6
```

## Python API

```python
from datetime import date
from backtesting import BacktraderRunner, print_report

# Create runner
runner = BacktraderRunner(initial_cash=100000)

# Run backtest
result = runner.run(
    symbol="WPM",
    start_date=date(2021, 1, 1),
    rules=["buy_dip_in_uptrend", "strong_buy_signal"],
    min_confidence=0.6,
    profit_target=0.07,
    stop_loss=0.05,
)

# Display results
print_report(result, show_trades=True)

# Access metrics
print(f"Win rate: {result.win_rate:.1%}")
print(f"Total return: {result.total_return:.1%}")
print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")

# Multi-symbol backtest
results = runner.run_multiple(
    symbols=["WPM", "GOLD", "NEM"],
    start_date=date(2021, 1, 1),
    rules=["buy_dip_in_uptrend"],
)
```

## Architecture

```
TimescaleDB (ohlcv_1day)
        │
        ▼
  TimescaleLoader  ──────►  pandas DataFrame
        │
        ▼
  pandas-ta bridge  ──────►  DataFrame + Indicators
        │
        ▼
  PandasDataWithIndicators  ──────►  backtrader DataFeed
        │
        ▼
  DecisionEngineStrategy  ──────►  Evaluates rules per bar
        │
        ▼
  BacktraderRunner  ──────►  Orchestrates backtest
        │
        ▼
  Analyzers  ──────►  Metrics, trade records
        │
        ▼
  Reporters  ──────►  Console/JSON output
```

## Integration with Decision Engine

The backtesting service imports rules directly from decision-engine:

```python
# In strategies/rule_based_strategy.py
from decision_engine.rules.base import Rule, SymbolContext, SignalType
from decision_engine.rules.registry import RuleRegistry

# Create rules by name
rules = [
    RuleRegistry.create_rule('buy_dip_in_uptrend', {}),
    RuleRegistry.create_rule('strong_buy_signal', {}),
]
```

## Docker

```bash
# Build (requires decision-engine in same parent directory)
docker build -t backtesting-service .

# Run backtest
docker run --network trading-network \
    -e MARKET_DATA_DB_HOST=timescale \
    backtesting-service \
    --symbol WPM --start 2021-01-01
```

## Next Steps

- [ ] Add walk-forward analysis
- [ ] Implement parameter optimization
- [ ] Add Monte Carlo simulation for risk estimation
- [ ] Build equity curve visualization
- [ ] Add compare mode for side-by-side strategy testing
