# S&P 500 ML Prediction & Advisory System

An end-to-end machine learning system that predicts S&P 500 direction, generates actionable
buy/sell/hold/wait recommendations, and continuously improves itself through automated
retraining, model tournaments, and feature evolution.

**⚠️ For educational/research purposes only — not financial advice.**

---

## Quick Start

```bash
# Install dependencies
pip install pandas scikit-learn numpy matplotlib openpyxl

# Optional (recommended)
pip install pygame torch yfinance

# First time: train models interactively
python run.py train

# Daily: get today's market signal
python sp500_daily.py --data SP500_Analysis.xlsx

# Or via the launcher
python run.py
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DAILY PIPELINE                            │
│   sp500_daily.py  →  signal + email + desktop notification       │
├──────────────┬───────────────┬──────────────┬───────────────────┤
│  DATA LAYER  │  MODEL LAYER  │ SIGNAL LAYER │   OUTPUT LAYER    │
│              │               │              │                   │
│  Yahoo       │  Random       │  Composite   │  Terminal         │
│  Finance     │  Forest (RF)  │  Score       │  signal           │
│  ↓           │  ↓            │  (-50 to +50)│  ↓                │
│  27 features │  Gradient     │  ↓           │  HTML email       │
│  engineered  │  Boost (GB)   │  7 action    │  ↓                │
│  ↓           │  ↓            │  zones:      │  Desktop          │
│  StandardScl │  LSTM         │  STRONG BUY  │  notification     │
│              │  (optional)   │  BUY         │  ↓                │
│              │               │  LEAN BULL   │  React dashboard  │
│              │               │  HOLD/WAIT   │  ↓                │
│              │               │  LEAN BEAR   │  HTML report      │
│              │               │  SELL        │                   │
│              │               │  STRONG SELL │                   │
├──────────────┴───────────────┴──────────────┴───────────────────┤
│                     SELF-IMPROVEMENT                             │
│  Walk-forward validation · Model tournaments · Feature evolution │
│  Drift detection · Auto-retraining · Paper trading tracker       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Size | Purpose |
|------|------|---------|
| `sp500_daily.py` | 14 KB | **Run this daily.** Integrated pipeline: data → models → signal → email |
| `sp500_advisor.py` | 22 KB | Signal engine: composite scoring, action generation, backtest |
| `sp500_advisor.jsx` | 49 KB | React dashboard: interactive signal, regime, history, backtest |
| `sp500_predictor.py` | 46 KB | Pygame training app: interactive model training with visual UI |
| `sp500_evolution.py` | 41 KB | Self-improvement: walk-forward, tournaments, drift, features |
| `sp500_live_tracker.py` | 43 KB | Paper trading, alert system, performance Pygame dashboard |
| `sp500_backtester.py` | 48 KB | 10 strategies, Monte Carlo simulation, HTML report generator |
| `sp500_dashboard.jsx` | 25 KB | React dashboard: model performance, feature analysis |
| `run.py` | 7 KB | Unified launcher with dependency checking |
| `README.md` | — | This file |

---

## The Signal System

The advisor combines **ML predictions** with **technical regime analysis** into a weighted
composite score from -50 to +50, which maps to 7 action zones.

### Signal Components (8 factors)

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Direction Ensemble | 30% | RF + GB blended probability of next-day up move |
| Trend Regime | 18% | SMA 50/200 crossover + price vs 200-day SMA |
| Momentum | 12% | MACD histogram + 5-day and 20-day returns |
| Mean Reversion | 10% | RSI overbought/oversold + Bollinger Band position |
| Volatility | 8% | Predicted volatility regime (affects position sizing & wait times) |
| Model Agreement | 8% | Whether RF and GB agree on direction |
| Return Predictions | 8% | 5-day return forecast magnitude |
| Recent Accuracy | 6% | How well the model has predicted recently (confidence) |

### Action Zones

| Score Range | Action | Meaning |
|-------------|--------|---------|
| +35 to +50 | **STRONG BUY** | Full position, enter today |
| +20 to +35 | **BUY** | 70-100% position, enter within 1-3 days |
| +8 to +20 | **LEAN BULLISH** | Partial position, limit orders, wait for confirmation |
| -8 to +8 | **HOLD / WAIT** | No edge — reassess in 3-7 days (or 2-4 weeks in high vol) |
| -20 to -8 | **LEAN BEARISH** | Trim 20-30%, tighten stops, wait 1-3 weeks |
| -35 to -20 | **SELL** | Reduce to 30-50%, sell rallies, defensive positions |
| -50 to -35 | **STRONG SELL** | 70%+ cash, sell immediately, consider hedges |

### Backtest Performance (2011–2026)

Tested on 3,791 out-of-sample trading days:

| Strategy | Return | Max Drawdown | Sharpe | Exposure |
|----------|--------|-------------|--------|----------|
| Buy & Hold | +437.5% | **-33.9%** | 0.530 | 100% |
| Signal Binary | +141.6% | **-16.9%** | 0.426 | 47% |
| Signal Scaled | +127.9% | **-10.2%** | 0.459 | 96% |
| Signal Conservative | +58.3% | **-8.7%** | 0.231 | 47% |

**Key insight:** The system's value is drawdown protection. Signal Binary cuts max drawdown
from -33.9% to -16.9% while still capturing meaningful gains.

---

## Daily Usage

### Option 1: Full automated pipeline (recommended)

```bash
python sp500_daily.py --data SP500_Analysis.xlsx
```

This runs the complete cycle:
1. Fetches new data from Yahoo Finance (if yfinance installed)
2. Engineers 27 technical features
3. Trains RF + GB models on full history
4. Scores yesterday's prediction (if any)
5. Generates today's signal with explanation + suggested actions
6. Sends HTML email alert (if configured)
7. Sends desktop notification
8. Logs signal to history

### Option 2: Signal only (no email)

```bash
python sp500_daily.py --signal-only --data SP500_Analysis.xlsx
```

### Option 3: Interactive launcher

```bash
python run.py
```

Menu options:
- **4. Today's signal** — Generate buy/sell/hold recommendation
- **5. Backtest signals** — Historical performance of signal strategies
- **8. Full backtest + report** — 10 strategies + Monte Carlo → HTML report

---

## Setting Up Automation

### Cron (Linux/Mac) — Recommended

Run at 6:30 PM ET every weekday (after market close):

```bash
crontab -e
# Add this line:
30 18 * * 1-5 cd /path/to/project && /usr/bin/python3 sp500_daily.py >> sp500_logs/daily.log 2>&1
```

### Windows Task Scheduler

1. Open Task Scheduler → Create Basic Task
2. Trigger: Weekly, Mon-Fri at 6:30 PM
3. Action: Start a program
   - Program: `python`
   - Arguments: `sp500_daily.py --data SP500_Analysis.xlsx`
   - Start in: `C:\path\to\project`

### GitHub Actions

Create `.github/workflows/daily_signal.yml`:

```yaml
name: Daily S&P 500 Signal
on:
  schedule:
    - cron: '30 22 * * 1-5'  # 6:30 PM ET = 22:30 UTC
jobs:
  signal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install pandas scikit-learn numpy openpyxl yfinance
      - run: python sp500_daily.py --data SP500_Analysis.xlsx
```

---

## Email Alerts

### Setup

```bash
python sp500_daily.py --setup-email
```

For Gmail:
1. Go to https://myaccount.google.com/apppasswords
2. Generate an App Password for "Mail"
3. Enter your email and the 16-character app password when prompted

### Preview

```bash
python sp500_daily.py --email-preview --data SP500_Analysis.xlsx
```

Opens an HTML file showing exactly what the email will look like.

---

## Self-Improvement System

The evolution engine (`sp500_evolution.py`) provides 6 layers of self-improvement:

1. **Auto-updating data** — Fetches new data from Yahoo Finance daily
2. **Walk-forward validation** — Expanding window evaluation, no future leakage
3. **Model tournament** — 11 model configs compete head-to-head, champions promoted
4. **Drift detection** — Alerts when model accuracy degrades vs historical baseline
5. **Feature evolution** — Tests 9 experimental features, keeps winners
6. **Automated scheduling** — Cron, Task Scheduler, or GitHub Actions

Run the full evolution cycle:

```bash
python sp500_evolution.py --update --data SP500_Analysis.xlsx
```

---

## Advanced Backtesting

The backtester (`sp500_backtester.py`) tests 10 trading strategies:

- Long/Cash, Long/Short (RF and GB)
- Confidence-weighted positioning
- Kelly criterion sizing
- Volatility targeting (15% annual)
- Drawdown circuit breaker
- Momentum filter (200-day SMA)
- Mean reversion (contrarian)
- RF + GB ensemble voting

Plus Monte Carlo simulation (1,000 bootstrapped 1-year paths).

```bash
python sp500_backtester.py --full --data SP500_Analysis.xlsx
```

Generates a self-contained HTML report with embedded charts.

---

## Directory Structure

Auto-created on first run:

```
project/
├── SP500_Analysis.xlsx          ← Your data (19,157 daily rows)
├── sp500_daily.py               ← Run this daily
├── sp500_advisor.py             ← Signal engine
├── sp500_advisor.jsx            ← React dashboard
├── sp500_predictor.py           ← Interactive trainer
├── sp500_evolution.py           ← Self-improvement
├── sp500_live_tracker.py        ← Paper trading
├── sp500_backtester.py          ← Strategy backtester
├── sp500_dashboard.jsx          ← React dashboard
├── run.py                       ← Launcher
├── sp500_data/
│   └── sp500_master.csv         ← Continuously updated dataset
├── sp500_models/
│   ├── live/live_models.pkl     ← Production models
│   ├── advisor_models.pkl       ← Advisor models
│   ├── tournament_results.json  ← Model rankings
│   └── backtest_results.pkl     ← Backtest results
└── sp500_logs/
    ├── signal_history.json      ← All advisor signals
    ├── daily_state.json         ← Pipeline state
    ├── paper_trades.json        ← Paper trading record
    ├── evolution.log            ← Training logs
    ├── drift_log.json           ← Degradation alerts
    ├── alert_config.json        ← Email settings
    ├── email_preview.html       ← Latest email preview
    └── reports/
        └── backtest_report_*.html
```

---

## Dependencies

**Required:**
```
pandas scikit-learn numpy matplotlib openpyxl
```

**Recommended:**
```
pygame           # Interactive training dashboards
torch            # LSTM models + DQN for snake AI
yfinance         # Auto-fetch new market data
```

Install everything:
```bash
pip install pandas scikit-learn numpy matplotlib openpyxl pygame torch yfinance
```

---

## Model Details

- **Training data:** 19,157 daily rows (Jan 1950 – Feb 2026)
- **Features:** 27 engineered (returns, SMAs, MACD, RSI, BB, ATR, volume, volatility)
- **Models:** Random Forest (100-300 trees), Gradient Boosting (100-300 estimators), LSTM (optional)
- **Ensemble:** RF × 0.55 + GB × 0.45 weighted probability
- **Targets:** Next-day direction, 1-day return, 5-day return, 5-day volatility
- **RF accuracy:** ~54.6% out-of-sample (2011–2026)
- **Ensemble Sharpe:** 0.614 (vs 0.530 buy-and-hold) in strategy backtests
