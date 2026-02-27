#!/usr/bin/env python3
"""
sp500_evolution.py — Self-Improving Market Prediction System
Plugs into sp500_predictor.py and adds:

  1. AUTO-UPDATE     Fetch new daily S&P 500 data from Yahoo Finance
  2. WALK-FORWARD    Retrain on expanding windows, test on fresh data only
  3. MODEL TOURNAMENT  Compete model configs, promote winners, retire losers
  4. DRIFT DETECTION   Flag when live accuracy degrades vs training accuracy
  5. FEATURE EVOLUTION Test new feature combos, keep what improves performance
  6. FULL LOGGING      Version every model, track all metrics over time

Usage:
    python sp500_evolution.py              # Interactive menu
    python sp500_evolution.py --update     # Fetch new data + retrain (cron-friendly)
    python sp500_evolution.py --dashboard  # Launch Pygame performance tracker

Requires: pandas, scikit-learn, numpy, matplotlib
Optional: pygame (for dashboard), torch (for LSTM), yfinance (for easy data fetch)

⚠️ For educational/research purposes only — not financial advice.
"""

import os
import sys
import json
import time
import hashlib
import pickle
import logging
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              mean_absolute_error, r2_score)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ========================== CONFIG ========================== #
DATA_DIR = "sp500_data"
MODEL_DIR = "sp500_models"
LOG_DIR = "sp500_logs"
HISTORY_FILE = os.path.join(LOG_DIR, "evolution_history.json")
DRIFT_LOG = os.path.join(LOG_DIR, "drift_log.json")
TOURNAMENT_FILE = os.path.join(MODEL_DIR, "tournament_results.json")

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "evolution.log"))
    ]
)
log = logging.getLogger("evolution")


# ========================== 1. DATA AUTO-UPDATER ========================== #
class DataUpdater:
    """Fetch new S&P 500 data and merge with existing dataset."""

    MASTER_CSV = os.path.join(DATA_DIR, "sp500_master.csv")
    YAHOO_URL = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC"

    def __init__(self):
        self.df = None

    def load_existing(self, excel_path=None):
        """Load from master CSV or original Excel file."""
        if os.path.exists(self.MASTER_CSV):
            self.df = pd.read_csv(self.MASTER_CSV, parse_dates=["Date"])
            log.info(f"Loaded master CSV: {len(self.df)} rows, "
                     f"last date: {self.df['Date'].iloc[-1].date()}")
            return True

        # First run: import from Excel
        if excel_path and os.path.exists(excel_path):
            self.df = pd.read_excel(excel_path, sheet_name="Daily Data")
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            self.df.to_csv(self.MASTER_CSV, index=False)
            log.info(f"Imported Excel → CSV: {len(self.df)} rows")
            return True

        return False

    def fetch_new_data(self):
        """Fetch recent data from Yahoo Finance."""
        if self.df is None:
            log.error("No existing data loaded. Call load_existing() first.")
            return False

        last_date = self.df['Date'].iloc[-1]
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        if start >= today:
            log.info("Data is already up to date.")
            return False

        # Try yfinance first (cleanest API)
        try:
            import yfinance as yf
            ticker = yf.Ticker("^GSPC")
            new = ticker.history(start=start, end=today)
            if new.empty:
                log.info("No new data available from yfinance.")
                return False
            new = new.reset_index()
            new = new.rename(columns={"Stock Splits": "Stock_Splits"})
            new = new[["Date", "Open", "High", "Low", "Close", "Volume"]]
            new["Adj Close"] = new["Close"]
            new["Date"] = pd.to_datetime(new["Date"]).dt.tz_localize(None)
            log.info(f"yfinance: fetched {len(new)} new rows")
        except ImportError:
            # Fallback: urllib direct download
            try:
                import urllib.request
                import io
                p1 = int(pd.Timestamp(start).timestamp())
                p2 = int(pd.Timestamp(today).timestamp())
                url = (f"{self.YAHOO_URL}?period1={p1}&period2={p2}"
                       f"&interval=1d&events=history&includeAdjustedClose=true")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=30)
                raw = resp.read().decode("utf-8")
                new = pd.read_csv(io.StringIO(raw), parse_dates=["Date"])
                if new.empty:
                    log.info("No new data available from Yahoo direct.")
                    return False
                log.info(f"Yahoo direct: fetched {len(new)} new rows")
            except Exception as e:
                log.warning(f"Yahoo fetch failed: {e}")
                log.info("Tip: Install yfinance for reliable updates: pip install yfinance")
                return False

        # Merge — avoid duplicates
        existing_dates = set(self.df['Date'].dt.date)
        new = new[~new['Date'].dt.date.isin(existing_dates)]

        if new.empty:
            log.info("All fetched data already exists.")
            return False

        self.df = pd.concat([self.df, new], ignore_index=True)
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        self.df.to_csv(self.MASTER_CSV, index=False)
        log.info(f"Added {len(new)} new rows. Total: {len(self.df)} rows, "
                 f"latest: {self.df['Date'].iloc[-1].date()}")
        return True


# ========================== 2. FEATURE ENGINEERING ========================== #
class FeatureEngine:
    """Modular feature engineering with evolution support."""

    # Base features (proven performers from initial analysis)
    BASE_FEATURES = {
        "returns": [1, 2, 3, 5, 10, 20],
        "sma_ratios": [5, 10, 20, 50, 200],
        "sma_cross": [(5, 20), (20, 50), (50, 200)],
        "macd": True,
        "rsi": [14],
        "bollinger": [20],
        "atr": [14],
        "volatility": [5, 10, 20],
        "volume_ratio": [20],
        "range_gap": True,
        "day_of_week": True,
    }

    # Experimental features (tested during evolution)
    EXPERIMENTAL_FEATURES = {
        "rsi_multi": [7, 21],          # Additional RSI periods
        "volatility_long": [60],        # Longer vol lookback
        "returns_long": [40, 60],       # Longer return windows
        "sma_ratios_ext": [100],        # Additional SMA
        "obv": True,                    # On-balance volume
        "momentum": [10, 20],           # Price momentum
        "mean_reversion": [5, 20],      # Z-score of returns
        "month_of_year": True,          # Seasonality
        "vix_proxy": True,              # Intraday range as vol proxy
    }

    def __init__(self, feature_config=None):
        self.config = feature_config or self.BASE_FEATURES.copy()
        self.feature_cols = []

    def engineer(self, df):
        """Apply all configured features to dataframe. Returns feature column names."""
        df = df.copy()
        cols = []

        # Returns
        if "returns" in self.config:
            for n in self.config["returns"]:
                col = f'ret_{n}d'
                df[col] = df['Close'].pct_change(n)
                cols.append(col)

        # Extended returns
        if "returns_long" in self.config:
            for n in self.config["returns_long"]:
                col = f'ret_{n}d'
                df[col] = df['Close'].pct_change(n)
                cols.append(col)

        # SMAs and ratios
        if "sma_ratios" in self.config:
            for n in self.config["sma_ratios"]:
                df[f'sma_{n}'] = df['Close'].rolling(n).mean()
                col = f'close_sma_{n}_ratio'
                df[col] = df['Close'] / df[f'sma_{n}']
                cols.append(col)

        if "sma_ratios_ext" in self.config:
            for n in self.config["sma_ratios_ext"]:
                df[f'sma_{n}'] = df['Close'].rolling(n).mean()
                col = f'close_sma_{n}_ratio'
                df[col] = df['Close'] / df[f'sma_{n}']
                cols.append(col)

        # SMA crosses
        if "sma_cross" in self.config:
            for short, long in self.config["sma_cross"]:
                if f'sma_{short}' not in df:
                    df[f'sma_{short}'] = df['Close'].rolling(short).mean()
                if f'sma_{long}' not in df:
                    df[f'sma_{long}'] = df['Close'].rolling(long).mean()
                col = f'sma_{short}_{long}_ratio'
                df[col] = df[f'sma_{short}'] / df[f'sma_{long}']
                cols.append(col)

        # MACD
        if self.config.get("macd"):
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            cols.extend(['macd', 'macd_signal', 'macd_hist'])

        # RSI
        for rsi_key in ["rsi", "rsi_multi"]:
            if rsi_key in self.config:
                delta = df['Close'].diff()
                for period in self.config[rsi_key]:
                    gain = delta.clip(lower=0).rolling(period).mean()
                    loss = (-delta.clip(upper=0)).rolling(period).mean()
                    rs = gain / loss.replace(0, np.nan)
                    col = f'rsi_{period}'
                    df[col] = 100 - (100 / (1 + rs))
                    cols.append(col)

        # Bollinger Bands
        if "bollinger" in self.config:
            for period in self.config["bollinger"]:
                bb_sma = df['Close'].rolling(period).mean()
                bb_std = df['Close'].rolling(period).std()
                col = f'bb_pct_{period}'
                df[col] = (df['Close'] - (bb_sma - 2 * bb_std)) / (4 * bb_std)
                cols.append(col)

        # ATR
        if "atr" in self.config:
            for period in self.config["atr"]:
                hl = df['High'] - df['Low']
                hc = (df['High'] - df['Close'].shift()).abs()
                lc = (df['Low'] - df['Close'].shift()).abs()
                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                col = f'atr_pct_{period}'
                df[col] = tr.rolling(period).mean() / df['Close']
                cols.append(col)

        # Volatility
        for vol_key in ["volatility", "volatility_long"]:
            if vol_key in self.config:
                ret_1d = df['Close'].pct_change()
                for n in self.config[vol_key]:
                    col = f'vol_{n}d'
                    df[col] = ret_1d.rolling(n).std() * np.sqrt(252)
                    cols.append(col)

        # Volume ratio
        if "volume_ratio" in self.config:
            for n in self.config["volume_ratio"]:
                col = f'vol_ratio_{n}'
                df[col] = df['Volume'] / df['Volume'].rolling(n).mean()
                cols.append(col)

        # Range & gap
        if self.config.get("range_gap"):
            df['hl_range'] = (df['High'] - df['Low']) / df['Close']
            df['gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
            cols.extend(['hl_range', 'gap'])

        # Day of week
        if self.config.get("day_of_week"):
            df['dow'] = df['Date'].dt.dayofweek
            cols.append('dow')

        # Momentum
        if "momentum" in self.config:
            for n in self.config["momentum"]:
                col = f'momentum_{n}'
                df[col] = df['Close'] / df['Close'].shift(n) - 1
                cols.append(col)

        # Mean reversion (z-score)
        if "mean_reversion" in self.config:
            ret_1d = df['Close'].pct_change()
            for n in self.config["mean_reversion"]:
                col = f'zscore_{n}'
                df[col] = (ret_1d - ret_1d.rolling(n).mean()) / ret_1d.rolling(n).std()
                cols.append(col)

        # Month of year
        if self.config.get("month_of_year"):
            df['month'] = df['Date'].dt.month
            cols.append('month')

        # VIX proxy (intraday range normalized)
        if self.config.get("vix_proxy"):
            df['vix_proxy'] = ((df['High'] - df['Low']) / df['Close']).rolling(10).mean() * np.sqrt(252)
            cols.append('vix_proxy')

        # OBV
        if self.config.get("obv"):
            sign = np.sign(df['Close'].diff()).fillna(0)
            df['obv'] = (sign * df['Volume']).cumsum()
            df['obv_norm'] = df['obv'] / df['obv'].rolling(50).mean()
            cols.append('obv_norm')

        self.feature_cols = cols
        return df, cols

    def add_targets(self, df):
        """Add prediction targets."""
        df['target_ret_1d'] = df['Close'].shift(-1) / df['Close'] - 1
        df['target_dir_1d'] = (df['target_ret_1d'] > 0).astype(int)
        df['target_ret_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target_vol_5d'] = df['Close'].pct_change().shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
        return df


# ========================== 3. WALK-FORWARD ENGINE ========================== #
class WalkForwardEngine:
    """Expanding-window walk-forward training and evaluation."""

    def __init__(self, min_train_years=10, retrain_every_days=63, test_window_days=63):
        self.min_train_years = min_train_years
        self.retrain_every = retrain_every_days   # ~quarterly
        self.test_window = test_window_days       # ~quarter forward test

    def generate_folds(self, dates):
        """Generate walk-forward (train_end, test_start, test_end) indices."""
        folds = []
        min_train = self.min_train_years * 252  # approx trading days
        n = len(dates)

        if n < min_train + self.test_window:
            log.warning("Not enough data for walk-forward.")
            return []

        fold_start = min_train
        while fold_start + self.test_window <= n:
            train_end = fold_start
            test_start = fold_start
            test_end = min(fold_start + self.test_window, n)
            folds.append((train_end, test_start, test_end))
            fold_start += self.retrain_every

        return folds

    def run(self, df, feature_cols, model_configs, progress_cb=None):
        """Run walk-forward over all folds and model configs.
        
        Returns:
            dict: {model_name: {fold_metrics: [...], aggregate: {...}}}
        """
        target_cols = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']
        valid = df.dropna(subset=feature_cols + target_cols).copy()
        dates = valid['Date'].values
        X = valid[feature_cols].values
        y_dict = {col: valid[col].values for col in target_cols}

        folds = self.generate_folds(dates)
        if not folds:
            return {}

        log.info(f"Walk-forward: {len(folds)} folds × {len(model_configs)} configs")

        results = defaultdict(lambda: {"fold_metrics": [], "predictions": []})
        total_work = len(folds) * len(model_configs)
        done = 0

        for fi, (train_end, test_start, test_end) in enumerate(folds):
            X_train = X[:train_end]
            X_test = X[test_start:test_end]
            test_dates_fold = dates[test_start:test_end]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            for config_name, config in model_configs.items():
                model = config["factory"]()
                target_col = config["target"]
                y_tr = y_dict[target_col][:train_end]
                y_te = y_dict[target_col][test_start:test_end]

                model.fit(X_train_s, y_tr)
                y_pred = model.predict(X_test_s)

                metrics = {"fold": fi, "train_size": train_end, "test_size": test_end - test_start}

                if config.get("task") == "classification":
                    metrics["accuracy"] = accuracy_score(y_te, y_pred)
                    metrics["f1"] = f1_score(y_te, y_pred, zero_division=0)
                    if hasattr(model, 'predict_proba'):
                        metrics["probabilities"] = model.predict_proba(X_test_s)[:, 1].tolist()
                else:
                    metrics["rmse"] = np.sqrt(mean_squared_error(y_te, y_pred))
                    metrics["r2"] = r2_score(y_te, y_pred)
                    metrics["mae"] = mean_absolute_error(y_te, y_pred)

                results[config_name]["fold_metrics"].append(metrics)
                done += 1

                if progress_cb:
                    progress_cb(f"Fold {fi+1}/{len(folds)} | {config_name}", done / total_work)

        # Aggregate
        for name in results:
            folds_m = results[name]["fold_metrics"]
            agg = {}
            for key in ["accuracy", "f1", "rmse", "r2", "mae"]:
                vals = [f[key] for f in folds_m if key in f]
                if vals:
                    agg[f"mean_{key}"] = np.mean(vals)
                    agg[f"std_{key}"] = np.std(vals)
                    agg[f"min_{key}"] = np.min(vals)
                    agg[f"max_{key}"] = np.max(vals)
            agg["n_folds"] = len(folds_m)
            results[name]["aggregate"] = agg

        return dict(results)


# ========================== 4. MODEL TOURNAMENT ========================== #
class ModelTournament:
    """Compete model configurations, promote winners, retire losers."""

    # Candidate model configurations
    CANDIDATES = {
        # --- Direction classifiers ---
        "RF_dir_200_d10": {
            "factory": lambda: RandomForestClassifier(n_estimators=200, max_depth=10,
                        min_samples_leaf=20, random_state=42, n_jobs=-1),
            "target": "target_dir_1d", "task": "classification", "family": "direction"
        },
        "RF_dir_500_d12": {
            "factory": lambda: RandomForestClassifier(n_estimators=500, max_depth=12,
                        min_samples_leaf=15, random_state=42, n_jobs=-1),
            "target": "target_dir_1d", "task": "classification", "family": "direction"
        },
        "RF_dir_300_d8": {
            "factory": lambda: RandomForestClassifier(n_estimators=300, max_depth=8,
                        min_samples_leaf=30, random_state=42, n_jobs=-1),
            "target": "target_dir_1d", "task": "classification", "family": "direction"
        },
        "GB_dir_200_lr05": {
            "factory": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5,
                        learning_rate=0.05, min_samples_leaf=20, random_state=42),
            "target": "target_dir_1d", "task": "classification", "family": "direction"
        },
        "GB_dir_300_lr03": {
            "factory": lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4,
                        learning_rate=0.03, min_samples_leaf=25, random_state=42),
            "target": "target_dir_1d", "task": "classification", "family": "direction"
        },
        "GB_dir_500_lr01": {
            "factory": lambda: GradientBoostingClassifier(n_estimators=500, max_depth=3,
                        learning_rate=0.01, min_samples_leaf=30, random_state=42),
            "target": "target_dir_1d", "task": "classification", "family": "direction"
        },

        # --- Volatility regressors ---
        "RF_vol_200_d10": {
            "factory": lambda: RandomForestRegressor(n_estimators=200, max_depth=10,
                        min_samples_leaf=20, random_state=42, n_jobs=-1),
            "target": "target_vol_5d", "task": "regression", "family": "volatility"
        },
        "RF_vol_500_d15": {
            "factory": lambda: RandomForestRegressor(n_estimators=500, max_depth=15,
                        min_samples_leaf=10, random_state=42, n_jobs=-1),
            "target": "target_vol_5d", "task": "regression", "family": "volatility"
        },
        "GB_vol_300_lr05": {
            "factory": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=5,
                        learning_rate=0.05, min_samples_leaf=20, random_state=42),
            "target": "target_vol_5d", "task": "regression", "family": "volatility"
        },

        # --- Return regressors ---
        "RF_ret1d_200": {
            "factory": lambda: RandomForestRegressor(n_estimators=200, max_depth=10,
                        min_samples_leaf=20, random_state=42, n_jobs=-1),
            "target": "target_ret_1d", "task": "regression", "family": "return_1d"
        },
        "GB_ret1d_300": {
            "factory": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4,
                        learning_rate=0.03, min_samples_leaf=25, random_state=42),
            "target": "target_ret_1d", "task": "regression", "family": "return_1d"
        },
    }

    def __init__(self):
        self.rankings = {}
        self.history = []

    def run_tournament(self, wf_results):
        """Rank models within each family based on walk-forward results."""
        families = defaultdict(list)

        for name, config in self.CANDIDATES.items():
            family = config["family"]
            if name in wf_results:
                agg = wf_results[name].get("aggregate", {})
                families[family].append({"name": name, "metrics": agg})

        rankings = {}
        for family, entries in families.items():
            if not entries:
                continue

            # Sort: classifiers by mean_f1 desc, regressors by mean_r2 desc
            if entries[0]["metrics"].get("mean_f1") is not None:
                entries.sort(key=lambda x: x["metrics"].get("mean_f1", 0), reverse=True)
            else:
                entries.sort(key=lambda x: x["metrics"].get("mean_r2", -999), reverse=True)

            rankings[family] = entries
            winner = entries[0]
            log.info(f"  🏆 {family}: Winner = {winner['name']} "
                     f"(metrics: { {k: round(v, 4) for k, v in winner['metrics'].items() if isinstance(v, float)} })")

        self.rankings = rankings
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "rankings": {f: [e["name"] for e in entries] for f, entries in rankings.items()},
            "metrics": {f: {e["name"]: e["metrics"] for e in entries} for f, entries in rankings.items()},
        })

        # Save
        with open(TOURNAMENT_FILE, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        return rankings

    def get_champion(self, family):
        """Return the best model config name for a family."""
        if family in self.rankings and self.rankings[family]:
            return self.rankings[family][0]["name"]
        return None


# ========================== 5. DRIFT DETECTOR ========================== #
class DriftDetector:
    """Monitor model performance over time and flag degradation."""

    def __init__(self, window=10, threshold_drop=0.03):
        self.window = window                # Number of recent folds to compare
        self.threshold_drop = threshold_drop # Flag if metric drops more than this
        self.alerts = []

    def check(self, wf_results):
        """Compare recent fold performance vs historical performance."""
        self.alerts = []

        for name, data in wf_results.items():
            folds = data.get("fold_metrics", [])
            if len(folds) < self.window * 2:
                continue

            for metric in ["accuracy", "f1", "r2"]:
                vals = [f[metric] for f in folds if metric in f]
                if len(vals) < self.window * 2:
                    continue

                early = np.mean(vals[:self.window])
                recent = np.mean(vals[-self.window:])
                drop = early - recent

                if drop > self.threshold_drop:
                    alert = {
                        "model": name,
                        "metric": metric,
                        "early_avg": round(early, 4),
                        "recent_avg": round(recent, 4),
                        "drop": round(drop, 4),
                        "severity": "HIGH" if drop > self.threshold_drop * 2 else "MEDIUM",
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.alerts.append(alert)
                    log.warning(f"⚠️  DRIFT: {name} | {metric} dropped {drop:.4f} "
                               f"({early:.4f} → {recent:.4f})")

        # Save drift log
        existing = []
        if os.path.exists(DRIFT_LOG):
            with open(DRIFT_LOG) as f:
                existing = json.load(f)
        existing.extend(self.alerts)
        with open(DRIFT_LOG, "w") as f:
            json.dump(existing[-500:], f, indent=2)  # Keep last 500

        if not self.alerts:
            log.info("✓ No drift detected.")
        return self.alerts


# ========================== 6. FEATURE EVOLUTION ========================== #
class FeatureEvolver:
    """Test new feature combinations and keep improvements."""

    def __init__(self):
        self.tested_configs = []
        self.best_config = None
        self.best_score = -np.inf

    def evolve(self, df, base_engine, wf_engine, n_experiments=5, progress_cb=None):
        """Run experiments adding/removing features."""
        base_config = base_engine.config.copy()
        experimental_keys = list(FeatureEngine.EXPERIMENTAL_FEATURES.keys())

        # Baseline score
        log.info("Running baseline for feature evolution...")
        df_base, base_cols = base_engine.engineer(df)
        df_base = base_engine.add_targets(df_base)

        # Use a single fast model for feature testing
        test_configs = {
            "test_dir": {
                "factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=8,
                            min_samples_leaf=30, random_state=42, n_jobs=-1),
                "target": "target_dir_1d", "task": "classification", "family": "test"
            }
        }

        base_results = wf_engine.run(df_base, base_cols, test_configs)
        base_score = 0.0
        if "test_dir" in base_results:
            base_score = base_results["test_dir"]["aggregate"].get("mean_f1", 0)
        log.info(f"  Baseline F1: {base_score:.4f}")

        # Test adding experimental features
        for i in range(min(n_experiments, len(experimental_keys))):
            key = experimental_keys[i]
            if progress_cb:
                progress_cb(f"Testing feature: {key}", (i + 1) / n_experiments)

            test_config = base_config.copy()
            test_config[key] = FeatureEngine.EXPERIMENTAL_FEATURES[key]
            test_engine = FeatureEngine(test_config)

            df_test, test_cols = test_engine.engineer(df)
            df_test = test_engine.add_targets(df_test)

            results = wf_engine.run(df_test, test_cols, test_configs)
            score = 0.0
            if "test_dir" in results:
                score = results["test_dir"]["aggregate"].get("mean_f1", 0)

            improvement = score - base_score
            status = "✓ KEEP" if improvement > 0.001 else "✗ skip"
            log.info(f"  {key}: F1={score:.4f} ({'+' if improvement >= 0 else ''}{improvement:.4f}) → {status}")

            self.tested_configs.append({
                "feature": key,
                "score": score,
                "improvement": improvement,
                "kept": improvement > 0.001,
                "timestamp": datetime.now().isoformat(),
            })

            if improvement > 0.001:
                base_config[key] = FeatureEngine.EXPERIMENTAL_FEATURES[key]
                base_score = score

        self.best_config = base_config
        self.best_score = base_score
        log.info(f"Feature evolution complete. Best F1: {base_score:.4f}")
        return base_config


# ========================== 7. FULL EVOLUTION PIPELINE ========================== #
class EvolutionPipeline:
    """Orchestrate the full self-improvement cycle."""

    def __init__(self):
        self.updater = DataUpdater()
        self.feature_engine = FeatureEngine()
        self.wf_engine = WalkForwardEngine()
        self.tournament = ModelTournament()
        self.drift_detector = DriftDetector()
        self.feature_evolver = FeatureEvolver()
        self.run_history = []

    def run_full_cycle(self, excel_path=None, evolve_features=False, progress_cb=None):
        """Run one complete evolution cycle."""
        log.info("=" * 60)
        log.info("STARTING EVOLUTION CYCLE")
        log.info("=" * 60)
        t0 = time.time()

        # 1. Load/update data
        log.info("\n--- Step 1: Data Update ---")
        if not self.updater.load_existing(excel_path):
            log.error("No data available. Provide Excel path on first run.")
            return None
        self.updater.fetch_new_data()

        if progress_cb:
            progress_cb("Data loaded. Engineering features...", 0.1)

        # 2. Feature engineering
        log.info("\n--- Step 2: Feature Engineering ---")
        df, feature_cols = self.feature_engine.engineer(self.updater.df)
        df = self.feature_engine.add_targets(df)

        if progress_cb:
            progress_cb(f"Engineered {len(feature_cols)} features. Running walk-forward...", 0.2)

        # 3. Walk-forward evaluation
        log.info("\n--- Step 3: Walk-Forward Evaluation ---")
        wf_results = self.wf_engine.run(
            df, feature_cols, ModelTournament.CANDIDATES,
            progress_cb=lambda msg, pct: progress_cb(msg, 0.2 + pct * 0.4) if progress_cb else None
        )

        if progress_cb:
            progress_cb("Walk-forward complete. Running tournament...", 0.65)

        # 4. Tournament
        log.info("\n--- Step 4: Model Tournament ---")
        rankings = self.tournament.run_tournament(wf_results)

        # 5. Drift detection
        log.info("\n--- Step 5: Drift Detection ---")
        alerts = self.drift_detector.check(wf_results)

        # 6. Feature evolution (optional, slow)
        if evolve_features:
            log.info("\n--- Step 6: Feature Evolution ---")
            new_config = self.feature_evolver.evolve(
                self.updater.df, self.feature_engine, self.wf_engine,
                progress_cb=lambda msg, pct: progress_cb(msg, 0.75 + pct * 0.2) if progress_cb else None
            )
            self.feature_engine = FeatureEngine(new_config)

        elapsed = time.time() - t0
        log.info(f"\nEvolution cycle complete in {elapsed:.1f}s")

        # Save run summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "data_rows": len(self.updater.df),
            "features": len(feature_cols),
            "wf_folds": len(next(iter(wf_results.values()))["fold_metrics"]) if wf_results else 0,
            "champions": {f: self.tournament.get_champion(f) for f in rankings},
            "drift_alerts": len(alerts),
            "feature_evolution": evolve_features,
        }
        self.run_history.append(summary)

        with open(HISTORY_FILE, "w") as f:
            json.dump(self.run_history, f, indent=2)

        if progress_cb:
            progress_cb("Evolution cycle complete!", 1.0)

        return summary


# ========================== 8. SCHEDULED RUNNER ========================== #
def run_scheduled_update(excel_path=None):
    """Run from cron or scheduled task."""
    pipeline = EvolutionPipeline()
    summary = pipeline.run_full_cycle(
        excel_path=excel_path or "SP500_Analysis.xlsx",
        evolve_features=False,
        progress_cb=lambda msg, pct: log.info(f"  [{int(pct*100):3d}%] {msg}")
    )
    if summary:
        log.info(f"\nSummary: {json.dumps(summary, indent=2)}")
    return summary


# ========================== 9. INTERACTIVE CLI ========================== #
def interactive_menu():
    """Terminal-based interactive menu."""
    pipeline = EvolutionPipeline()
    excel_path = None

    # Find Excel file
    for candidate in ["SP500_Analysis.xlsx", os.path.expanduser("~/Downloads/SP500_Analysis.xlsx")]:
        if os.path.exists(candidate):
            excel_path = candidate
            break

    while True:
        print("\n" + "=" * 50)
        print("  S&P 500 Evolution System")
        print("=" * 50)
        print(f"  Data: {pipeline.updater.MASTER_CSV}")
        if pipeline.updater.df is not None:
            print(f"  Rows: {len(pipeline.updater.df)} | "
                  f"Latest: {pipeline.updater.df['Date'].iloc[-1].date()}")
        print()
        print("  1. Run full evolution cycle")
        print("  2. Run with feature evolution (slower)")
        print("  3. Update data only")
        print("  4. View tournament results")
        print("  5. View drift alerts")
        print("  6. View run history")
        print("  7. Quit")
        print()

        choice = input("  Select (1-7): ").strip()

        if choice == "1":
            pipeline.run_full_cycle(
                excel_path=excel_path,
                evolve_features=False,
                progress_cb=lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}")
            )

        elif choice == "2":
            pipeline.run_full_cycle(
                excel_path=excel_path,
                evolve_features=True,
                progress_cb=lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}")
            )

        elif choice == "3":
            pipeline.updater.load_existing(excel_path)
            pipeline.updater.fetch_new_data()

        elif choice == "4":
            if os.path.exists(TOURNAMENT_FILE):
                with open(TOURNAMENT_FILE) as f:
                    data = json.load(f)
                if data:
                    latest = data[-1]
                    print(f"\n  Latest tournament ({latest.get('timestamp', 'N/A')}):")
                    for family, names in latest.get("rankings", {}).items():
                        print(f"    {family}:")
                        for i, name in enumerate(names):
                            prefix = "    🏆" if i == 0 else "      "
                            metrics = latest.get("metrics", {}).get(family, {}).get(name, {})
                            m_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                            if isinstance(v, float))
                            print(f"{prefix} {name} ({m_str})")
            else:
                print("  No tournament results yet. Run an evolution cycle first.")

        elif choice == "5":
            if os.path.exists(DRIFT_LOG):
                with open(DRIFT_LOG) as f:
                    alerts = json.load(f)
                if alerts:
                    print(f"\n  {len(alerts)} drift alerts:")
                    for a in alerts[-10:]:
                        print(f"    [{a['severity']}] {a['model']}: {a['metric']} "
                              f"dropped {a['drop']:.4f} ({a['early_avg']:.4f} → {a['recent_avg']:.4f})")
                else:
                    print("  No drift alerts.")
            else:
                print("  No drift log yet.")

        elif choice == "6":
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE) as f:
                    history = json.load(f)
                for run in history[-5:]:
                    print(f"\n  {run['timestamp']}:")
                    print(f"    Data: {run['data_rows']} rows | Features: {run['features']}")
                    print(f"    Folds: {run['wf_folds']} | Drift alerts: {run['drift_alerts']}")
                    print(f"    Champions: {run.get('champions', {})}")
                    print(f"    Time: {run['elapsed_seconds']}s")
            else:
                print("  No run history yet.")

        elif choice == "7":
            break


# ========================== 10. CRON SETUP HELPER ========================== #
CRON_SETUP = """
# ──────────────────────────────────────────────────
#  AUTOMATED DAILY UPDATES — Setup Guide
# ──────────────────────────────────────────────────
#
#  OPTION A: Cron (Linux/Mac)
#  ─────────────────────────
#  Run `crontab -e` and add:
#
#    # Update S&P 500 model every weekday at 6:30 PM ET (after market close)
#    30 18 * * 1-5 cd /path/to/your/project && python sp500_evolution.py --update >> sp500_logs/cron.log 2>&1
#
#  OPTION B: Task Scheduler (Windows)
#  ───────────────────────────────────
#  1. Open Task Scheduler
#  2. Create Basic Task → "SP500 ML Update"
#  3. Trigger: Daily, 6:30 PM, weekdays only
#  4. Action: Start program
#     Program: python
#     Arguments: sp500_evolution.py --update
#     Start in: C:\\path\\to\\your\\project
#
#  OPTION C: Python scheduler (always-on)
#  ───────────────────────────────────────
#    pip install schedule
#    # See schedule_loop() below
#
#  OPTION D: GitHub Actions (free, cloud-based)
#  ─────────────────────────────────────────────
#  Create .github/workflows/update.yml:
#
#    name: Daily SP500 Update
#    on:
#      schedule:
#        - cron: '30 22 * * 1-5'  # 10:30 PM UTC = 6:30 PM ET
#    jobs:
#      update:
#        runs-on: ubuntu-latest
#        steps:
#          - uses: actions/checkout@v4
#          - uses: actions/setup-python@v5
#            with: { python-version: '3.11' }
#          - run: pip install pandas scikit-learn numpy yfinance openpyxl
#          - run: python sp500_evolution.py --update
#          - uses: actions/upload-artifact@v4
#            with: { name: models, path: sp500_models/ }
"""


def schedule_loop():
    """Run update loop with Python schedule library."""
    try:
        import schedule
    except ImportError:
        print("Install schedule: pip install schedule")
        return

    def job():
        log.info("Scheduled update triggered.")
        run_scheduled_update()

    schedule.every().monday.at("18:30").do(job)
    schedule.every().tuesday.at("18:30").do(job)
    schedule.every().wednesday.at("18:30").do(job)
    schedule.every().thursday.at("18:30").do(job)
    schedule.every().friday.at("18:30").do(job)

    log.info("Scheduler started. Running weekdays at 18:30.")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ========================== ENTRY POINT ========================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 ML Evolution System")
    parser.add_argument("--update", action="store_true", help="Run update cycle (cron-friendly)")
    parser.add_argument("--evolve", action="store_true", help="Run with feature evolution")
    parser.add_argument("--schedule", action="store_true", help="Start continuous scheduler")
    parser.add_argument("--setup", action="store_true", help="Show automation setup guide")
    parser.add_argument("--data", type=str, default=None, help="Path to SP500_Analysis.xlsx")
    args = parser.parse_args()

    if args.setup:
        print(CRON_SETUP)
    elif args.schedule:
        schedule_loop()
    elif args.update or args.evolve:
        pipeline = EvolutionPipeline()
        pipeline.run_full_cycle(
            excel_path=args.data or "SP500_Analysis.xlsx",
            evolve_features=args.evolve,
            progress_cb=lambda msg, pct: log.info(f"  [{int(pct*100):3d}%] {msg}")
        )
    else:
        interactive_menu()
