#!/usr/bin/env python3
"""
sp500_backtester.py — Advanced Backtesting & Strategy Engine

  1. ENSEMBLE STRATEGIES   Combine RF + GB + LSTM votes with confidence weighting
  2. POSITION SIZING       Kelly criterion, volatility-targeting, risk parity
  3. RISK MANAGEMENT       Stop-losses, max drawdown circuit breakers, exposure limits
  4. STRATEGY LIBRARY      Long/cash, long/short, mean-reversion, momentum combos
  5. HTML REPORTS          Auto-generated performance reports for email or browser
  6. MONTE CARLO           Simulate strategy variance with bootstrapped returns

Usage:
    python sp500_backtester.py                # Interactive menu
    python sp500_backtester.py --full         # Run all strategies, generate report
    python sp500_backtester.py --report-only  # Generate HTML from existing results
    python sp500_backtester.py --monte-carlo  # Run Monte Carlo simulation

Requires: pandas, scikit-learn, numpy, matplotlib, openpyxl
Optional: pygame, torch
"""

import os
import sys
import json
import time
import pickle
import math
import warnings
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score

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
    import matplotlib.dates as mdates
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    from sp500_evolution import DataUpdater, FeatureEngine, LOG_DIR, MODEL_DIR
except ImportError:
    LOG_DIR = "sp500_logs"
    MODEL_DIR = "sp500_models"

for d in [LOG_DIR, MODEL_DIR, os.path.join(LOG_DIR, "reports")]:
    os.makedirs(d, exist_ok=True)

REPORT_DIR = os.path.join(LOG_DIR, "reports")
BACKTEST_RESULTS_FILE = os.path.join(MODEL_DIR, "backtest_results.pkl")


# ========================== LSTM MODEL ========================== #
if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers,
                                batch_first=True, dropout=dropout)
            self.fc_dir = nn.Linear(hidden, 1)
            self.fc_ret = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            h = out[:, -1, :]
            return torch.sigmoid(self.fc_dir(h)), self.fc_ret(h)


# ========================== STRATEGY ENGINE ========================== #
class StrategyEngine:
    """Library of trading strategies that consume model predictions."""

    @staticmethod
    def long_cash(predictions, actual_returns, **kwargs):
        """Long when predicted up, cash when predicted down."""
        equity = [1.0]
        positions = []
        for i in range(len(predictions)):
            pos = 1.0 if predictions[i] > 0.5 else 0.0
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def long_short(predictions, actual_returns, **kwargs):
        """Long when predicted up, short when predicted down."""
        equity = [1.0]
        positions = []
        for i in range(len(predictions)):
            pos = 1.0 if predictions[i] > 0.5 else -1.0
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def confidence_weighted(predictions, actual_returns, **kwargs):
        """Scale position size by model confidence."""
        equity = [1.0]
        positions = []
        for i in range(len(predictions)):
            confidence = abs(predictions[i] - 0.5) * 2  # 0 to 1
            direction = 1.0 if predictions[i] > 0.5 else -0.5
            pos = direction * confidence
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def kelly_criterion(predictions, actual_returns, win_rate=0.52, avg_win=0.008,
                        avg_loss=0.008, max_leverage=1.5, **kwargs):
        """Kelly criterion position sizing based on historical edge."""
        # f* = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = 1-p
        b = avg_win / max(avg_loss, 1e-10)
        p = win_rate
        q = 1 - p
        kelly_f = max(0, min((b * p - q) / b, max_leverage))

        equity = [1.0]
        positions = []
        for i in range(len(predictions)):
            if predictions[i] > 0.5:
                pos = kelly_f
            else:
                pos = 0.0
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def vol_targeting(predictions, actual_returns, vol_estimates,
                      target_vol=0.15, max_leverage=2.0, **kwargs):
        """Target a specific portfolio volatility using vol predictions."""
        equity = [1.0]
        positions = []
        for i in range(len(predictions)):
            if predictions[i] > 0.5 and i < len(vol_estimates):
                # Scale position to hit target vol
                current_vol = max(vol_estimates[i], 0.05)
                pos = min(target_vol / current_vol, max_leverage)
            else:
                pos = 0.0
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def drawdown_limiter(predictions, actual_returns,
                          max_dd=-0.10, cooldown=10, **kwargs):
        """Cut exposure after hitting drawdown threshold."""
        equity = [1.0]
        positions = []
        peak = 1.0
        cooldown_remaining = 0

        for i in range(len(predictions)):
            peak = max(peak, equity[-1])
            dd = (equity[-1] - peak) / peak

            if dd < max_dd:
                cooldown_remaining = cooldown

            if cooldown_remaining > 0:
                pos = 0.0
                cooldown_remaining -= 1
            else:
                pos = 1.0 if predictions[i] > 0.5 else 0.0

            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def ensemble_vote(model_predictions, actual_returns, weights=None, threshold=0.5,
                      **kwargs):
        """Combine multiple model predictions with weighted voting."""
        n_models = len(model_predictions)
        if weights is None:
            weights = [1.0 / n_models] * n_models

        n = len(actual_returns)
        equity = [1.0]
        positions = []

        for i in range(n):
            # Weighted average probability
            avg_prob = sum(model_predictions[m][i] * weights[m]
                          for m in range(n_models)
                          if i < len(model_predictions[m]))
            pos = 1.0 if avg_prob > threshold else 0.0
            positions.append(pos)
            equity.append(equity[-1] * (1 + actual_returns[i] * pos))

        return equity, positions

    @staticmethod
    def mean_reversion(ret_predictions, actual_returns, threshold=0.005, **kwargs):
        """Fade extreme predicted moves — contrarian on big predictions."""
        equity = [1.0]
        positions = []
        for i in range(len(ret_predictions)):
            pred = ret_predictions[i]
            if pred > threshold:
                pos = -0.5     # Predicted big up → fade (short light)
            elif pred < -threshold:
                pos = 1.0      # Predicted big down → buy (contrarian)
            else:
                pos = 0.0      # Small prediction → stay out
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions

    @staticmethod
    def momentum_filter(dir_predictions, actual_returns, sma_ratio, **kwargs):
        """Only take long signals when above 200-day SMA (trend filter)."""
        equity = [1.0]
        positions = []
        for i in range(len(dir_predictions)):
            above_sma = sma_ratio[i] > 1.0 if i < len(sma_ratio) else True
            if dir_predictions[i] > 0.5 and above_sma:
                pos = 1.0
            elif not above_sma:
                pos = 0.0  # Below SMA = cash regardless
            else:
                pos = 0.0
            positions.append(pos)
            if i < len(actual_returns):
                equity.append(equity[-1] * (1 + actual_returns[i] * pos))
        return equity, positions


# ========================== METRICS ========================== #
class PerformanceMetrics:
    """Compute comprehensive performance statistics."""

    @staticmethod
    def compute(equity, actual_returns, positions=None, risk_free=0.02, periods=252):
        """Full performance breakdown."""
        equity = np.array(equity, dtype=float)
        returns = np.diff(np.log(np.clip(equity, 1e-10, None)))

        if len(returns) == 0:
            return {}

        total_return = (equity[-1] / equity[0] - 1) * 100
        n_years = len(returns) / periods
        cagr = ((equity[-1] / equity[0]) ** (1 / max(n_years, 0.01)) - 1) * 100

        # Sharpe
        ann_ret = np.mean(returns) * periods
        ann_vol = np.std(returns) * np.sqrt(periods)
        sharpe = (ann_ret - risk_free) / max(ann_vol, 1e-10)

        # Sortino
        downside = returns[returns < 0]
        down_vol = np.std(downside) * np.sqrt(periods) if len(downside) > 0 else 1e-10
        sortino = (ann_ret - risk_free) / max(down_vol, 1e-10)

        # Drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = np.min(dd) * 100

        # Drawdown duration
        in_dd = dd < 0
        dd_periods = []
        current = 0
        for d in in_dd:
            if d:
                current += 1
            else:
                if current > 0:
                    dd_periods.append(current)
                current = 0
        if current > 0:
            dd_periods.append(current)
        max_dd_duration = max(dd_periods) if dd_periods else 0

        # Calmar
        calmar = cagr / max(abs(max_dd), 0.01)

        # Win rate
        if positions is not None and len(actual_returns) > 0:
            active = [(p, r) for p, r in zip(positions, actual_returns) if abs(p) > 0.01]
            if active:
                wins = sum(1 for p, r in active if p * r > 0)
                win_rate = wins / len(active)
                avg_win = np.mean([p * r for p, r in active if p * r > 0]) if wins > 0 else 0
                avg_loss = np.mean([abs(p * r) for p, r in active if p * r <= 0]) if len(active) - wins > 0 else 0
                profit_factor = (avg_win * wins) / max(avg_loss * (len(active) - wins), 1e-10)
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
            exposure = sum(1 for p in positions if abs(p) > 0.01) / max(len(positions), 1)
        else:
            win_rate = avg_win = avg_loss = profit_factor = exposure = 0

        return {
            "total_return": round(total_return, 2),
            "cagr": round(cagr, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "calmar": round(calmar, 3),
            "max_drawdown": round(max_dd, 2),
            "max_dd_duration_days": max_dd_duration,
            "ann_volatility": round(ann_vol * 100, 2),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win * 100, 4),
            "avg_loss": round(avg_loss * 100, 4),
            "profit_factor": round(profit_factor, 3),
            "exposure": round(exposure, 4),
            "n_periods": len(returns),
            "n_years": round(n_years, 1),
        }


# ========================== MONTE CARLO ========================== #
class MonteCarloSimulator:
    """Bootstrap strategy returns to estimate confidence intervals."""

    @staticmethod
    def simulate(daily_returns, n_sims=1000, n_days=252, seed=42):
        """Run Monte Carlo simulation on daily strategy returns."""
        rng = np.random.RandomState(seed)
        daily_returns = np.array(daily_returns)

        # Bootstrap: sample with replacement
        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = 1.0

        for sim in range(n_sims):
            sampled = rng.choice(daily_returns, size=n_days, replace=True)
            paths[sim, 1:] = np.cumprod(1 + sampled)

        # Statistics
        final_values = paths[:, -1]
        percentiles = {
            "p5": np.percentile(final_values, 5),
            "p25": np.percentile(final_values, 25),
            "p50": np.percentile(final_values, 50),
            "p75": np.percentile(final_values, 75),
            "p95": np.percentile(final_values, 95),
        }

        # Probability of profit
        prob_profit = np.mean(final_values > 1.0)

        # Max drawdown distribution
        max_dds = []
        for sim in range(n_sims):
            peak = np.maximum.accumulate(paths[sim])
            dd = (paths[sim] - peak) / peak
            max_dds.append(np.min(dd))

        return {
            "paths": paths,
            "final_values": final_values,
            "percentiles": {k: round(v, 4) for k, v in percentiles.items()},
            "prob_profit": round(prob_profit, 4),
            "expected_return": round(np.mean(final_values) - 1, 4),
            "median_max_dd": round(np.median(max_dds), 4),
            "p5_max_dd": round(np.percentile(max_dds, 5), 4),
        }


# ========================== FULL BACKTEST RUNNER ========================== #
class BacktestRunner:
    """Run comprehensive backtest across all strategies and models."""

    def __init__(self):
        self.results = {}

    def run(self, excel_path, progress_cb=None):
        """Full backtest pipeline."""
        if progress_cb:
            progress_cb("Loading data...", 0.0)

        # Load and prepare data
        df = pd.read_excel(excel_path, sheet_name="Daily Data")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        engine = FeatureEngine()
        df, feature_cols = engine.engineer(df)
        df = engine.add_targets(df)

        target_cols = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']
        valid = df.dropna(subset=feature_cols + target_cols).copy()

        split = int(len(valid) * 0.8)
        train = valid.iloc[:split]
        test = valid.iloc[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols].values)
        X_test = scaler.transform(test[feature_cols].values)
        actual_returns = test['target_ret_1d'].values
        test_dates = test['Date'].values
        test_close = test['Close'].values

        # Extra features for strategies
        sma_200_ratio = test['close_sma_200_ratio'].values if 'close_sma_200_ratio' in test.columns else np.ones(len(test))
        vol_20d = test['vol_20d'].values if 'vol_20d' in test.columns else np.full(len(test), 0.15)

        if progress_cb:
            progress_cb("Training models...", 0.1)

        # ─── Train models ───
        models = {}

        # RF direction
        rf_dir = RandomForestClassifier(n_estimators=300, max_depth=10,
                                         min_samples_leaf=20, random_state=42, n_jobs=-1)
        rf_dir.fit(X_train, train['target_dir_1d'].values)
        rf_dir_prob = rf_dir.predict_proba(X_test)[:, 1]
        models["rf_dir"] = rf_dir_prob

        if progress_cb:
            progress_cb("Training GB...", 0.2)

        # GB direction
        gb_dir = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                             learning_rate=0.03, min_samples_leaf=20, random_state=42)
        gb_dir.fit(X_train, train['target_dir_1d'].values)
        gb_dir_prob = gb_dir.predict_proba(X_test)[:, 1]
        models["gb_dir"] = gb_dir_prob

        # RF return predictions
        rf_ret = RandomForestRegressor(n_estimators=200, max_depth=10,
                                        min_samples_leaf=20, random_state=42, n_jobs=-1)
        rf_ret.fit(X_train, train['target_ret_1d'].values)
        rf_ret_pred = rf_ret.predict(X_test)
        models["rf_ret"] = rf_ret_pred

        # RF volatility
        rf_vol = RandomForestRegressor(n_estimators=200, max_depth=10,
                                        min_samples_leaf=20, random_state=42, n_jobs=-1)
        rf_vol.fit(X_train, train['target_vol_5d'].values)
        vol_pred = rf_vol.predict(X_test)
        models["rf_vol"] = vol_pred

        if progress_cb:
            progress_cb("Training LSTM...", 0.3)

        # LSTM
        lstm_prob = None
        if TORCH_AVAILABLE:
            try:
                lstm_prob = self._train_lstm(X_train, train, X_test, feature_cols, progress_cb)
                models["lstm_dir"] = lstm_prob
            except Exception as e:
                print(f"LSTM training failed: {e}")

        # ─── Model accuracy metrics ───
        if progress_cb:
            progress_cb("Computing model metrics...", 0.5)

        actual_dir = test['target_dir_1d'].values
        model_metrics = {}
        for name, probs in [("Random Forest", rf_dir_prob), ("Gradient Boosting", gb_dir_prob)]:
            preds = (probs > 0.5).astype(int)
            model_metrics[name] = {
                "accuracy": round(accuracy_score(actual_dir, preds), 4),
                "f1": round(f1_score(actual_dir, preds), 4),
            }
        if lstm_prob is not None:
            preds = (lstm_prob > 0.5).astype(int)
            model_metrics["LSTM"] = {
                "accuracy": round(accuracy_score(actual_dir[:len(preds)], preds), 4),
                "f1": round(f1_score(actual_dir[:len(preds)], preds), 4),
            }

        # RF vol r2
        model_metrics["RF Volatility"] = {
            "r2": round(r2_score(test['target_vol_5d'].values, vol_pred), 4)
        }

        # Feature importance
        feat_imp = dict(zip(feature_cols, rf_dir.feature_importances_))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15])

        # ─── Run strategies ───
        if progress_cb:
            progress_cb("Running strategies...", 0.55)

        strategies = {}
        se = StrategyEngine

        # Buy and hold baseline
        bh_eq = [1.0]
        for r in actual_returns:
            bh_eq.append(bh_eq[-1] * (1 + r))
        bh_metrics = PerformanceMetrics.compute(bh_eq, actual_returns)

        strategy_configs = [
            ("RF Long/Cash", lambda: se.long_cash(rf_dir_prob, actual_returns)),
            ("GB Long/Cash", lambda: se.long_cash(gb_dir_prob, actual_returns)),
            ("RF Long/Short", lambda: se.long_short(rf_dir_prob, actual_returns)),
            ("RF Confidence-Weighted", lambda: se.confidence_weighted(rf_dir_prob, actual_returns)),
            ("RF Kelly Criterion", lambda: se.kelly_criterion(
                rf_dir_prob, actual_returns,
                win_rate=model_metrics["Random Forest"]["accuracy"],
                avg_win=np.mean(actual_returns[actual_returns > 0]) if any(actual_returns > 0) else 0.008,
                avg_loss=np.mean(np.abs(actual_returns[actual_returns < 0])) if any(actual_returns < 0) else 0.008)),
            ("RF Vol-Targeting (15%)", lambda: se.vol_targeting(
                rf_dir_prob, actual_returns, vol_pred, target_vol=0.15)),
            ("RF Drawdown Limiter", lambda: se.drawdown_limiter(
                rf_dir_prob, actual_returns, max_dd=-0.08, cooldown=15)),
            ("RF + Momentum Filter", lambda: se.momentum_filter(
                rf_dir_prob, actual_returns, sma_200_ratio)),
            ("Mean Reversion", lambda: se.mean_reversion(rf_ret_pred, actual_returns)),
        ]

        # Ensemble strategies
        if lstm_prob is not None:
            min_len = min(len(rf_dir_prob), len(gb_dir_prob), len(lstm_prob))
            strategy_configs.append(
                ("Ensemble (RF+GB+LSTM)", lambda: se.ensemble_vote(
                    [rf_dir_prob[:min_len], gb_dir_prob[:min_len], lstm_prob[:min_len]],
                    actual_returns[:min_len],
                    weights=[0.45, 0.25, 0.30]))
            )
            strategy_configs.append(
                ("Ensemble Confidence", lambda: se.confidence_weighted(
                    np.array([rf_dir_prob[:min_len][i] * 0.45 + gb_dir_prob[:min_len][i] * 0.25 + lstm_prob[:min_len][i] * 0.30
                              for i in range(min_len)]),
                    actual_returns[:min_len]))
            )
        else:
            min_len = min(len(rf_dir_prob), len(gb_dir_prob))
            strategy_configs.append(
                ("Ensemble (RF+GB)", lambda: se.ensemble_vote(
                    [rf_dir_prob[:min_len], gb_dir_prob[:min_len]],
                    actual_returns[:min_len],
                    weights=[0.6, 0.4]))
            )

        total_strats = len(strategy_configs)
        for si, (name, strat_fn) in enumerate(strategy_configs):
            if progress_cb:
                progress_cb(f"Backtesting: {name}", 0.55 + 0.3 * (si / total_strats))

            eq, pos = strat_fn()
            ar = actual_returns[:len(pos)]
            metrics = PerformanceMetrics.compute(eq, ar, pos)
            strategies[name] = {"equity": eq, "positions": pos, "metrics": metrics}

        # ─── Monte Carlo on best strategy ───
        if progress_cb:
            progress_cb("Running Monte Carlo...", 0.88)

        # Find best strategy by Sharpe
        best_name = max(strategies, key=lambda s: strategies[s]["metrics"].get("sharpe", -999))
        best_eq = strategies[best_name]["equity"]
        best_daily = np.diff(np.log(np.clip(best_eq, 1e-10, None)))

        mc = MonteCarloSimulator.simulate(best_daily, n_sims=1000, n_days=252)

        if progress_cb:
            progress_cb("Compiling results...", 0.95)

        # ─── Compile results ───
        self.results = {
            "data_info": {
                "total_rows": len(df),
                "test_size": len(test),
                "test_range": f"{test['Date'].iloc[0].strftime('%Y-%m-%d')} to {test['Date'].iloc[-1].strftime('%Y-%m-%d')}",
                "train_range": f"{train['Date'].iloc[0].strftime('%Y-%m-%d')} to {train['Date'].iloc[-1].strftime('%Y-%m-%d')}",
                "features": len(feature_cols),
            },
            "model_metrics": model_metrics,
            "feature_importance": feat_imp,
            "buy_hold": {"equity": bh_eq, "metrics": bh_metrics},
            "strategies": {name: {"metrics": s["metrics"], "equity_len": len(s["equity"])}
                          for name, s in strategies.items()},
            "strategy_equities": {name: s["equity"] for name, s in strategies.items()},
            "best_strategy": best_name,
            "monte_carlo": mc,
            "test_dates": [str(d)[:10] for d in test_dates],
            "test_close": test_close.tolist(),
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        with open(BACKTEST_RESULTS_FILE, "wb") as f:
            pickle.dump(self.results, f)

        if progress_cb:
            progress_cb("Backtest complete!", 1.0)

        return self.results

    def _train_lstm(self, X_train, train_df, X_test, feature_cols, progress_cb=None):
        """Train LSTM and return test probabilities."""
        SEQ_LEN = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Make sequences
        def make_seqs(X, y, sl):
            seqs, targets = [], []
            for i in range(sl, len(X)):
                seqs.append(X[i-sl:i])
                targets.append(y[i])
            return np.array(seqs), np.array(targets)

        y_dir = train_df['target_dir_1d'].values
        X_seq, y_seq = make_seqs(X_train, y_dir, SEQ_LEN)

        model = LSTMNet(X_train.shape[1], hidden=64, layers=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_t = torch.FloatTensor(X_seq).to(device)
        y_t = torch.FloatTensor(y_seq).unsqueeze(1).to(device)

        # Quick training — 30 epochs
        model.train()
        batch = 256
        for epoch in range(30):
            idx = torch.randperm(len(X_seq))
            for start in range(0, len(X_seq), batch):
                bi = idx[start:start+batch]
                dir_pred, _ = model(X_t[bi])
                loss = nn.BCELoss()(dir_pred, y_t[bi])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Predict on test
        model.eval()
        test_seqs = []
        for i in range(SEQ_LEN, len(X_test)):
            test_seqs.append(X_test[i-SEQ_LEN:i])
        if not test_seqs:
            return None

        test_seqs = np.array(test_seqs)
        with torch.no_grad():
            X_te = torch.FloatTensor(test_seqs).to(device)
            dir_prob, _ = model(X_te)
            probs = dir_prob.cpu().numpy().flatten()

        # Pad front with 0.5 for alignment
        padded = np.full(len(X_test), 0.5)
        padded[SEQ_LEN:SEQ_LEN+len(probs)] = probs
        return padded

    def load(self):
        """Load existing results."""
        if os.path.exists(BACKTEST_RESULTS_FILE):
            with open(BACKTEST_RESULTS_FILE, "rb") as f:
                self.results = pickle.load(f)
            return True
        return False


# ========================== HTML REPORT GENERATOR ========================== #
class HTMLReportGenerator:
    """Generate publication-quality HTML performance reports."""

    def __init__(self, results):
        self.r = results

    def generate(self, output_path=None):
        """Generate full HTML report."""
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(REPORT_DIR, f"backtest_report_{ts}.html")

        # Generate charts as base64
        charts = {}
        if MPL_AVAILABLE:
            charts["equity"] = self._chart_equity()
            charts["monte_carlo"] = self._chart_monte_carlo()
            charts["features"] = self._chart_features()
            charts["drawdown"] = self._chart_drawdown()

        html = self._build_html(charts)

        with open(output_path, "w") as f:
            f.write(html)

        print(f"Report saved: {output_path}")
        return output_path

    def _chart_to_base64(self, fig):
        import base64
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e', dpi=120)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{b64}"

    def _chart_equity(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        # Plot top 5 strategies + buy & hold
        strats = self.r.get("strategies", {})
        sorted_strats = sorted(strats.items(),
                               key=lambda x: x[1]["metrics"].get("sharpe", -999), reverse=True)[:5]

        colors = ['#10b981', '#f59e0b', '#3b82f6', '#a78bfa', '#ec4899']
        for i, (name, s) in enumerate(sorted_strats):
            eq = self.r.get("strategy_equities", {}).get(name, [])
            if eq:
                ax.plot(eq, color=colors[i % len(colors)], linewidth=1.5,
                       label=f"{name} (Sharpe={s['metrics'].get('sharpe', 0):.2f})", alpha=0.85)

        bh = self.r.get("buy_hold", {}).get("equity", [])
        if bh:
            ax.plot(bh, color='#6b7280', linewidth=1, linestyle='--',
                   label=f"Buy & Hold (Sharpe={self.r['buy_hold']['metrics'].get('sharpe', 0):.2f})", alpha=0.6)

        ax.set_title("Equity Curves — Top Strategies vs Buy & Hold", color='#e0e0e0', fontsize=13)
        ax.set_ylabel("Growth of $1", color='#999')
        ax.legend(fontsize=8, facecolor='#262640', edgecolor='#444', labelcolor='#ccc', loc='upper left')
        ax.tick_params(colors='#888')
        ax.grid(True, linestyle='--', alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#444')

        return self._chart_to_base64(fig)

    def _chart_monte_carlo(self):
        mc = self.r.get("monte_carlo", {})
        paths = mc.get("paths")
        if paths is None:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')

        # Path spaghetti
        ax1.set_facecolor('#1a1a2e')
        n_show = min(100, len(paths))
        for i in range(n_show):
            color = '#10b981' if paths[i, -1] > 1.0 else '#ef4444'
            ax1.plot(paths[i], color=color, alpha=0.05, linewidth=0.5)

        # Percentile bands
        p5 = np.percentile(paths, 5, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        ax1.fill_between(range(len(p5)), p5, p95, color='#3b82f6', alpha=0.15)
        ax1.plot(p50, color='#f59e0b', linewidth=2, label='Median')
        ax1.axhline(y=1.0, color='#6b7280', linestyle='--', alpha=0.5)
        ax1.set_title("Monte Carlo: 1-Year Paths", color='#e0e0e0', fontsize=11)
        ax1.legend(fontsize=8, facecolor='#262640', edgecolor='#444', labelcolor='#ccc')
        ax1.tick_params(colors='#888')
        for spine in ax1.spines.values():
            spine.set_color('#444')

        # Distribution histogram
        ax2.set_facecolor('#1a1a2e')
        finals = mc.get("final_values", paths[:, -1])
        ax2.hist(finals, bins=50, color='#3b82f6', alpha=0.6, edgecolor='#1a1a2e')
        ax2.axvline(x=1.0, color='#ef4444', linestyle='--', linewidth=2, label='Break-even')
        ax2.axvline(x=np.median(finals), color='#f59e0b', linestyle='-', linewidth=2, label=f'Median: {np.median(finals):.2f}x')
        ax2.set_title("1-Year Return Distribution", color='#e0e0e0', fontsize=11)
        ax2.set_xlabel("Portfolio Value ($1 start)", color='#999')
        ax2.legend(fontsize=8, facecolor='#262640', edgecolor='#444', labelcolor='#ccc')
        ax2.tick_params(colors='#888')
        for spine in ax2.spines.values():
            spine.set_color('#444')

        fig.tight_layout()
        return self._chart_to_base64(fig)

    def _chart_features(self):
        feat = self.r.get("feature_importance", {})
        if not feat:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        names = list(reversed(list(feat.keys())))
        vals = list(reversed(list(feat.values())))
        colors = ['#10b981' if v > 0.05 else '#3b82f6' if v > 0.03 else '#6366f1' for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_title("Top 15 Feature Importance (RF Direction)", color='#e0e0e0', fontsize=11)
        ax.tick_params(colors='#888', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#444')

        fig.tight_layout()
        return self._chart_to_base64(fig)

    def _chart_drawdown(self):
        strats = self.r.get("strategies", {})
        best = self.r.get("best_strategy", "")
        eq = self.r.get("strategy_equities", {}).get(best, [])
        if not eq:
            return None

        eq = np.array(eq)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        ax.fill_between(range(len(dd)), dd, 0, color='#ef4444', alpha=0.3)
        ax.plot(dd, color='#ef4444', linewidth=0.8)
        ax.set_title(f"Drawdown — {best}", color='#e0e0e0', fontsize=11)
        ax.set_ylabel("Drawdown %", color='#999')
        ax.tick_params(colors='#888')
        ax.grid(True, linestyle='--', alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#444')

        fig.tight_layout()
        return self._chart_to_base64(fig)

    def _build_html(self, charts):
        strats = self.r.get("strategies", {})
        bh = self.r.get("buy_hold", {}).get("metrics", {})
        mc = self.r.get("monte_carlo", {})
        mm = self.r.get("model_metrics", {})
        best = self.r.get("best_strategy", "N/A")
        info = self.r.get("data_info", {})

        # Pre-compute values to avoid f-string escaping issues
        best_metrics = strats.get(best, {}).get("metrics", {})
        best_sharpe = best_metrics.get("sharpe", 0)
        best_return = best_metrics.get("total_return", 0)
        best_dd = best_metrics.get("max_drawdown", 0)
        bh_sharpe = bh.get("sharpe", 0)
        bh_return = bh.get("total_return", 0)
        bh_dd = bh.get("max_drawdown", 0)
        bh_cagr = bh.get("cagr", 0)
        bh_sortino = bh.get("sortino", 0)
        bh_vol = bh.get("ann_volatility", 0)
        mc_prob = mc.get("prob_profit", 0)
        mc_expected = mc.get("expected_return", 0) * 100
        mc_p5 = mc.get("percentiles", {}).get("p5", 1)
        mc_p50 = mc.get("percentiles", {}).get("p50", 1)
        mc_p95 = mc.get("percentiles", {}).get("p95", 1)
        mc_mdd = mc.get("median_max_dd", 0) * 100
        mc_p5_mdd = mc.get("p5_max_dd", 0) * 100
        test_range = info.get("test_range", "N/A")
        train_range = info.get("train_range", "N/A")
        test_size = info.get("test_size", "?")
        n_features = info.get("features", "?")
        prob_class = "positive" if mc_prob > 0.5 else "negative"

        # Sort strategies by Sharpe
        sorted_strats = sorted(strats.items(),
                               key=lambda x: x[1]["metrics"].get("sharpe", -999), reverse=True)

        # Strategy table rows
        strat_rows = ""
        for name, s in sorted_strats:
            m = s["metrics"]
            is_best = name == best
            row_class = "best-row" if is_best else ""
            badge = ' <span class="badge">★ BEST</span>' if is_best else ""
            sharpe_class = "positive" if m.get("sharpe", 0) > bh.get("sharpe", 0) else "negative"
            ret_class = "positive" if m.get("total_return", 0) > 0 else "negative"

            strat_rows += f"""
            <tr class="{row_class}">
                <td class="strategy-name">{name}{badge}</td>
                <td class="{ret_class}">{m.get('total_return', 0):+.1f}%</td>
                <td>{m.get('cagr', 0):.1f}%</td>
                <td class="{sharpe_class}">{m.get('sharpe', 0):.3f}</td>
                <td>{m.get('sortino', 0):.3f}</td>
                <td class="negative">{m.get('max_drawdown', 0):.1f}%</td>
                <td>{m.get('ann_volatility', 0):.1f}%</td>
                <td>{m.get('win_rate', 0):.1%}</td>
                <td>{m.get('profit_factor', 0):.2f}</td>
                <td>{m.get('exposure', 0):.0%}</td>
            </tr>"""

        # Model metrics rows
        model_rows = ""
        for name, m in mm.items():
            metrics_str = " | ".join(f"{k}: {v}" for k, v in m.items())
            model_rows += f"<tr><td>{name}</td><td>{metrics_str}</td></tr>"

        # Chart images
        def img_tag(key, alt=""):
            return f'<img src="{charts[key]}" alt="{alt}" class="chart">' if charts.get(key) else ""

        # Build HTML using pre-computed variables (avoids f-string dict issues)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

        css = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 24px; max-width: 1200px; margin: 0 auto; line-height: 1.5; }
    h1 { color: #10b981; font-size: 28px; margin-bottom: 4px; }
    h2 { color: #60a5fa; font-size: 20px; margin: 32px 0 12px; border-bottom: 1px solid #333; padding-bottom: 6px; }
    .subtitle { color: #888; font-size: 14px; margin-bottom: 20px; }
    .disclaimer { color: #ef4444; font-size: 12px; margin-bottom: 24px; font-style: italic; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
    .card { background: #1a1a2e; border: 1px solid #333; border-radius: 10px; padding: 14px; }
    .card .label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
    .card .value { font-size: 24px; font-weight: 700; margin: 4px 0; }
    .card .sub { color: #666; font-size: 12px; }
    .positive { color: #10b981; } .negative { color: #ef4444; } .neutral { color: #f59e0b; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
    th { background: #1a1a2e; color: #888; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; padding: 10px 8px; text-align: left; border-bottom: 2px solid #333; }
    td { padding: 8px; border-bottom: 1px solid #222; }
    tr:hover { background: #1a1a2e; }
    .best-row { background: #0a2a1a !important; }
    .strategy-name { font-weight: 600; }
    .badge { background: #10b981; color: #000; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 700; margin-left: 6px; }
    .chart { width: 100%; border-radius: 8px; margin: 8px 0; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
    .mc-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; }
    .mc-stat { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 10px; text-align: center; }
    .mc-stat .val { font-size: 20px; font-weight: 700; }
    .footer { color: #555; font-size: 12px; margin-top: 40px; padding-top: 16px; border-top: 1px solid #222; text-align: center; }
"""

        equity_img = img_tag("equity", "Equity curves")
        mc_img = img_tag("monte_carlo", "Monte Carlo")
        dd_img = img_tag("drawdown", "Drawdown")
        feat_img = img_tag("features", "Features")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S&P 500 ML Backtest Report</title>
<style>{css}</style>
</head>
<body>

<h1>S&amp;P 500 ML Backtest Report</h1>
<div class="subtitle">Generated {now_str} | Test period: {test_range} | {n_features} features</div>
<div class="disclaimer">For educational/research purposes only — not financial advice. Past performance does not guarantee future results.</div>

<div class="cards">
    <div class="card"><div class="label">Best Strategy</div><div class="value positive" style="font-size:16px">{best}</div><div class="sub">Highest risk-adjusted returns</div></div>
    <div class="card"><div class="label">Best Sharpe</div><div class="value positive">{best_sharpe:.3f}</div><div class="sub">vs B&amp;H: {bh_sharpe:.3f}</div></div>
    <div class="card"><div class="label">Best Return</div><div class="value positive">{best_return:+.1f}%</div><div class="sub">vs B&amp;H: {bh_return:+.1f}%</div></div>
    <div class="card"><div class="label">Max Drawdown</div><div class="value negative">{best_dd:.1f}%</div><div class="sub">vs B&amp;H: {bh_dd:.1f}%</div></div>
    <div class="card"><div class="label">MC Prob of Profit</div><div class="value {prob_class}">{mc_prob:.1%}</div><div class="sub">1-year forward simulation</div></div>
    <div class="card"><div class="label">Test Period</div><div class="value neutral" style="font-size:14px">{test_range}</div><div class="sub">{test_size} trading days</div></div>
</div>

<h2>Strategy Comparison</h2>
{equity_img}

<table>
<thead><tr><th>Strategy</th><th>Return</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Vol</th><th>Win Rate</th><th>Profit Factor</th><th>Exposure</th></tr></thead>
<tbody>
    <tr class="best-row">
        <td class="strategy-name">Buy &amp; Hold (Baseline)</td>
        <td>{bh_return:+.1f}%</td><td>{bh_cagr:.1f}%</td><td>{bh_sharpe:.3f}</td><td>{bh_sortino:.3f}</td>
        <td class="negative">{bh_dd:.1f}%</td><td>{bh_vol:.1f}%</td><td>—</td><td>—</td><td>100%</td>
    </tr>
    {strat_rows}
</tbody>
</table>

<h2>Monte Carlo Simulation</h2>
<p style="color:#888;font-size:13px">1,000 bootstrapped paths simulating 1 year forward using the best strategy's daily returns.</p>

<div class="mc-stats">
    <div class="mc-stat"><div class="val positive">{mc_prob:.1%}</div><div class="label">Prob of Profit</div></div>
    <div class="mc-stat"><div class="val">{mc_expected:+.1f}%</div><div class="label">Expected Return</div></div>
    <div class="mc-stat"><div class="val">{mc_p50:.2f}x</div><div class="label">Median Outcome</div></div>
    <div class="mc-stat"><div class="val negative">{mc_p5:.2f}x</div><div class="label">5th Percentile</div></div>
    <div class="mc-stat"><div class="val positive">{mc_p95:.2f}x</div><div class="label">95th Percentile</div></div>
    <div class="mc-stat"><div class="val negative">{mc_mdd:.1f}%</div><div class="label">Median Max DD</div></div>
</div>
{mc_img}

<h2>Drawdown Analysis</h2>
{dd_img}

<div class="grid-2">
<div><h2>Model Accuracy</h2>
<table><thead><tr><th>Model</th><th>Metrics</th></tr></thead><tbody>{model_rows}</tbody></table></div>
<div><h2>Feature Importance</h2>{feat_img}</div>
</div>

<div class="footer">
    S&amp;P 500 ML Prediction System | Train: {train_range} | Test: {test_range} | {n_features} features<br>
    For educational purposes only — not financial advice
</div>

</body>
</html>"""
        return html


# ========================== CLI ========================== #
def find_data():
    for p in ["SP500_Analysis.xlsx", os.path.expanduser("~/Downloads/SP500_Analysis.xlsx")]:
        if os.path.exists(p):
            return p
    return None


def interactive():
    runner = BacktestRunner()

    print("\n" + "=" * 56)
    print("   S&P 500 Advanced Backtester")
    print("=" * 56)
    print(f"   LSTM: {'Available' if TORCH_AVAILABLE else 'Not available (pip install torch)'}")
    print(f"   Charts: {'Available' if MPL_AVAILABLE else 'Not available'}")

    data_path = find_data()
    if data_path:
        print(f"   Data: {data_path}")
    else:
        print("   ⚠️  SP500_Analysis.xlsx not found")

    print("""
   1. Run full backtest + generate report
   2. Generate HTML report (from existing results)
   3. Run Monte Carlo only
   4. Print strategy comparison
   5. Quit
    """)

    choice = input("   Select (1-5): ").strip()

    if choice == "1":
        if not data_path:
            print("   No data file found!")
            return
        results = runner.run(data_path,
                             progress_cb=lambda msg, pct: print(f"   [{int(pct*100):3d}%] {msg}"))
        report = HTMLReportGenerator(results)
        path = report.generate()
        print(f"\n   ✓ Report: {path}")
        print(f"   Open in browser: file://{os.path.abspath(path)}")

        # Print summary
        _print_summary(results)

    elif choice == "2":
        if runner.load():
            report = HTMLReportGenerator(runner.results)
            path = report.generate()
            print(f"\n   ✓ Report: {path}")
        else:
            print("   No existing results. Run a backtest first.")

    elif choice == "3":
        if runner.load():
            best = runner.results.get("best_strategy", "")
            eq = runner.results.get("strategy_equities", {}).get(best, [])
            if eq:
                daily = np.diff(np.log(np.clip(eq, 1e-10, None)))
                mc = MonteCarloSimulator.simulate(daily)
                print(f"\n   Monte Carlo (1-year, 1000 sims) for: {best}")
                for k, v in mc.items():
                    if k != "paths" and k != "final_values":
                        print(f"   {k}: {v}")
        else:
            print("   No existing results. Run a backtest first.")

    elif choice == "4":
        if runner.load():
            _print_summary(runner.results)
        else:
            print("   No existing results.")

    elif choice == "5":
        return


def _print_summary(results):
    strats = results.get("strategies", {})
    bh = results.get("buy_hold", {}).get("metrics", {})
    best = results.get("best_strategy", "N/A")

    print(f"\n   {'Strategy':<30} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("   " + "-" * 68)
    print(f"   {'Buy & Hold':<30} {bh.get('total_return',0):>+7.1f}% {bh.get('sharpe',0):>8.3f} "
          f"{bh.get('max_drawdown',0):>7.1f}% {'—':>8}")

    sorted_s = sorted(strats.items(), key=lambda x: x[1]["metrics"].get("sharpe", -999), reverse=True)
    for name, s in sorted_s:
        m = s["metrics"]
        flag = " ★" if name == best else ""
        print(f"   {name:<30} {m.get('total_return',0):>+7.1f}% {m.get('sharpe',0):>8.3f} "
              f"{m.get('max_drawdown',0):>7.1f}% {m.get('win_rate',0):>7.1%}{flag}")

    mc = results.get("monte_carlo", {})
    if mc:
        print(f"\n   Monte Carlo (best strategy, 1yr):")
        print(f"   Prob of profit: {mc.get('prob_profit',0):.1%} | "
              f"Median: {mc.get('percentiles',{}).get('p50',1):.2f}x | "
              f"5th-95th: {mc.get('percentiles',{}).get('p5',1):.2f}x – "
              f"{mc.get('percentiles',{}).get('p95',1):.2f}x")


# ========================== ENTRY ========================== #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S&P 500 Advanced Backtester")
    parser.add_argument("--full", action="store_true", help="Run full backtest + report")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation")
    parser.add_argument("--data", type=str, default=None, help="Path to Excel data")
    args = parser.parse_args()

    if args.full:
        data_path = args.data or find_data()
        if not data_path:
            print("Data file not found!")
            sys.exit(1)
        runner = BacktestRunner()
        results = runner.run(data_path,
                             progress_cb=lambda msg, pct: print(f"[{int(pct*100):3d}%] {msg}"))
        report = HTMLReportGenerator(results)
        path = report.generate()
        print(f"\nReport: {path}")
        _print_summary(results)

    elif args.report_only:
        runner = BacktestRunner()
        if runner.load():
            report = HTMLReportGenerator(runner.results)
            path = report.generate()
            print(f"Report: {path}")
        else:
            print("No results found. Run --full first.")

    elif args.monte_carlo:
        runner = BacktestRunner()
        if runner.load():
            best = runner.results.get("best_strategy", "")
            eq = runner.results.get("strategy_equities", {}).get(best, [])
            if eq:
                daily = np.diff(np.log(np.clip(eq, 1e-10, None)))
                mc = MonteCarloSimulator.simulate(daily, n_sims=5000)
                for k, v in mc.items():
                    if k not in ("paths", "final_values"):
                        print(f"{k}: {v}")
    else:
        interactive()
