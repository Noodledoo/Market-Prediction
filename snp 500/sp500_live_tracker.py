#!/usr/bin/env python3
"""
sp500_live_tracker.py — Live Performance Tracking & Alerts
Plugs into sp500_evolution.py and adds:

  1. PAPER TRADING     Record daily predictions, compare to actual outcomes
  2. ALERT SYSTEM      Email/desktop notifications for drift & signals
  3. PERFORMANCE LOG    Track cumulative P&L, accuracy over rolling windows
  4. PYGAME DASHBOARD   Full visual evolution + live performance tracker

Usage:
    python sp500_live_tracker.py                # Pygame dashboard
    python sp500_live_tracker.py --record       # Record today's prediction (cron)
    python sp500_live_tracker.py --evaluate     # Score yesterday's prediction
    python sp500_live_tracker.py --report       # Print performance report
    python sp500_live_tracker.py --alert-test   # Send a test alert

Requires: pandas, scikit-learn, numpy, matplotlib, pygame
Optional: torch, yfinance

⚠️ For educational/research purposes only — not financial advice.
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import smtplib
import platform
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# Import evolution system
try:
    from sp500_evolution import (
        DataUpdater, FeatureEngine, WalkForwardEngine,
        ModelTournament, DriftDetector, EvolutionPipeline,
        DATA_DIR, MODEL_DIR, LOG_DIR
    )
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    DATA_DIR = "sp500_data"
    MODEL_DIR = "sp500_models"
    LOG_DIR = "sp500_logs"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "live_tracker.log"))
    ]
)
log = logging.getLogger("tracker")

# ========================== CONFIG ========================== #
PAPER_TRADES_FILE = os.path.join(LOG_DIR, "paper_trades.json")
ALERT_CONFIG_FILE = os.path.join(LOG_DIR, "alert_config.json")
LIVE_MODELS_DIR = os.path.join(MODEL_DIR, "live")
PERFORMANCE_FILE = os.path.join(LOG_DIR, "performance_history.json")
DAILY_SNAPSHOT_DIR = os.path.join(LOG_DIR, "daily_snapshots")

for d in [LIVE_MODELS_DIR, DAILY_SNAPSHOT_DIR]:
    os.makedirs(d, exist_ok=True)

COLORS = {
    "bg": (20, 20, 30), "panel": (30, 30, 42), "card": (40, 40, 55),
    "text": (220, 220, 220), "dim": (120, 120, 130), "accent": (0, 220, 120),
    "warn": (220, 180, 50), "err": (220, 70, 70), "blue": (80, 160, 255),
    "purple": (160, 120, 255), "grid": (45, 45, 60),
    "green_bg": (20, 60, 40), "red_bg": (60, 20, 20),
    "rf": (16, 185, 129), "gb": (245, 158, 11), "bh": (107, 114, 128),
}


# ========================== 1. ALERT SYSTEM ========================== #
class AlertSystem:
    """Multi-channel alert system for drift warnings and trading signals."""

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        if os.path.exists(ALERT_CONFIG_FILE):
            with open(ALERT_CONFIG_FILE) as f:
                return json.load(f)
        # Default config — user fills in their details
        default = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender": "your_email@gmail.com",
                "password": "",   # Use app password for Gmail
                "recipient": "your_email@gmail.com",
            },
            "desktop": {
                "enabled": True,
            },
            "thresholds": {
                "drift_alert": True,
                "daily_prediction": True,
                "weekly_summary": True,
                "high_confidence_signal": 0.65,  # Alert when prob > this
            }
        }
        with open(ALERT_CONFIG_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default

    def save_config(self):
        with open(ALERT_CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

    def send(self, subject, body, level="INFO"):
        """Send alert through all enabled channels."""
        log.info(f"ALERT [{level}]: {subject}")

        # Desktop notification
        if self.config.get("desktop", {}).get("enabled", True):
            self._desktop_notify(subject, body)

        # Email
        if self.config.get("email", {}).get("enabled", False):
            self._email_notify(subject, body)

    def _desktop_notify(self, title, message):
        """Cross-platform desktop notification."""
        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                os.system(f"""osascript -e 'display notification "{message}" with title "{title}"'""")
            elif system == "Linux":
                os.system(f'notify-send "{title}" "{message}" 2>/dev/null')
            elif system == "Windows":
                # Use Windows toast notification via PowerShell
                ps_cmd = f"""
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                $template.GetElementsByTagName('text')[0].AppendChild($template.CreateTextNode('{title}')) | Out-Null
                $template.GetElementsByTagName('text')[1].AppendChild($template.CreateTextNode('{message}')) | Out-Null
                """
                # Fallback: simple print
                print(f"\n🔔 {title}: {message}\n")
            else:
                print(f"\n🔔 {title}: {message}\n")
        except Exception:
            print(f"\n🔔 {title}: {message}\n")

    def _email_notify(self, subject, body):
        """Send email notification."""
        cfg = self.config.get("email", {})
        if not cfg.get("enabled") or not cfg.get("password"):
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = cfg["sender"]
            msg["To"] = cfg["recipient"]
            msg["Subject"] = f"[SP500 ML] {subject}"

            # HTML body
            html = f"""
            <html><body style="font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px;">
            <h2 style="color: #00dc78;">S&P 500 ML Predictor</h2>
            <pre style="background: #262640; padding: 15px; border-radius: 8px; color: #d0d0d0;">
{body}
            </pre>
            <p style="color: #888; font-size: 12px;">
                Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | For research purposes only
            </p>
            </body></html>
            """
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["sender"], cfg["password"])
                server.send_message(msg)
            log.info("Email alert sent successfully.")
        except Exception as e:
            log.warning(f"Email send failed: {e}")

    def alert_drift(self, alerts):
        """Send drift detection alerts."""
        if not alerts or not self.config.get("thresholds", {}).get("drift_alert"):
            return
        body = "DRIFT DETECTED IN MODEL PERFORMANCE\n" + "=" * 40 + "\n\n"
        for a in alerts:
            body += (f"  [{a['severity']}] {a['model']}\n"
                     f"  Metric: {a['metric']} dropped {a['drop']:.4f}\n"
                     f"  Historical: {a['early_avg']:.4f} → Recent: {a['recent_avg']:.4f}\n\n")
        body += "Consider retraining or reviewing feature configuration."
        self.send("⚠️ Model Drift Detected", body, level="WARNING")

    def alert_prediction(self, prediction):
        """Send daily prediction alert."""
        if not self.config.get("thresholds", {}).get("daily_prediction"):
            return
        prob = prediction.get("rf_dir_prob", 0.5)
        direction = "BULLISH ↑" if prob > 0.5 else "BEARISH ↓"
        confidence = abs(prob - 0.5) * 200

        body = f"DAILY PREDICTION — {prediction.get('date', 'Today')}\n"
        body += "=" * 40 + "\n\n"
        body += f"  Direction:  {direction}\n"
        body += f"  RF Prob:    {prob:.1%} (confidence: {confidence:.0f}%)\n"
        body += f"  GB Prob:    {prediction.get('gb_dir_prob', 0.5):.1%}\n"
        body += f"  Pred Ret:   {prediction.get('rf_ret_pred', 0)*100:.3f}%\n"
        body += f"  Pred Vol:   {prediction.get('rf_vol_pred', 0)*100:.1f}%\n"

        # High confidence signal
        threshold = self.config.get("thresholds", {}).get("high_confidence_signal", 0.65)
        if prob > threshold:
            body += f"\n  🔥 HIGH CONFIDENCE BULLISH SIGNAL (>{threshold:.0%})\n"
        elif prob < (1 - threshold):
            body += f"\n  ❄️ HIGH CONFIDENCE BEARISH SIGNAL (<{1-threshold:.0%})\n"

        self.send(f"📊 {direction} — {confidence:.0f}% confidence", body)

    def alert_weekly_summary(self, stats):
        """Send weekly performance summary."""
        if not self.config.get("thresholds", {}).get("weekly_summary"):
            return
        body = "WEEKLY PERFORMANCE SUMMARY\n" + "=" * 40 + "\n\n"
        body += f"  Period:     {stats.get('period', 'Last 5 days')}\n"
        body += f"  Predictions: {stats.get('total', 0)}\n"
        body += f"  Correct:    {stats.get('correct', 0)} ({stats.get('accuracy', 0):.1%})\n"
        body += f"  Strategy:   {stats.get('strategy_return', 0):+.2f}%\n"
        body += f"  Buy&Hold:   {stats.get('bh_return', 0):+.2f}%\n"
        body += f"  Cumulative: {stats.get('cumulative_return', 0):+.2f}%\n"
        self.send("📈 Weekly Summary", body)


# ========================== 2. PAPER TRADING TRACKER ========================== #
class PaperTrader:
    """Track predictions vs actual outcomes without real money."""

    def __init__(self):
        self.trades = self._load_trades()
        self.alerts = AlertSystem()

    def _load_trades(self):
        if os.path.exists(PAPER_TRADES_FILE):
            with open(PAPER_TRADES_FILE) as f:
                return json.load(f)
        return []

    def _save_trades(self):
        with open(PAPER_TRADES_FILE, "w") as f:
            json.dump(self.trades, f, indent=2)

    def record_prediction(self, date, predictions):
        """Record today's model predictions before market close."""
        # Check for duplicate
        if self.trades and self.trades[-1].get("date") == date:
            log.info(f"Prediction for {date} already recorded. Updating.")
            self.trades[-1]["predictions"] = predictions
        else:
            entry = {
                "date": date,
                "timestamp": datetime.now().isoformat(),
                "predictions": predictions,
                "actual": None,
                "scored": False,
            }
            self.trades.append(entry)

        self._save_trades()
        log.info(f"Recorded prediction for {date}")

        # Send alert
        pred_summary = predictions.get("rf_direction", {})
        self.alerts.alert_prediction({
            "date": date,
            "rf_dir_prob": pred_summary.get("probability", 0.5),
            "gb_dir_prob": predictions.get("gb_direction", {}).get("probability", 0.5),
            "rf_ret_pred": predictions.get("rf_return_1d", {}).get("value", 0),
            "rf_vol_pred": predictions.get("rf_volatility", {}).get("value", 0),
        })

    def evaluate(self, date, actual_close, prev_close):
        """Score a previous prediction against actual outcome."""
        actual_return = (actual_close - prev_close) / prev_close
        actual_direction = 1 if actual_return > 0 else 0

        for trade in reversed(self.trades):
            if trade["date"] == date and not trade.get("scored"):
                trade["actual"] = {
                    "close": actual_close,
                    "prev_close": prev_close,
                    "return": round(actual_return, 6),
                    "direction": actual_direction,
                }

                # Score each model's prediction
                preds = trade["predictions"]
                trade["scores"] = {}
                for model_name, pred in preds.items():
                    if "direction" in model_name:
                        correct = int(pred.get("value", -1) == actual_direction)
                        trade["scores"][model_name] = {
                            "correct": correct,
                            "predicted": pred.get("value"),
                            "probability": pred.get("probability"),
                            "actual": actual_direction,
                        }
                    elif "return" in model_name or "volatility" in model_name:
                        error = abs(pred.get("value", 0) - actual_return)
                        trade["scores"][model_name] = {
                            "predicted": pred.get("value"),
                            "actual": actual_return,
                            "error": round(error, 6),
                        }

                trade["scored"] = True
                self._save_trades()
                log.info(f"Scored {date}: actual_ret={actual_return:+.4f}, dir={actual_direction}")
                return trade["scores"]

        log.warning(f"No unscored prediction found for {date}")
        return None

    def get_performance(self, last_n=None):
        """Compute cumulative performance metrics."""
        scored = [t for t in self.trades if t.get("scored")]
        if last_n:
            scored = scored[-last_n:]
        if not scored:
            return {}

        # Direction accuracy per model
        model_stats = defaultdict(lambda: {"correct": 0, "total": 0, "equity": [1.0]})

        for trade in scored:
            actual_ret = trade["actual"]["return"]
            for model_name, score in trade.get("scores", {}).items():
                if "direction" in model_name:
                    model_stats[model_name]["total"] += 1
                    model_stats[model_name]["correct"] += score.get("correct", 0)

                    # Paper equity: long if predicted up, cash if predicted down
                    eq = model_stats[model_name]["equity"][-1]
                    if score.get("predicted") == 1:
                        eq *= (1 + actual_ret)
                    model_stats[model_name]["equity"].append(eq)

        # Buy and hold equity
        bh_equity = [1.0]
        for trade in scored:
            bh_equity.append(bh_equity[-1] * (1 + trade["actual"]["return"]))

        result = {
            "total_predictions": len(scored),
            "date_range": f"{scored[0]['date']} to {scored[-1]['date']}" if scored else "",
            "buy_hold_equity": bh_equity,
            "buy_hold_return": round((bh_equity[-1] - 1) * 100, 2),
            "models": {},
        }

        for model_name, stats in model_stats.items():
            acc = stats["correct"] / max(stats["total"], 1)
            eq = stats["equity"]
            ret = (eq[-1] - 1) * 100
            result["models"][model_name] = {
                "accuracy": round(acc, 4),
                "correct": stats["correct"],
                "total": stats["total"],
                "equity": eq,
                "return": round(ret, 2),
            }

        return result

    def get_recent_trades(self, n=30):
        """Get last n trades for display."""
        return self.trades[-n:]

    def check_weekly_summary(self):
        """Send weekly summary if it's Friday or enough trades accumulated."""
        scored = [t for t in self.trades if t.get("scored")]
        if len(scored) < 5:
            return

        last_5 = scored[-5:]
        correct = sum(1 for t in last_5
                      for s in t.get("scores", {}).values()
                      if "correct" in s and s["correct"])
        total_dir = sum(1 for t in last_5
                        for s in t.get("scores", {}).values()
                        if "correct" in s)

        bh_ret = 1.0
        strat_ret = 1.0
        for t in last_5:
            r = t["actual"]["return"]
            bh_ret *= (1 + r)
            # Use RF direction
            rf_score = t.get("scores", {}).get("rf_direction", {})
            if rf_score.get("predicted") == 1:
                strat_ret *= (1 + r)

        perf = self.get_performance()
        cum_ret = 0
        if perf.get("models", {}).get("rf_direction"):
            cum_ret = perf["models"]["rf_direction"]["return"]

        self.alerts.alert_weekly_summary({
            "period": f"{last_5[0]['date']} to {last_5[-1]['date']}",
            "total": total_dir,
            "correct": correct,
            "accuracy": correct / max(total_dir, 1),
            "strategy_return": (strat_ret - 1) * 100,
            "bh_return": (bh_ret - 1) * 100,
            "cumulative_return": cum_ret,
        })


# ========================== 3. LIVE MODEL MANAGER ========================== #
class LiveModelManager:
    """Train and manage production models for daily predictions."""

    SAVE_PATH = os.path.join(LIVE_MODELS_DIR, "live_models.pkl")

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_cols = []
        self.last_trained = None

    def train(self, df, feature_cols, progress_cb=None):
        """Train production models on ALL available data."""
        target_cols = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']
        valid = df.dropna(subset=feature_cols + target_cols).copy()

        X = valid[feature_cols].values
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        self.feature_cols = feature_cols

        configs = {
            "rf_direction": (
                RandomForestClassifier(n_estimators=300, max_depth=10,
                                       min_samples_leaf=20, random_state=42, n_jobs=-1),
                "target_dir_1d", "classification"
            ),
            "gb_direction": (
                GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                           learning_rate=0.03, min_samples_leaf=20, random_state=42),
                "target_dir_1d", "classification"
            ),
            "rf_return_1d": (
                RandomForestRegressor(n_estimators=300, max_depth=10,
                                      min_samples_leaf=20, random_state=42, n_jobs=-1),
                "target_ret_1d", "regression"
            ),
            "rf_volatility": (
                RandomForestRegressor(n_estimators=300, max_depth=10,
                                      min_samples_leaf=20, random_state=42, n_jobs=-1),
                "target_vol_5d", "regression"
            ),
        }

        total = len(configs)
        for i, (name, (model, target, task)) in enumerate(configs.items()):
            if progress_cb:
                progress_cb(f"Training {name}...", (i + 0.5) / total)
            y = valid[target].values
            model.fit(X_s, y)
            self.models[name] = {"model": model, "task": task}

        self.last_trained = datetime.now().isoformat()

        if progress_cb:
            progress_cb("Live models trained!", 1.0)

        self.save()
        log.info(f"Live models trained on {len(valid)} samples.")

    def predict(self, df_row_features):
        """Generate predictions for a single day's features."""
        if not self.models or self.scaler is None:
            log.error("No live models trained. Run train() first.")
            return None

        X = np.array(df_row_features).reshape(1, -1)
        X_s = self.scaler.transform(X)

        predictions = {}
        for name, info in self.models.items():
            model = info["model"]
            if info["task"] == "classification":
                pred = int(model.predict(X_s)[0])
                prob = float(model.predict_proba(X_s)[0][1])
                predictions[name] = {"value": pred, "probability": prob}
            else:
                pred = float(model.predict(X_s)[0])
                predictions[name] = {"value": pred}

        return predictions

    def save(self):
        with open(self.SAVE_PATH, "wb") as f:
            pickle.dump({
                "models": self.models,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "last_trained": self.last_trained,
            }, f)

    def load(self):
        if not os.path.exists(self.SAVE_PATH):
            return False
        try:
            with open(self.SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            self.models = data["models"]
            self.scaler = data["scaler"]
            self.feature_cols = data["feature_cols"]
            self.last_trained = data.get("last_trained")
            log.info(f"Loaded live models (trained: {self.last_trained})")
            return True
        except Exception as e:
            log.error(f"Failed to load live models: {e}")
            return False


# ========================== 4. DAILY PIPELINE ========================== #
def run_daily_cycle(excel_path=None):
    """Full daily pipeline: update data → predict → evaluate yesterday → alert."""
    log.info("=" * 50)
    log.info("DAILY CYCLE STARTING")
    log.info("=" * 50)

    # 1. Update data
    updater = DataUpdater()
    if not updater.load_existing(excel_path):
        log.error("No data found. Provide Excel path on first run.")
        return

    updater.fetch_new_data()
    df = updater.df

    # 2. Engineer features
    engine = FeatureEngine()
    df, feature_cols = engine.engineer(df)
    df = engine.add_targets(df)

    # 3. Train or load live models
    live = LiveModelManager()
    if not live.load() or _should_retrain(live):
        log.info("Training fresh live models...")
        live.train(df, feature_cols,
                   progress_cb=lambda msg, pct: log.info(f"  [{int(pct*100):3d}%] {msg}"))

    # 4. Make today's prediction
    valid = df.dropna(subset=feature_cols)
    if len(valid) > 0:
        latest = valid.iloc[-1]
        today = latest['Date'].strftime('%Y-%m-%d')
        features = latest[feature_cols].values

        predictions = live.predict(features)
        if predictions:
            trader = PaperTrader()
            trader.record_prediction(today, predictions)

            log.info(f"Today's predictions ({today}):")
            for name, pred in predictions.items():
                if "value" in pred and "probability" in pred:
                    log.info(f"  {name}: {pred['value']} (prob={pred['probability']:.3f})")
                else:
                    log.info(f"  {name}: {pred.get('value', 'N/A'):.6f}")

    # 5. Evaluate yesterday's prediction
    if len(valid) >= 2:
        yesterday = valid.iloc[-2]
        y_date = yesterday['Date'].strftime('%Y-%m-%d')
        y_close = float(valid.iloc[-1]['Close'])
        prev_close = float(yesterday['Close'])

        trader = PaperTrader()
        scores = trader.evaluate(y_date, y_close, prev_close)
        if scores:
            log.info(f"Yesterday's scores ({y_date}):")
            for name, score in scores.items():
                log.info(f"  {name}: {score}")

        # Weekly summary check (every 5 trades)
        trader.check_weekly_summary()

    log.info("Daily cycle complete.")


def _should_retrain(live, max_age_days=7):
    """Check if live models need retraining."""
    if live.last_trained is None:
        return True
    try:
        trained = datetime.fromisoformat(live.last_trained)
        age = (datetime.now() - trained).days
        return age >= max_age_days
    except Exception:
        return True


# ========================== 5. PERFORMANCE REPORTER ========================== #
def print_performance_report():
    """Print a formatted performance report to terminal."""
    trader = PaperTrader()
    perf = trader.get_performance()

    if not perf or not perf.get("models"):
        print("\n  No scored predictions yet. Run some daily cycles first.")
        return

    print("\n" + "=" * 60)
    print("  S&P 500 ML — LIVE PERFORMANCE REPORT")
    print("=" * 60)
    print(f"\n  Period:      {perf['date_range']}")
    print(f"  Predictions: {perf['total_predictions']}")
    print(f"  Buy & Hold:  {perf['buy_hold_return']:+.2f}%")

    print("\n  Model Results:")
    print("  " + "-" * 50)
    for name, stats in perf["models"].items():
        print(f"  {name:25s}  Acc: {stats['accuracy']:.1%}  "
              f"({stats['correct']}/{stats['total']})  "
              f"Return: {stats['return']:+.2f}%")

    # Recent trades
    recent = trader.get_recent_trades(10)
    if recent:
        print(f"\n  Last {len(recent)} Trades:")
        print("  " + "-" * 50)
        for t in recent:
            date = t["date"]
            if t.get("scored"):
                act_dir = "↑" if t["actual"]["direction"] == 1 else "↓"
                act_ret = t["actual"]["return"] * 100
                rf_correct = t.get("scores", {}).get("rf_direction", {}).get("correct", "?")
                mark = "✓" if rf_correct == 1 else "✗"
                rf_prob = t["predictions"].get("rf_direction", {}).get("probability", 0)
                print(f"  {date}  {act_dir} {act_ret:+.2f}%  RF: {rf_prob:.0%} {mark}")
            else:
                rf_prob = t["predictions"].get("rf_direction", {}).get("probability", 0)
                direction = "↑" if rf_prob > 0.5 else "↓"
                print(f"  {date}  {direction} RF: {rf_prob:.0%}  (pending)")

    print()


# ========================== 6. PYGAME DASHBOARD ========================== #
def plot_to_surface(plot_fn, width, height):
    """Render matplotlib plot to pygame surface."""
    if not MPL_AVAILABLE:
        return None
    plt.ioff()
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#1e1e2a")
    ax.set_facecolor("#1e1e2a")
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444455")
    plot_fn(ax)
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    surf = pygame.image.frombuffer(buf, (w, h), "RGBA")
    plt.close(fig)
    return surf


def draw_text(screen, text, pos, font, color=COLORS["text"]):
    screen.blit(font.render(str(text), True, color), pos)


def draw_bar(screen, rect, pct, color=COLORS["accent"], bg=COLORS["card"]):
    pygame.draw.rect(screen, bg, rect, border_radius=3)
    filled = pygame.Rect(rect.x, rect.y, max(int(rect.width * pct), 1), rect.height)
    pygame.draw.rect(screen, color, filled, border_radius=3)


def run_dashboard():
    """Full Pygame evolution + live tracking dashboard."""
    if not PYGAME_AVAILABLE:
        print("pygame not installed. Run: pip install pygame")
        return

    pygame.init()
    W, H = 1100, 750
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("S&P 500 ML — Live Dashboard")
    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 32)
    font_sm = pygame.font.SysFont(None, 20)
    font_xs = pygame.font.SysFont(None, 17)
    clock = pygame.time.Clock()

    # Load data
    trader = PaperTrader()
    perf = trader.get_performance()
    recent = trader.get_recent_trades(25)

    # Load evolution history
    evo_history = []
    history_file = os.path.join(LOG_DIR, "evolution_history.json")
    if os.path.exists(history_file):
        with open(history_file) as f:
            evo_history = json.load(f)

    # Load tournament results
    tournament_data = []
    tournament_file = os.path.join(MODEL_DIR, "tournament_results.json")
    if os.path.exists(tournament_file):
        with open(tournament_file) as f:
            tournament_data = json.load(f)

    # Load drift log
    drift_alerts = []
    drift_file = os.path.join(LOG_DIR, "drift_log.json")
    if os.path.exists(drift_file):
        with open(drift_file) as f:
            drift_alerts = json.load(f)

    # Pre-render charts
    charts = {}
    if MPL_AVAILABLE and perf and perf.get("models"):
        def plot_equity(ax):
            for name, stats in perf["models"].items():
                color = '#10b981' if 'rf' in name else '#f59e0b'
                label = name.replace("_", " ").title()
                ax.plot(stats["equity"], color=color, linewidth=1.5, label=label)
            ax.plot(perf["buy_hold_equity"], color='#6b7280', linewidth=1,
                    linestyle='--', label="Buy & Hold", alpha=0.7)
            ax.set_title("Paper Trading Equity", color="#cccccc", fontsize=11)
            ax.legend(fontsize=8, facecolor="#2a2a3a", edgecolor="#444455", labelcolor="#cccccc")
            ax.grid(True, linestyle="--", alpha=0.3)
        charts["equity"] = plot_to_surface(plot_equity, 500, 230)

    if MPL_AVAILABLE and recent:
        scored = [t for t in recent if t.get("scored")]
        if len(scored) >= 3:
            def plot_accuracy(ax):
                # Rolling accuracy
                rf_correct = []
                for t in scored:
                    c = t.get("scores", {}).get("rf_direction", {}).get("correct", 0)
                    rf_correct.append(c)
                if len(rf_correct) >= 5:
                    rolling = pd.Series(rf_correct).rolling(5, min_periods=3).mean()
                    ax.plot(rolling.values, color='#10b981', linewidth=2, label="RF 5-day rolling")
                ax.axhline(y=0.5, color='#6b7280', linestyle='--', alpha=0.5)
                ax.set_ylim(0.2, 0.8)
                ax.set_title("Rolling Direction Accuracy", color="#cccccc", fontsize=11)
                ax.legend(fontsize=8, facecolor="#2a2a3a", edgecolor="#444455", labelcolor="#cccccc")
                ax.grid(True, linestyle="--", alpha=0.3)
            charts["accuracy"] = plot_to_surface(plot_accuracy, 500, 230)

    # Tabs
    tabs = ["Live Trades", "Performance", "Tournament", "Drift", "History"]
    tab = 0
    scroll = 0

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_LEFT:
                    tab = (tab - 1) % len(tabs)
                    scroll = 0
                elif e.key == pygame.K_RIGHT:
                    tab = (tab + 1) % len(tabs)
                    scroll = 0
                elif e.key == pygame.K_UP:
                    scroll = max(0, scroll - 1)
                elif e.key == pygame.K_DOWN:
                    scroll += 1
            elif e.type == pygame.MOUSEBUTTONDOWN:
                for i in range(len(tabs)):
                    tx = 25 + i * 140
                    if tx <= e.pos[0] <= tx + 130 and 55 <= e.pos[1] <= 80:
                        tab = i
                        scroll = 0

        screen.fill(COLORS["bg"])

        # Header
        draw_text(screen, "S&P 500 ML — Live Dashboard", (25, 15), font_big, COLORS["accent"])

        # Status bar
        n_trades = len([t for t in trader.trades if t.get("scored")])
        n_alerts = len(drift_alerts)
        status = f"Trades: {n_trades} | Drift alerts: {n_alerts}"
        if perf and perf.get("models", {}).get("rf_direction"):
            rf = perf["models"]["rf_direction"]
            status += f" | RF Acc: {rf['accuracy']:.1%} | Return: {rf['return']:+.1f}%"
        draw_text(screen, status, (25, 45), font_sm, COLORS["dim"])

        # Tab bar
        for i, t in enumerate(tabs):
            tx = 25 + i * 140
            rect = pygame.Rect(tx, 68, 130, 25)
            if i == tab:
                pygame.draw.rect(screen, (20, 60, 45), rect, border_radius=4)
                pygame.draw.rect(screen, COLORS["accent"], rect, 1, border_radius=4)
            draw_text(screen, t, (tx + 10, 72), font_sm,
                     COLORS["accent"] if i == tab else COLORS["dim"])

        y0 = 105

        # ─── Tab: Live Trades ───
        if tab == 0:
            draw_text(screen, "Recent Predictions & Outcomes", (25, y0), font, COLORS["text"])
            y = y0 + 30

            # Header row
            cols = ["Date", "RF Prob", "RF Dir", "GB Dir", "Actual", "Return", "RF ✓"]
            col_x = [25, 140, 240, 330, 420, 510, 620]
            for ci, (col, cx) in enumerate(zip(cols, col_x)):
                draw_text(screen, col, (cx, y), font_sm, COLORS["dim"])
            y += 22
            pygame.draw.line(screen, COLORS["grid"], (25, y), (W - 25, y))
            y += 5

            display_trades = list(reversed(recent))[scroll:scroll+22]
            for trade in display_trades:
                preds = trade.get("predictions", {})
                rf_prob = preds.get("rf_direction", {}).get("probability", 0)
                rf_dir = "↑" if preds.get("rf_direction", {}).get("value") == 1 else "↓"
                gb_dir = "↑" if preds.get("gb_direction", {}).get("value") == 1 else "↓"

                draw_text(screen, trade["date"], (col_x[0], y), font_sm, COLORS["text"])
                prob_color = COLORS["rf"] if rf_prob > 0.55 else COLORS["err"] if rf_prob < 0.45 else COLORS["dim"]
                draw_text(screen, f"{rf_prob:.1%}", (col_x[1], y), font_sm, prob_color)
                draw_text(screen, rf_dir, (col_x[2], y), font_sm,
                         COLORS["rf"] if rf_dir == "↑" else COLORS["err"])
                draw_text(screen, gb_dir, (col_x[3], y), font_sm,
                         COLORS["gb"] if gb_dir == "↑" else COLORS["err"])

                if trade.get("scored") and trade.get("actual"):
                    act = trade["actual"]
                    act_dir = "↑" if act["direction"] == 1 else "↓"
                    ret = act["return"] * 100
                    draw_text(screen, act_dir, (col_x[4], y), font_sm,
                             COLORS["rf"] if act_dir == "↑" else COLORS["err"])
                    draw_text(screen, f"{ret:+.2f}%", (col_x[5], y), font_sm,
                             COLORS["rf"] if ret > 0 else COLORS["err"])

                    rf_correct = trade.get("scores", {}).get("rf_direction", {}).get("correct")
                    if rf_correct is not None:
                        mark = "✓" if rf_correct else "✗"
                        mark_color = COLORS["accent"] if rf_correct else COLORS["err"]
                        draw_text(screen, mark, (col_x[6], y), font, mark_color)
                else:
                    draw_text(screen, "pending", (col_x[4], y), font_xs, COLORS["dim"])

                y += 22
                if y > H - 40:
                    break

        # ─── Tab: Performance ───
        elif tab == 1:
            if perf and perf.get("models"):
                draw_text(screen, "Cumulative Performance", (25, y0), font, COLORS["text"])
                y = y0 + 30

                for name, stats in perf["models"].items():
                    color = COLORS["rf"] if "rf" in name else COLORS["gb"]
                    acc = stats["accuracy"]
                    ret = stats["return"]

                    draw_text(screen, name.replace("_", " ").title(), (25, y), font_sm, color)
                    y += 20

                    # Accuracy bar
                    draw_text(screen, f"Accuracy: {acc:.1%}", (40, y), font_xs, COLORS["dim"])
                    bar = pygame.Rect(200, y + 2, 200, 12)
                    draw_bar(screen, bar, acc, color)
                    y += 18

                    # Return
                    ret_color = COLORS["accent"] if ret > 0 else COLORS["err"]
                    draw_text(screen, f"Return: {ret:+.2f}%", (40, y), font_xs, ret_color)
                    y += 25

                # Buy & hold
                draw_text(screen, f"Buy & Hold: {perf['buy_hold_return']:+.2f}%",
                         (25, y), font_sm, COLORS["bh"])
                y += 35

                # Charts
                if charts.get("equity"):
                    screen.blit(charts["equity"], (25, y))
                if charts.get("accuracy"):
                    screen.blit(charts["accuracy"], (550, y))

            else:
                draw_text(screen, "No scored predictions yet.", (25, y0 + 30), font, COLORS["dim"])
                draw_text(screen, "Run daily cycles to start building a track record.",
                         (25, y0 + 60), font_sm, COLORS["dim"])

        # ─── Tab: Tournament ───
        elif tab == 2:
            draw_text(screen, "Model Tournament Rankings", (25, y0), font, COLORS["text"])
            y = y0 + 30

            if tournament_data:
                latest = tournament_data[-1]
                draw_text(screen, f"Last run: {latest.get('timestamp', 'N/A')[:19]}",
                         (25, y), font_xs, COLORS["dim"])
                y += 25

                for family, names in latest.get("rankings", {}).items():
                    draw_text(screen, family.upper(), (25, y), font_sm, COLORS["accent"])
                    y += 22
                    metrics = latest.get("metrics", {}).get(family, {})

                    for i, name in enumerate(names):
                        prefix = "🏆 " if i == 0 else f" {i+1}. "
                        color = COLORS["accent"] if i == 0 else COLORS["text"] if i == 1 else COLORS["dim"]
                        draw_text(screen, f"{prefix}{name}", (40, y), font_sm, color)

                        m = metrics.get(name, {})
                        m_str = "  ".join(f"{k}={v:.4f}" for k, v in m.items()
                                         if isinstance(v, float) and 'mean' in k)
                        draw_text(screen, m_str, (380, y), font_xs, COLORS["dim"])
                        y += 20
                    y += 10
            else:
                draw_text(screen, "No tournament results yet.", (25, y), font_sm, COLORS["dim"])
                draw_text(screen, "Run: python sp500_evolution.py --update", (25, y + 25), font_sm, COLORS["warn"])

        # ─── Tab: Drift ───
        elif tab == 3:
            draw_text(screen, "Drift Detection Alerts", (25, y0), font, COLORS["text"])
            y = y0 + 30

            if drift_alerts:
                display = list(reversed(drift_alerts))[scroll:scroll+20]
                for a in display:
                    severity_color = COLORS["err"] if a["severity"] == "HIGH" else COLORS["warn"]
                    draw_text(screen, f"[{a['severity']}]", (25, y), font_sm, severity_color)
                    draw_text(screen, a["model"], (100, y), font_sm, COLORS["text"])
                    draw_text(screen, f"{a['metric']}: {a['early_avg']:.4f} → {a['recent_avg']:.4f} "
                             f"(Δ {a['drop']:+.4f})", (320, y), font_xs, COLORS["dim"])
                    ts = a.get("timestamp", "")[:16]
                    draw_text(screen, ts, (700, y), font_xs, COLORS["dim"])
                    y += 22
            else:
                draw_text(screen, "✓ No drift alerts — models performing within expected range.",
                         (25, y), font_sm, COLORS["accent"])

        # ─── Tab: History ───
        elif tab == 4:
            draw_text(screen, "Evolution Run History", (25, y0), font, COLORS["text"])
            y = y0 + 30

            if evo_history:
                display = list(reversed(evo_history))[scroll:scroll+15]
                for run in display:
                    ts = run.get("timestamp", "")[:16]
                    draw_text(screen, ts, (25, y), font_sm, COLORS["text"])

                    info_parts = [
                        f"Data: {run.get('data_rows', '?')}",
                        f"Features: {run.get('features', '?')}",
                        f"Folds: {run.get('wf_folds', '?')}",
                        f"Drift: {run.get('drift_alerts', 0)}",
                        f"Time: {run.get('elapsed_seconds', '?')}s",
                    ]
                    draw_text(screen, " | ".join(info_parts), (200, y), font_xs, COLORS["dim"])
                    y += 18

                    champs = run.get("champions", {})
                    if champs:
                        champ_str = "  Champions: " + ", ".join(
                            f"{f}={n}" for f, n in champs.items() if n)
                        draw_text(screen, champ_str, (200, y), font_xs, COLORS["accent"])
                    y += 26
            else:
                draw_text(screen, "No evolution runs yet.", (25, y), font_sm, COLORS["dim"])
                draw_text(screen, "Run: python sp500_evolution.py", (25, y + 25), font_sm, COLORS["warn"])

        # Footer
        draw_text(screen, "← → tabs | ↑ ↓ scroll | ESC quit", (25, H - 25), font_xs, COLORS["dim"])
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ========================== ENTRY POINT ========================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 ML Live Tracker")
    parser.add_argument("--record", action="store_true", help="Record today's prediction")
    parser.add_argument("--evaluate", action="store_true", help="Score yesterday's prediction")
    parser.add_argument("--report", action="store_true", help="Print performance report")
    parser.add_argument("--alert-test", action="store_true", help="Send test alert")
    parser.add_argument("--daily", action="store_true", help="Run full daily cycle")
    parser.add_argument("--data", type=str, default=None, help="Path to SP500_Analysis.xlsx")
    args = parser.parse_args()

    if args.report:
        print_performance_report()
    elif args.alert_test:
        alerts = AlertSystem()
        alerts.send("Test Alert", "This is a test notification from SP500 ML Predictor.")
    elif args.daily or args.record or args.evaluate:
        run_daily_cycle(args.data or "SP500_Analysis.xlsx")
    else:
        if PYGAME_AVAILABLE:
            run_dashboard()
        else:
            print_performance_report()
