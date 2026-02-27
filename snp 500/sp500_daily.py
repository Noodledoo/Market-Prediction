#!/usr/bin/env python3
"""
sp500_daily.py — Integrated Daily Pipeline

The single script to run every day after market close. Combines:
  • Data update (fetches latest from Yahoo Finance)
  • Model predictions (RF + GB direction, return, volatility)
  • Advisor signal (buy/sell/hold/wait with explanation)
  • Yesterday's score (how accurate was the last prediction)
  • Email alert (HTML email with full action recommendation)
  • Signal logging (persistent history for tracking)

Usage:
    python sp500_daily.py                     # Run full daily cycle
    python sp500_daily.py --signal-only       # Just print today's signal
    python sp500_daily.py --email-preview     # Generate email HTML without sending
    python sp500_daily.py --data FILE.xlsx    # Use specific data file

Automate with cron (6:30 PM ET weekdays):
    30 18 * * 1-5 cd /path/to/project && python sp500_daily.py >> sp500_logs/daily.log 2>&1

Requires: pandas, scikit-learn, numpy
Optional: yfinance, matplotlib
"""

import os
import sys
import json
import time
import logging
import warnings
import argparse
import smtplib
import platform
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler

# Import our modules
from sp500_advisor import (SignalEngine, engineer_features, extract_regime,
                            format_signal_email, DailyAdvisor, SIGNAL_LOG)

LOG_DIR = "sp500_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "daily_pipeline.log")),
    ]
)
log = logging.getLogger(__name__)

ALERT_CONFIG = os.path.join(LOG_DIR, "alert_config.json")
DAILY_STATE = os.path.join(LOG_DIR, "daily_state.json")


# ════════════════════════════════════════════════
#  DATA FETCHING
# ════════════════════════════════════════════════
def update_data(excel_path):
    """Load existing data and try to fetch new rows from Yahoo Finance."""
    log.info(f"Loading data from {excel_path}")
    df = pd.read_excel(excel_path, sheet_name="Daily Data")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    log.info(f"Loaded {len(df)} rows, latest: {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

    # Try to fetch new data
    try:
        import yfinance as yf
        last_date = df['Date'].iloc[-1]
        ticker = yf.Ticker("^GSPC")
        new = ticker.history(start=last_date + pd.Timedelta(days=1), auto_adjust=False)
        if len(new) > 0:
            new = new.reset_index()
            new.columns = [c.title() if c != 'Date' else 'Date' for c in new.columns]
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in new.columns:
                    new[col] = pd.to_numeric(new[col], errors='coerce')
            new['Date'] = pd.to_datetime(new['Date']).dt.tz_localize(None)
            df = pd.concat([df, new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]], ignore_index=True)
            df = df.drop_duplicates(subset='Date', keep='last').sort_values('Date').reset_index(drop=True)
            log.info(f"Fetched {len(new)} new rows. Total: {len(df)}")
        else:
            log.info("No new data available (market may be closed)")
    except ImportError:
        log.info("yfinance not installed — using existing data only")
    except Exception as e:
        log.warning(f"Data fetch failed: {e} — using existing data")

    return df


# ════════════════════════════════════════════════
#  LOAD PREVIOUS STATE
# ════════════════════════════════════════════════
def load_state():
    if os.path.exists(DAILY_STATE):
        with open(DAILY_STATE) as f:
            return json.load(f)
    return {"last_signal_date": None, "signals": []}


def save_state(state):
    with open(DAILY_STATE, "w") as f:
        json.dump(state, f, indent=2)


# ════════════════════════════════════════════════
#  SCORE YESTERDAY'S PREDICTION
# ════════════════════════════════════════════════
def score_yesterday(state, df):
    """Check if yesterday's signal was correct."""
    signals = state.get("signals", [])
    if not signals:
        return None

    last = signals[-1]
    last_date = last.get("date")
    if not last_date or last.get("scored"):
        return None

    # Find this date in the dataframe
    mask = df['Date'].dt.strftime('%Y-%m-%d') == last_date
    if not mask.any():
        return None

    idx = df[mask].index[0]
    if idx + 1 >= len(df):
        return None  # Next day not available yet

    next_close = float(df.iloc[idx + 1]['Close'])
    this_close = float(df.iloc[idx]['Close'])
    actual_ret = (next_close - this_close) / this_close
    actual_dir = 1 if actual_ret > 0 else 0

    # Was the signal correct?
    action = last.get("action", "")
    bullish = "BUY" in action or "BULLISH" in action
    correct = bullish == (actual_dir == 1)

    last["scored"] = True
    last["actual_ret_pct"] = round(actual_ret * 100, 3)
    last["actual_dir"] = actual_dir
    last["correct"] = correct

    save_state(state)
    log.info(f"Yesterday ({last_date}): Signal was {'✓ CORRECT' if correct else '✗ WRONG'} "
             f"| Action: {action} | Actual: {actual_ret*100:+.2f}%")
    return last


# ════════════════════════════════════════════════
#  FULL DAILY PIPELINE
# ════════════════════════════════════════════════
def run_daily(excel_path, send_email=True):
    """Complete daily pipeline."""
    log.info("=" * 55)
    log.info("  DAILY PIPELINE STARTING")
    log.info("=" * 55)
    t0 = time.time()

    # Load state
    state = load_state()

    # 1. Update data
    df = update_data(excel_path)

    # 2. Engineer features
    log.info("Engineering features...")
    df, feature_cols = engineer_features(df)
    target_cols = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']
    valid = df.dropna(subset=feature_cols + target_cols)
    latest_valid = df.dropna(subset=feature_cols)

    if len(latest_valid) == 0:
        log.error("No valid data after feature engineering")
        return None

    # 3. Score yesterday
    yesterday_score = score_yesterday(state, df)

    # 4. Train models
    log.info("Training models...")
    split = len(valid) - 60
    if split < 1000:
        log.error("Not enough training data")
        return None

    train = valid.iloc[:split]
    recent = valid.iloc[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols].values)

    rf_dir = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_dir.fit(X_train, train['target_dir_1d'].values)

    gb_dir = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                         learning_rate=0.05, min_samples_leaf=20, random_state=42)
    gb_dir.fit(X_train, train['target_dir_1d'].values)

    rf_ret1 = RandomForestRegressor(n_estimators=80, max_depth=10,
                                     min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_ret1.fit(X_train, train['target_ret_1d'].values)

    rf_ret5 = RandomForestRegressor(n_estimators=80, max_depth=10,
                                     min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_ret5.fit(X_train, train['target_ret_5d'].values)

    rf_vol = RandomForestRegressor(n_estimators=80, max_depth=10,
                                    min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_vol.fit(X_train, train['target_vol_5d'].values)

    # 5. Generate predictions on latest data
    log.info("Generating predictions...")
    lr = latest_valid.iloc[-1]
    X_latest = scaler.transform(lr[feature_cols].values.reshape(1, -1))

    rf_prob = float(rf_dir.predict_proba(X_latest)[:, 1][0])
    gb_prob = float(gb_dir.predict_proba(X_latest)[:, 1][0])
    ens_prob = rf_prob * 0.55 + gb_prob * 0.45

    predictions = {
        "rf_prob": rf_prob, "gb_prob": gb_prob, "ensemble_prob": ens_prob,
        "ret1_pct": float(rf_ret1.predict(X_latest)[0]) * 100,
        "ret5_pct": float(rf_ret5.predict(X_latest)[0]) * 100,
        "vol_pct": float(rf_vol.predict(X_latest)[0]) * 100,
        "close": float(lr['Close']),
    }
    regime = extract_regime(lr)

    # Build recent history for confidence calibration
    X_recent = scaler.transform(recent[feature_cols].values)
    rfp = rf_dir.predict_proba(X_recent)[:, 1]
    gbp = gb_dir.predict_proba(X_recent)[:, 1]
    ensp = rfp * 0.55 + gbp * 0.45
    history = []
    for i in range(len(recent)):
        ad = recent['target_dir_1d'].values[i]
        history.append({
            "ensemble_prob": float(ensp[i]),
            "actual_dir": int(ad) if not np.isnan(ad) else None,
        })

    # 6. Generate advisor signal
    log.info("Computing advisor signal...")
    engine = SignalEngine()
    signal = engine.generate(predictions, regime, history)
    signal["date"] = lr['Date'].strftime('%Y-%m-%d')
    signal["close"] = float(lr['Close'])
    signal["predictions"] = {k: round(v, 4) for k, v in predictions.items()}
    signal["regime"] = {k: round(v, 4) for k, v in regime.items()}

    # Add yesterday's result
    if yesterday_score:
        signal["yesterday"] = {
            "date": yesterday_score["date"],
            "action": yesterday_score.get("action", ""),
            "correct": yesterday_score["correct"],
            "actual_ret": yesterday_score["actual_ret_pct"],
        }

    # Track running accuracy
    scored_signals = [s for s in state.get("signals", []) if s.get("scored")]
    if scored_signals:
        n_correct = sum(1 for s in scored_signals if s.get("correct"))
        signal["running_accuracy"] = round(n_correct / len(scored_signals), 3)
        signal["total_scored"] = len(scored_signals)

    # 7. Log signal
    state["last_signal_date"] = signal["date"]
    state["signals"].append({
        "date": signal["date"],
        "action": signal["action"],
        "composite": signal["composite_score"],
        "close": signal["close"],
        "ensemble_prob": predictions["ensemble_prob"],
    })
    # Keep last 500 signals
    state["signals"] = state["signals"][-500:]
    save_state(state)

    # Also log to advisor's signal history
    advisor_log = []
    if os.path.exists(SIGNAL_LOG):
        try:
            with open(SIGNAL_LOG) as f:
                advisor_log = json.load(f)
        except Exception:
            pass
    advisor_log.append(signal)
    with open(SIGNAL_LOG, "w") as f:
        json.dump(advisor_log[-500:], f, indent=2)

    # 8. Print signal
    _print_signal(signal)

    # 9. Send email
    if send_email:
        _send_signal_email(signal)

    elapsed = time.time() - t0
    log.info(f"Daily pipeline complete in {elapsed:.1f}s")
    return signal


# ════════════════════════════════════════════════
#  TERMINAL PRINT
# ════════════════════════════════════════════════
def _print_signal(signal):
    action = signal["action"]
    score = signal["composite_score"]
    preds = signal.get("predictions", {})
    regime = signal.get("regime", {})

    cm = {"STRONG BUY": "\033[92m", "BUY": "\033[92m", "LEAN BULLISH": "\033[32m",
          "HOLD / WAIT": "\033[93m", "LEAN BEARISH": "\033[33m",
          "SELL": "\033[91m", "STRONG SELL": "\033[91m"}
    c = cm.get(action, "")
    r = "\033[0m"

    print(f"\n{'═' * 60}")
    print(f"  S&P 500 ML ADVISOR — {signal.get('date', 'Today')}")
    print(f"{'═' * 60}")

    # Yesterday's result
    y = signal.get("yesterday")
    if y:
        yc = "\033[32m" if y["correct"] else "\033[31m"
        print(f"\n  Yesterday ({y['date']}): {yc}{'✓ CORRECT' if y['correct'] else '✗ WRONG'}{r} "
              f"| {y['action']} | Actual: {y['actual_ret']:+.2f}%")

    # Running accuracy
    ra = signal.get("running_accuracy")
    if ra is not None:
        print(f"  Running accuracy: {ra:.1%} ({signal.get('total_scored', 0)} signals scored)")

    print(f"\n  {c}{'▲' if score > 0 else '▼'} {action}  (Score: {score:+.0f}){r}")
    print(f"  Close: {signal.get('close', 0):,.2f}")
    print(f"  Timeframe: {signal.get('timeframe', 'N/A')}")
    print(f"  Position size: {signal.get('position_size', 'N/A')}")
    print(f"\n  {signal.get('explanation', '')}")

    subs = signal.get("sub_actions", [])
    if subs:
        print(f"\n  Suggested Actions:")
        for i, s in enumerate(subs):
            print(f"  {i+1}. {s['label']}")
            print(f"     {s['detail']}")

    print(f"\n  {'─' * 40}")
    print(f"  Model:  RF={preds.get('rf_prob', 0):.1%}  GB={preds.get('gb_prob', 0):.1%}  "
          f"Ens={preds.get('ensemble_prob', 0):.1%}")
    print(f"  Pred:   1d={preds.get('ret1_pct', 0):+.2f}%  5d={preds.get('ret5_pct', 0):+.2f}%  "
          f"Vol={preds.get('vol_pct', 0):.1f}%")
    print(f"  Regime: RSI={regime.get('rsi', 0):.0f}  MACD={regime.get('macd_hist', 0):.1f}  "
          f"SMA50/200={regime.get('sma_50_200', 0):.3f}")

    scores = signal.get("scores", {})
    print(f"\n  Score Breakdown:")
    for k, v in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(v) / 3)
        sign = "+" if v >= 0 else ""
        c2 = "\033[32m" if v > 0 else "\033[31m" if v < 0 else ""
        print(f"  {k:15s} {c2}{sign}{v:5.0f} {bar}{r}")
    print()


# ════════════════════════════════════════════════
#  EMAIL SENDING
# ════════════════════════════════════════════════
def _load_alert_config():
    if os.path.exists(ALERT_CONFIG):
        with open(ALERT_CONFIG) as f:
            return json.load(f)
    return {}


def _send_signal_email(signal):
    """Send signal email using configured SMTP."""
    config = _load_alert_config()
    email_cfg = config.get("email", {})

    if not email_cfg.get("enabled"):
        log.info("Email not configured. Run with --setup-email to configure.")
        return

    smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port", 587)
    sender = email_cfg.get("sender", "")
    password = email_cfg.get("password", "")
    recipient = email_cfg.get("recipient", sender)

    if not sender or not password:
        log.warning("Email credentials not configured")
        return

    action = signal["action"]
    score = signal["composite_score"]
    date = signal.get("date", "Today")

    # Build comprehensive HTML email
    html_body = _build_email_html(signal)

    subject = f"S&P 500 Signal: {action} ({score:+.0f}) — {date}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    # Plain text fallback
    plain = (f"S&P 500 ML Advisor — {date}\n"
             f"Action: {action} (Score: {score:+.0f})\n"
             f"Close: {signal.get('close', 0):,.2f}\n"
             f"Timeframe: {signal.get('timeframe', 'N/A')}\n\n"
             f"{signal.get('explanation', '')}\n\n"
             f"Not financial advice.")
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [recipient], msg.as_string())
        log.info(f"Email sent to {recipient}")
    except Exception as e:
        log.error(f"Email failed: {e}")


def _build_email_html(signal):
    """Build comprehensive HTML email with the signal."""
    action = signal["action"]
    score = signal["composite_score"]
    preds = signal.get("predictions", {})
    regime = signal.get("regime", {})
    color = signal.get("color", "#f59e0b")
    yesterday = signal.get("yesterday")
    running_acc = signal.get("running_accuracy")

    # Sub-actions HTML
    subs_html = ""
    for i, s in enumerate(signal.get("sub_actions", [])):
        subs_html += f"""
        <div style="padding:10px 0;border-bottom:1px solid #1e293b;">
            <span style="display:inline-block;width:24px;height:24px;border-radius:6px;
                background:{color}20;color:{color};text-align:center;line-height:24px;
                font-weight:800;font-size:12px;margin-right:10px;">{i+1}</span>
            <strong style="color:#f1f5f9;">{s['label']}</strong><br>
            <span style="color:#94a3b8;font-size:12px;margin-left:34px;">{s['detail']}</span>
        </div>"""

    # Yesterday's result
    yesterday_html = ""
    if yesterday:
        ycolor = "#10b981" if yesterday["correct"] else "#ef4444"
        ystatus = "✓ CORRECT" if yesterday["correct"] else "✗ WRONG"
        yesterday_html = f"""
        <div style="background:#1e293b;border-radius:8px;padding:12px;margin:12px 0;">
            <span style="color:#64748b;font-size:11px;text-transform:uppercase;">Yesterday's Result</span><br>
            <span style="color:{ycolor};font-weight:700;">{ystatus}</span>
            <span style="color:#94a3b8;"> — {yesterday['action']} | Actual: {yesterday['actual_ret']:+.2f}%</span>
        </div>"""

    # Accuracy tracker
    acc_html = ""
    if running_acc is not None:
        acc_color = "#10b981" if running_acc > 0.54 else "#f59e0b" if running_acc > 0.50 else "#ef4444"
        acc_html = f"""
        <div style="color:{acc_color};font-size:12px;margin-top:4px;">
            Running accuracy: {running_acc:.1%} ({signal.get('total_scored', 0)} signals scored)
        </div>"""

    # Score bars
    scores = signal.get("scores", {})
    bars_html = ""
    for k, v in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True):
        scolor = "#10b981" if v > 0 else "#ef4444" if v < 0 else "#64748b"
        pct = min(abs(v) / 40 * 100, 100)
        side = "left:50%;" if v >= 0 else f"left:{50-pct/2}%;"
        bars_html += f"""
        <div style="margin-bottom:4px;">
            <div style="display:flex;justify-content:space-between;font-size:11px;">
                <span style="color:#94a3b8;">{k}</span>
                <span style="color:{scolor};font-weight:600;">{'+' if v>0 else ''}{v:.0f}</span>
            </div>
            <div style="height:4px;background:#1e293b;border-radius:2px;position:relative;">
                <div style="position:absolute;top:0;height:100%;border-radius:2px;{side}
                    width:{pct/2}%;background:{scolor};"></div>
            </div>
        </div>"""

    return f"""
<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#020617;">
<div style="font-family:'SF Mono',Consolas,monospace;max-width:600px;margin:0 auto;
    background:#0f172a;color:#e2e8f0;padding:24px;border-radius:12px;">

    <div style="text-align:center;margin-bottom:16px;">
        <div style="font-size:11px;color:#64748b;letter-spacing:2px;text-transform:uppercase;">
            S&amp;P 500 ML Advisor</div>
        <div style="font-size:10px;color:#475569;margin-top:2px;">
            {signal.get('date','Today')} · Close: {signal.get('close',0):,.2f}</div>
    </div>

    {yesterday_html}

    <div style="background:{color}10;border:2px solid {color}30;border-radius:12px;
        padding:20px;margin:12px 0;text-align:center;">
        <div style="font-size:28px;font-weight:800;color:{color};">
            {'▲' if score > 0 else '▼'} {action}
        </div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">
            Composite Score: {score:+.0f} / 50
        </div>
        {acc_html}
    </div>

    <div style="margin:16px 0;">
        <div style="color:#cbd5e1;font-size:13px;line-height:1.6;">
            {signal.get('explanation', '')}
        </div>
        <div style="margin-top:8px;">
            <span style="color:#64748b;font-size:11px;">Timeframe:</span>
            <span style="color:#f1f5f9;font-size:13px;font-weight:600;">
                {signal.get('timeframe', 'N/A')}</span>
        </div>
        <div style="margin-top:4px;">
            <span style="color:#64748b;font-size:11px;">Position Size:</span>
            <span style="color:#f1f5f9;font-size:13px;font-weight:600;">
                {signal.get('position_size', 'N/A')}</span>
        </div>
    </div>

    <div style="border-top:1px solid #1e293b;padding-top:12px;margin-top:12px;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;
            margin-bottom:8px;font-weight:700;">Suggested Actions</div>
        {subs_html}
    </div>

    <div style="border-top:1px solid #1e293b;padding-top:12px;margin-top:16px;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;
            margin-bottom:8px;font-weight:700;">Score Breakdown</div>
        {bars_html}
    </div>

    <div style="border-top:1px solid #1e293b;padding-top:12px;margin-top:16px;
        font-size:12px;color:#64748b;">
        <div>RF: {preds.get('rf_prob',0):.1%} · GB: {preds.get('gb_prob',0):.1%} ·
            Ensemble: {preds.get('ensemble_prob',0):.1%}</div>
        <div>Pred 1d: {preds.get('ret1_pct',0):+.2f}% · 5d: {preds.get('ret5_pct',0):+.2f}% ·
            Vol: {preds.get('vol_pct',0):.1f}%</div>
        <div>RSI: {regime.get('rsi',0):.0f} · MACD: {regime.get('macd_hist',0):.1f} ·
            SMA 50/200: {regime.get('sma_50_200',0):.3f}</div>
    </div>

    <div style="text-align:center;margin-top:20px;padding-top:12px;border-top:1px solid #1e293b;">
        <div style="color:#ef4444;font-size:10px;font-style:italic;">
            For educational/research purposes only — not financial advice.
        </div>
    </div>
</div>
</body></html>"""


# ════════════════════════════════════════════════
#  EMAIL SETUP
# ════════════════════════════════════════════════
def setup_email():
    """Interactive email configuration."""
    print("\n  ═══ Email Alert Setup ═══")
    print("  For Gmail: create an App Password at https://myaccount.google.com/apppasswords")
    print()

    config = _load_alert_config()

    sender = input("  Sender email: ").strip()
    password = input("  App password: ").strip()
    recipient = input(f"  Recipient (Enter for {sender}): ").strip() or sender

    config["email"] = {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender": sender,
        "password": password,
        "recipient": recipient,
    }

    with open(ALERT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  ✓ Config saved to {ALERT_CONFIG}")
    print("  Test with: python sp500_daily.py --email-preview")


# ════════════════════════════════════════════════
#  DESKTOP NOTIFICATION (cross-platform)
# ════════════════════════════════════════════════
def _send_desktop_notification(signal):
    """Send OS-level notification."""
    action = signal["action"]
    score = signal["composite_score"]
    title = f"S&P 500: {action} ({score:+.0f})"
    body = signal.get("timeframe", "")

    system = platform.system()
    try:
        if system == "Darwin":
            os.system(f'osascript -e \'display notification "{body}" with title "{title}"\'')
        elif system == "Linux":
            os.system(f'notify-send "{title}" "{body}"')
        elif system == "Windows":
            os.system(f'powershell -command "New-BurntToastNotification -Text \'{title}\',\'{body}\'"')
    except Exception:
        pass  # Non-critical


# ════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════
def find_data():
    for p in ["SP500_Analysis.xlsx", os.path.expanduser("~/Downloads/SP500_Analysis.xlsx")]:
        if os.path.exists(p):
            return p
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 Daily Signal Pipeline")
    parser.add_argument("--signal-only", action="store_true", help="Print signal without email")
    parser.add_argument("--email-preview", action="store_true", help="Generate email HTML preview")
    parser.add_argument("--setup-email", action="store_true", help="Configure email alerts")
    parser.add_argument("--data", type=str, default=None, help="Path to Excel data file")
    args = parser.parse_args()

    if args.setup_email:
        setup_email()
        sys.exit(0)

    data_path = args.data or find_data()
    if not data_path:
        print("  SP500_Analysis.xlsx not found!")
        print("  Use: python sp500_daily.py --data /path/to/file.xlsx")
        sys.exit(1)

    if args.email_preview:
        signal = run_daily(data_path, send_email=False)
        if signal:
            html = _build_email_html(signal)
            preview = os.path.join(LOG_DIR, "email_preview.html")
            with open(preview, "w") as f:
                f.write(html)
            print(f"\n  Email preview: {preview}")
            print(f"  Open: file://{os.path.abspath(preview)}")
    elif args.signal_only:
        run_daily(data_path, send_email=False)
    else:
        signal = run_daily(data_path, send_email=True)
        if signal:
            _send_desktop_notification(signal)
