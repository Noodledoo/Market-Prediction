#!/usr/bin/env python3
"""
sp500_advisor.py — Market Action Advisor Engine

Generates specific buy / sell / hold / wait recommendations by combining:
  • ML ensemble predictions (RF + GB direction probabilities)
  • Regime analysis (trend, momentum, mean-reversion, volatility)
  • Risk management (position sizing, stop-loss levels, wait times)
  • Confidence calibration (recent accuracy, model agreement)

Plugs into sp500_live_tracker.py for daily automated alerts.

Usage:
    python sp500_advisor.py                # Generate today's signal (needs data)
    python sp500_advisor.py --backtest     # Backtest signal-following strategy
    python sp500_advisor.py --report       # Print recent signal history
    python sp500_advisor.py --email-test   # Test email with today's signal

Requires: pandas, scikit-learn, numpy
Optional: matplotlib (for report charts), pygame (for dashboard)
"""

import os
import sys
import json
import time
import math
import warnings
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

LOG_DIR = "sp500_logs"
MODEL_DIR = "sp500_models"
SIGNAL_LOG = os.path.join(LOG_DIR, "signal_history.json")
ADVISOR_MODELS = os.path.join(MODEL_DIR, "advisor_models.pkl")

for d in [LOG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════
#  SIGNAL ENGINE — The core recommendation logic
# ════════════════════════════════════════════════
class SignalEngine:
    """
    Combines model predictions + technical regime into actionable signals.
    
    Output actions:
      STRONG BUY   — Full position, enter today
      BUY          — Partial to full, enter within 1-2 days
      LEAN BULLISH — Partial position, consider limit orders below
      HOLD / WAIT  — No action, reassess in N days
      LEAN BEARISH — Trim positions, tighten stops
      SELL         — Reduce to defensive, go to cash
      STRONG SELL  — Maximum defensiveness, hedge
    """

    # Composite score thresholds → actions
    ZONES = [
        (35,   "STRONG BUY",   "#10b981"),
        (20,   "BUY",          "#34d399"),
        (8,    "LEAN BULLISH", "#6ee7b4"),
        (-8,   "HOLD / WAIT",  "#f59e0b"),
        (-20,  "LEAN BEARISH", "#f97316"),
        (-35,  "SELL",         "#ef4444"),
        (-999, "STRONG SELL",  "#dc2626"),
    ]

    # Component weights (must sum to ~1.0)
    WEIGHTS = {
        "direction":    0.30,  # ML ensemble probability
        "trend":        0.18,  # SMA regime
        "momentum":     0.12,  # MACD + returns
        "meanrev":      0.10,  # RSI + Bollinger
        "volatility":   0.08,  # Vol regime
        "agreement":    0.08,  # Model agreement
        "return_pred":  0.08,  # Return forecast
        "confidence":   0.06,  # Recent accuracy
    }

    def generate(self, predictions, regime, history=None):
        """
        Generate a full signal from model predictions and regime data.
        
        Args:
            predictions: dict with keys:
                rf_prob, gb_prob, ensemble_prob, ret1_pct, ret5_pct, vol_pct
            regime: dict with keys:
                sma_50_200, close_sma200, close_sma50, rsi, vol_20d,
                macd_hist, bb_pct, ret_5d, ret_20d
            history: list of recent signal dicts with 'ensemble_prob' and 'actual_dir'
                
        Returns:
            dict: Full signal with action, score, explanation, sub-actions, etc.
        """
        p = predictions
        r = regime
        
        scores = self._compute_scores(p, r, history)
        composite = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        composite = max(-50, min(50, composite))

        # Map to action zone
        action_name, color = "HOLD / WAIT", "#f59e0b"
        for threshold, name, col in self.ZONES:
            if composite >= threshold:
                action_name, color = name, col
                break

        # Generate context-aware details
        details = self._generate_details(
            action_name, composite, p, r, scores, history
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "composite_score": round(composite, 1),
            "action": action_name,
            "color": color,
            "scores": {k: round(v, 1) for k, v in scores.items()},
            **details,
        }

    def _compute_scores(self, p, r, history):
        """Compute individual signal component scores (-100 to +100 scale)."""
        scores = {}

        # 1. Direction ensemble
        ens = p.get("ensemble_prob", 0.5)
        scores["direction"] = (ens - 0.5) * 200

        # 2. Model agreement
        rf = p.get("rf_prob", 0.5)
        gb = p.get("gb_prob", 0.5)
        agree = 1 - abs(rf - gb)
        if (rf > 0.5) != (gb > 0.5):
            scores["agreement"] = -15  # Models disagree
        elif agree > 0.9:
            scores["agreement"] = 15
        elif agree > 0.8:
            scores["agreement"] = 8
        else:
            scores["agreement"] = 3

        # 3. Trend regime (SMA 50/200 + price vs SMA200)
        sma_ratio = r.get("sma_50_200", 1.0)
        close_200 = r.get("close_sma200", 1.0)
        t = 0
        if sma_ratio > 1.03:    t += 25
        elif sma_ratio > 1.01:  t += 15
        elif sma_ratio > 1.0:   t += 5
        elif sma_ratio > 0.99:  t -= 5
        elif sma_ratio > 0.97:  t -= 15
        else:                    t -= 25
        if close_200 > 1.0:     t += 10
        else:                    t -= 10
        scores["trend"] = max(-40, min(40, t))

        # 4. Momentum (MACD hist + recent returns)
        macd_h = r.get("macd_hist", 0)
        ret5 = r.get("ret_5d", 0)
        ret20 = r.get("ret_20d", 0)
        m = 0
        if macd_h > 3:      m += 15
        elif macd_h > 0:    m += 8
        elif macd_h > -3:   m -= 8
        else:                m -= 15
        if ret5 > 2:        m += 10
        elif ret5 > 0.5:    m += 3
        elif ret5 < -3:     m -= 15
        elif ret5 < -1:     m -= 5
        if ret20 > 5:       m += 5
        elif ret20 < -5:    m -= 15
        elif ret20 < -3:    m -= 8
        scores["momentum"] = max(-35, min(35, m))

        # 5. Mean reversion (RSI + BB)
        rsi = r.get("rsi", 50)
        bb = r.get("bb_pct", 0.5)
        mr = 0
        if rsi > 75:     mr -= 25
        elif rsi > 70:   mr -= 15
        elif rsi > 60:   mr -= 3
        elif rsi < 25:   mr += 25
        elif rsi < 30:   mr += 15
        elif rsi < 40:   mr += 8
        if bb > 0.95:    mr -= 12
        elif bb > 0.8:   mr -= 5
        elif bb < 0.05:  mr += 12
        elif bb < 0.2:   mr += 5
        scores["meanrev"] = max(-30, min(30, mr))

        # 6. Volatility regime
        vol = p.get("vol_pct", 15)
        v = 0
        if vol > 30:     v -= 25
        elif vol > 25:   v -= 18
        elif vol > 20:   v -= 10
        elif vol > 15:   v -= 3
        elif vol < 10:   v += 12
        elif vol < 12:   v += 8
        elif vol < 14:   v += 3
        scores["volatility"] = v

        # 7. Return predictions
        ret5_pred = p.get("ret5_pct", 0)
        rp = 0
        if ret5_pred > 1.0:     rp += 20
        elif ret5_pred > 0.5:   rp += 12
        elif ret5_pred > 0.2:   rp += 5
        elif ret5_pred < -1.0:  rp -= 20
        elif ret5_pred < -0.5:  rp -= 12
        elif ret5_pred < -0.2:  rp -= 5
        scores["return_pred"] = rp

        # 8. Recent accuracy → confidence
        if history:
            scored = [h for h in history if h.get("actual_dir") is not None]
            if len(scored) >= 10:
                last20 = scored[-20:]
                correct = sum(1 for h in last20
                              if (h.get("ensemble_prob", 0.5) > 0.5) == (h["actual_dir"] == 1))
                acc = correct / len(last20)
                scores["confidence"] = (acc - 0.5) * 80
            else:
                scores["confidence"] = 0
        else:
            scores["confidence"] = 0

        return scores

    def _generate_details(self, action, composite, p, r, scores, history):
        """Generate human-readable explanation and specific sub-actions."""
        vol = p.get("vol_pct", 15)
        close = p.get("close", 0)
        rsi = r.get("rsi", 50)
        ret20 = r.get("ret_20d", 0)
        ens = p.get("ensemble_prob", 0.5)

        details = {}

        # ─── Timeframe ───
        if "BUY" in action or "STRONG BUY" in action:
            details["timeframe"] = "Enter position today" if "STRONG" in action else "Enter within 1-3 days"
        elif "BULLISH" in action:
            wait = "1-3 days" if vol < 15 else "3-5 days"
            details["timeframe"] = f"Consider buying within {wait}"
        elif "WAIT" in action:
            wait = "2-4 weeks" if vol > 20 else "1-2 weeks" if vol > 15 else "3-7 days"
            details["timeframe"] = f"Reassess in {wait}"
        elif "BEARISH" in action:
            wait = "3-6 weeks" if vol > 20 else "2-4 weeks" if vol > 15 else "1-3 weeks"
            details["timeframe"] = f"Wait {wait} before buying"
        else:  # SELL
            wait = "6-10 weeks" if vol > 25 else "4-8 weeks"
            details["timeframe"] = f"Defensive position — wait {wait} before buying"

        # ─── Explanation ───
        bullish_factors = []
        bearish_factors = []

        if scores.get("trend", 0) > 10:
            bullish_factors.append("long-term trend is up (golden cross)")
        elif scores.get("trend", 0) < -10:
            bearish_factors.append("long-term trend is down (death cross)")

        if scores.get("momentum", 0) > 8:
            bullish_factors.append("momentum is positive")
        elif scores.get("momentum", 0) < -8:
            bearish_factors.append("momentum is fading")

        if rsi < 35:
            bullish_factors.append("RSI is oversold (mean reversion opportunity)")
        elif rsi > 65:
            bearish_factors.append("RSI is overbought (pullback likely)")

        if ens > 0.58:
            bullish_factors.append(f"ensemble model gives {ens:.0%} probability of up day")
        elif ens < 0.42:
            bearish_factors.append(f"ensemble model gives only {ens:.0%} probability of up day")

        if vol < 13:
            bullish_factors.append("volatility is low and calm")
        elif vol > 22:
            bearish_factors.append("volatility is elevated — increased risk")

        if ret20 < -5:
            bearish_factors.append(f"the market has dropped {abs(ret20):.1f}% over 20 days")
        
        if scores.get("agreement", 0) > 10:
            bullish_factors.append("both RF and GB models agree")
        elif scores.get("agreement", 0) < -10:
            bearish_factors.append("RF and GB models disagree (uncertain)")

        if "BUY" in action or "BULLISH" in action:
            factors_str = ", ".join(bullish_factors) if bullish_factors else "mild bullish tilt"
            caveat = ""
            if bearish_factors:
                caveat = f" However, {bearish_factors[0]}."
            details["explanation"] = f"Bullish signal — {factors_str}.{caveat}"
        elif "WAIT" in action:
            details["explanation"] = (
                f"Mixed signals with no strong edge. "
                + (f"Bullish: {', '.join(bullish_factors[:2])}. " if bullish_factors else "")
                + (f"Bearish: {', '.join(bearish_factors[:2])}. " if bearish_factors else "")
                + "Patience is the smart move here."
            )
        else:
            factors_str = ", ".join(bearish_factors) if bearish_factors else "mild bearish tilt"
            caveat = ""
            if bullish_factors:
                caveat = f" On the positive side, {bullish_factors[0]}."
            details["explanation"] = f"Caution warranted — {factors_str}.{caveat}"

        # ─── Sub-actions ───
        sub = []
        if close > 0:
            stop_pct = 0.02 if vol < 15 else 0.03 if vol < 20 else 0.05
            stop_price = close * (1 - stop_pct)
            limit_buy = close * 0.995
            resistance = close * 1.015
            support = close * 0.985

        if "STRONG BUY" in action:
            sub = [
                {"label": "Full position at market open",
                 "detail": "Deploy full intended allocation — strong confluence of signals."},
                {"label": "Set stop-loss",
                 "detail": f"Place stop at {stop_price:.0f} (-{stop_pct*100:.0f}%) to limit downside." if close > 0 else "Place stop 2-3% below entry."},
                {"label": "Target: hold for 1-4 weeks",
                 "detail": "Let the trend work. Don't sell on the first red day."},
            ]
        elif "BUY" in action and "STRONG" not in action:
            sub = [
                {"label": "Enter 70-100% position",
                 "detail": "Good entry conditions — most of your intended allocation."},
                {"label": "Scale in over 1-2 days",
                 "detail": "Split entry to reduce single-day timing risk."},
                {"label": "Set stop-loss",
                 "detail": f"Stop at {stop_price:.0f} (-{stop_pct*100:.0f}%)." if close > 0 else "Stop 2-3% below entry."},
            ]
        elif "LEAN BULLISH" in action:
            sub = [
                {"label": "Partial position (50-70%)",
                 "detail": "Enter some now, keep powder dry for a potential dip."},
                {"label": f"Limit order at {limit_buy:.0f}" if close > 0 else "Set limit order 0.5% below",
                 "detail": "Try to catch a small pullback for better entry."},
                {"label": "Wait 1-3 days for confirmation",
                 "detail": "See if the next session confirms the bullish tilt before going full size."},
            ]
        elif "WAIT" in action:
            sub = [
                {"label": "Stay in cash / hold existing",
                 "detail": "No compelling edge — don't force trades."},
                {"label": "Keep existing winners",
                 "detail": "Don't sell profitable positions, but don't add risk either."},
                {"label": f"Set alerts: above {resistance:.0f} or below {support:.0f}" if close > 0 else "Set alerts at key levels",
                 "detail": "A breakout from this range will clarify direction."},
            ]
        elif "LEAN BEARISH" in action:
            sub = [
                {"label": "Trim positions 20-30%",
                 "detail": "Reduce exposure while keeping core holdings."},
                {"label": f"Tighten stops to {stop_price:.0f}" if close > 0 else "Tighten stop-losses",
                 "detail": "Protect gains — don't let a small loss become a big one."},
                {"label": f"Wait for RSI < 35 to re-enter",
                 "detail": "That level has historically marked good buying opportunities."},
            ]
        elif "SELL" in action and "STRONG" not in action:
            sub = [
                {"label": "Reduce to 30-50% invested",
                 "detail": "Significant drawdown risk ahead — capital preservation first."},
                {"label": "Sell into any rallies",
                 "detail": "Use bounces as exit opportunities, not buying opportunities."},
                {"label": "Consider defensive positions",
                 "detail": "Treasury bonds, utilities, or cash. Hedge with protective puts if holding."},
            ]
        else:  # STRONG SELL
            sub = [
                {"label": "Move to 70%+ cash",
                 "detail": "Multiple danger signals — protect capital above all."},
                {"label": "Sell immediately at market open",
                 "detail": "Don't wait for a bounce — conditions are deteriorating."},
                {"label": "Consider hedges",
                 "detail": "Inverse ETFs or protective puts on remaining positions."},
                {"label": f"Re-evaluate when RSI drops below 30 and vol < 20%",
                 "detail": "Those conditions together have historically marked bottoms."},
            ]

        details["sub_actions"] = sub

        # ─── Position sizing suggestion ───
        if "BUY" in action or "BULLISH" in action:
            if vol < 12:
                details["position_size"] = "100% of intended"
            elif vol < 16:
                details["position_size"] = "75-100% of intended"
            elif vol < 20:
                details["position_size"] = "50-75% of intended"
            else:
                details["position_size"] = "25-50% of intended (high vol — size down)"
        elif "BEARISH" in action or "SELL" in action:
            details["position_size"] = "Reduce existing exposure"
        else:
            details["position_size"] = "No new positions"

        return details


# ════════════════════════════════════════════════
#  FEATURE ENGINEERING (shared with other modules)
# ════════════════════════════════════════════════
def engineer_features(df):
    """Full feature engineering pipeline. Returns (df, feature_cols, regime_cols)."""
    df = df.copy()
    for n in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{n}d'] = df['Close'].pct_change(n)
    for n in [5, 10, 20, 50, 200]:
        df[f'sma_{n}'] = df['Close'].rolling(n).mean()
        df[f'close_sma_{n}_ratio'] = df['Close'] / df[f'sma_{n}']
    df['sma_5_20_ratio'] = df['sma_5'] / df['sma_20']
    df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
    df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    bb_m = df['Close'].rolling(20).mean()
    bb_s = df['Close'].rolling(20).std()
    df['bb_pct'] = (df['Close'] - (bb_m - 2 * bb_s)) / (4 * bb_s)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(14).mean() / df['Close']
    for n in [5, 10, 20]:
        df[f'vol_{n}d'] = df['ret_1d'].rolling(n).std() * np.sqrt(252)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['hl_range'] = hl / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
    df['dow'] = df['Date'].dt.dayofweek

    # Targets
    df['target_ret_1d'] = df['Close'].shift(-1) / df['Close'] - 1
    df['target_dir_1d'] = (df['target_ret_1d'] > 0).astype(int)
    df['target_ret_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    df['target_vol_5d'] = df['ret_1d'].shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)

    feature_cols = [
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
        'close_sma_5_ratio', 'close_sma_10_ratio', 'close_sma_20_ratio',
        'close_sma_50_ratio', 'close_sma_200_ratio', 'sma_50_200_ratio',
        'macd', 'macd_signal', 'macd_hist', 'rsi_14', 'bb_pct', 'atr_pct',
        'vol_5d', 'vol_10d', 'vol_20d', 'vol_ratio', 'hl_range', 'gap', 'dow',
    ]

    return df, feature_cols


def extract_regime(row):
    """Extract regime dict from a dataframe row."""
    return {
        "sma_50_200": float(row.get('sma_50_200_ratio', 1)),
        "close_sma200": float(row.get('close_sma_200_ratio', 1)),
        "close_sma50": float(row.get('close_sma_50_ratio', 1)),
        "rsi": float(row.get('rsi_14', 50)),
        "vol_20d": float(row.get('vol_20d', 0.15)) * 100,
        "macd_hist": float(row.get('macd_hist', 0)),
        "bb_pct": float(row.get('bb_pct', 0.5)),
        "ret_5d": float(row.get('ret_5d', 0)) * 100,
        "ret_20d": float(row.get('ret_20d', 0)) * 100,
    }


# ════════════════════════════════════════════════
#  DAILY SIGNAL GENERATION
# ════════════════════════════════════════════════
class DailyAdvisor:
    """Generate and store daily trading signals."""

    def __init__(self):
        self.engine = SignalEngine()
        self.signal_log = self._load_log()

    def _load_log(self):
        if os.path.exists(SIGNAL_LOG):
            with open(SIGNAL_LOG) as f:
                return json.load(f)
        return []

    def _save_log(self):
        with open(SIGNAL_LOG, "w") as f:
            json.dump(self.signal_log[-500:], f, indent=2)  # Keep last 500

    def generate_signal(self, excel_path, progress_cb=None):
        """Full pipeline: load data → train → predict → generate signal."""
        if progress_cb:
            progress_cb("Loading data...", 0.0)

        df = pd.read_excel(excel_path, sheet_name="Daily Data")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        if progress_cb:
            progress_cb("Engineering features...", 0.1)

        df, feature_cols = engineer_features(df)
        target_cols = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']
        valid = df.dropna(subset=feature_cols + target_cols)

        # Train on everything except last 60 days
        split = len(valid) - 60
        train = valid.iloc[:split]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols].values)

        if progress_cb:
            progress_cb("Training models...", 0.2)

        rf_dir = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1)
        rf_dir.fit(X_train, train['target_dir_1d'].values)

        gb_dir = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42)
        gb_dir.fit(X_train, train['target_dir_1d'].values)

        if progress_cb:
            progress_cb("Training regressors...", 0.5)

        rf_ret5 = RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1)
        rf_ret5.fit(X_train, train['target_ret_5d'].values)

        rf_vol = RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1)
        rf_vol.fit(X_train, train['target_vol_5d'].values)

        rf_ret1 = RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1)
        rf_ret1.fit(X_train, train['target_ret_1d'].values)

        if progress_cb:
            progress_cb("Generating signal...", 0.7)

        # Get latest row with features
        latest_valid = df.dropna(subset=feature_cols)
        lr = latest_valid.iloc[-1]
        X_latest = scaler.transform(lr[feature_cols].values.reshape(1, -1))

        rf_prob = float(rf_dir.predict_proba(X_latest)[:, 1][0])
        gb_prob = float(gb_dir.predict_proba(X_latest)[:, 1][0])
        ens_prob = rf_prob * 0.55 + gb_prob * 0.45

        predictions = {
            "rf_prob": rf_prob,
            "gb_prob": gb_prob,
            "ensemble_prob": ens_prob,
            "ret1_pct": float(rf_ret1.predict(X_latest)[0]) * 100,
            "ret5_pct": float(rf_ret5.predict(X_latest)[0]) * 100,
            "vol_pct": float(rf_vol.predict(X_latest)[0]) * 100,
            "close": float(lr['Close']),
        }

        regime = extract_regime(lr)

        # Build recent history for confidence calibration
        recent = valid.iloc[split:]
        X_recent = scaler.transform(recent[feature_cols].values)
        rfp = rf_dir.predict_proba(X_recent)[:, 1]
        gbp = gb_dir.predict_proba(X_recent)[:, 1]
        ensp = rfp * 0.55 + gbp * 0.45
        history = []
        for i in range(len(recent)):
            ad = int(recent['target_dir_1d'].values[i])
            if np.isnan(recent['target_dir_1d'].values[i]):
                ad = None
            history.append({
                "ensemble_prob": float(ensp[i]),
                "actual_dir": ad,
            })

        if progress_cb:
            progress_cb("Computing signal...", 0.9)

        signal = self.engine.generate(predictions, regime, history)
        signal["date"] = lr['Date'].strftime('%Y-%m-%d')
        signal["close"] = float(lr['Close'])
        signal["predictions"] = {k: round(v, 4) for k, v in predictions.items()}
        signal["regime"] = {k: round(v, 4) for k, v in regime.items()}

        # Save to log
        self.signal_log.append(signal)
        self._save_log()

        # Save models for quick re-use
        with open(ADVISOR_MODELS, "wb") as f:
            pickle.dump({
                "scaler": scaler, "rf_dir": rf_dir, "gb_dir": gb_dir,
                "rf_ret1": rf_ret1, "rf_ret5": rf_ret5, "rf_vol": rf_vol,
                "feature_cols": feature_cols, "trained": datetime.now().isoformat(),
            }, f)

        if progress_cb:
            progress_cb("Signal generated!", 1.0)

        return signal

    def get_recent_signals(self, n=10):
        return self.signal_log[-n:]


# ════════════════════════════════════════════════
#  BACKTEST THE SIGNAL STRATEGY
# ════════════════════════════════════════════════
def backtest_signals(excel_path, progress_cb=None):
    """
    Backtest: what if you had followed the advisor's signals historically?
    Uses walk-forward: train on expanding window, generate signal, act on it.
    """
    if progress_cb:
        progress_cb("Loading data...", 0.0)

    df = pd.read_excel(excel_path, sheet_name="Daily Data")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df, feature_cols = engineer_features(df)
    target_cols = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']
    valid = df.dropna(subset=feature_cols + target_cols).copy()

    # Walk-forward: retrain every 126 days (semi-annually), test on each day
    min_train = 10 * 252  # 10 years
    retrain_every = 126
    n = len(valid)

    if n < min_train + 252:
        print("Not enough data for walk-forward backtest")
        return None

    engine = SignalEngine()
    results = []
    last_train = 0

    scaler = None
    rf_dir = None
    gb_dir = None
    rf_ret5 = None
    rf_vol = None
    rf_ret1 = None

    total_test = n - min_train
    for i in range(min_train, n):
        day_idx = i - min_train

        # Retrain periodically
        if i == min_train or (i - last_train) >= retrain_every:
            if progress_cb:
                progress_cb(f"Training fold ({day_idx}/{total_test})", day_idx / total_test * 0.9)

            train = valid.iloc[:i]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train[feature_cols].values)

            rf_dir = RandomForestClassifier(
                n_estimators=60, max_depth=8, min_samples_leaf=30,
                random_state=42, n_jobs=-1)
            rf_dir.fit(X_tr, train['target_dir_1d'].values)

            gb_dir = GradientBoostingClassifier(
                n_estimators=60, max_depth=4, learning_rate=0.05,
                min_samples_leaf=30, random_state=42)
            gb_dir.fit(X_tr, train['target_dir_1d'].values)

            rf_ret5 = RandomForestRegressor(n_estimators=40, max_depth=8,
                min_samples_leaf=30, random_state=42, n_jobs=-1)
            rf_ret5.fit(X_tr, train['target_ret_5d'].values)

            rf_vol = RandomForestRegressor(n_estimators=40, max_depth=8,
                min_samples_leaf=30, random_state=42, n_jobs=-1)
            rf_vol.fit(X_tr, train['target_vol_5d'].values)

            rf_ret1 = RandomForestRegressor(n_estimators=40, max_depth=8,
                min_samples_leaf=30, random_state=42, n_jobs=-1)
            rf_ret1.fit(X_tr, train['target_ret_1d'].values)

            last_train = i

        # Predict for day i
        row = valid.iloc[i]
        X_day = scaler.transform(row[feature_cols].values.reshape(1, -1))

        rfp = float(rf_dir.predict_proba(X_day)[:, 1][0])
        gbp = float(gb_dir.predict_proba(X_day)[:, 1][0])

        predictions = {
            "rf_prob": rfp, "gb_prob": gbp,
            "ensemble_prob": rfp * 0.55 + gbp * 0.45,
            "ret1_pct": float(rf_ret1.predict(X_day)[0]) * 100,
            "ret5_pct": float(rf_ret5.predict(X_day)[0]) * 100,
            "vol_pct": float(rf_vol.predict(X_day)[0]) * 100,
            "close": float(row['Close']),
        }
        regime = extract_regime(row)

        # Generate signal (no history for speed)
        signal = engine.generate(predictions, regime, None)

        actual_ret = float(row['target_ret_1d'])
        actual_dir = int(row['target_dir_1d'])

        results.append({
            "date": row['Date'].strftime('%Y-%m-%d'),
            "close": float(row['Close']),
            "action": signal["action"],
            "composite": signal["composite_score"],
            "ensemble_prob": predictions["ensemble_prob"],
            "actual_ret": actual_ret,
            "actual_dir": actual_dir,
        })

    if progress_cb:
        progress_cb("Computing strategy performance...", 0.95)

    # ─── Compute performance of different strategies based on signals ───
    df_results = pd.DataFrame(results)

    strategies = {}

    # Strategy 1: Binary — long if any BUY signal, cash otherwise
    def strat_binary(row):
        return 1.0 if "BUY" in row["action"] or "BULLISH" in row["action"] else 0.0

    # Strategy 2: Scaled — position size based on composite
    def strat_scaled(row):
        c = row["composite"]
        if c > 20: return 1.0
        elif c > 8: return 0.7
        elif c > -8: return 0.3
        elif c > -20: return 0.0
        else: return 0.0  # Could short, but keeping it long-only

    # Strategy 3: Conservative — only act on strong signals
    def strat_conservative(row):
        return 1.0 if "STRONG BUY" in row["action"] or "BUY" == row["action"] else (
            0.5 if "BULLISH" in row["action"] else 0.0)

    for name, fn in [("Signal Binary", strat_binary),
                     ("Signal Scaled", strat_scaled),
                     ("Signal Conservative", strat_conservative)]:
        equity = [1.0]
        positions = []
        for _, r in df_results.iterrows():
            pos = fn(r)
            positions.append(pos)
            equity.append(equity[-1] * (1 + r["actual_ret"] * pos))
        strategies[name] = {"equity": equity, "positions": positions}

    # Buy and hold
    bh_equity = [1.0]
    for _, r in df_results.iterrows():
        bh_equity.append(bh_equity[-1] * (1 + r["actual_ret"]))
    strategies["Buy & Hold"] = {"equity": bh_equity, "positions": [1.0] * len(df_results)}

    # ─── Metrics ───
    def calc_metrics(eq, positions, returns):
        eq = np.array(eq)
        rets = np.diff(np.log(np.clip(eq, 1e-10, None)))
        n_years = len(rets) / 252
        total = (eq[-1] / eq[0] - 1) * 100
        cagr = ((eq[-1] / eq[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
        vol = np.std(rets) * np.sqrt(252)
        sharpe = (np.mean(rets) * 252 - 0.02) / max(vol, 1e-10)
        peak = np.maximum.accumulate(eq)
        dd = np.min((eq - peak) / peak) * 100

        active = [(p, r) for p, r in zip(positions, returns) if abs(p) > 0.01]
        wins = sum(1 for p, r in active if p * r > 0) if active else 0
        wr = wins / len(active) if active else 0
        exposure = sum(1 for p in positions if p > 0.01) / max(len(positions), 1)

        return {
            "total_return": round(total, 1),
            "cagr": round(cagr, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(dd, 1),
            "volatility": round(vol * 100, 1),
            "win_rate": round(wr, 3),
            "exposure": round(exposure, 3),
            "n_years": round(n_years, 1),
        }

    actual_rets = df_results["actual_ret"].values
    metrics = {}
    for name, s in strategies.items():
        metrics[name] = calc_metrics(s["equity"], s["positions"], actual_rets)

    # Signal distribution
    action_counts = df_results["action"].value_counts().to_dict()

    # Accuracy by signal type
    accuracy_by_signal = {}
    for action in df_results["action"].unique():
        mask = df_results["action"] == action
        subset = df_results[mask]
        bullish = "BUY" in action or "BULLISH" in action
        if bullish:
            correct = (subset["actual_dir"] == 1).sum()
        else:
            correct = (subset["actual_dir"] == 0).sum()
        accuracy_by_signal[action] = {
            "count": int(mask.sum()),
            "correct": int(correct),
            "accuracy": round(correct / max(mask.sum(), 1), 3),
        }

    output = {
        "test_period": f"{df_results.iloc[0]['date']} to {df_results.iloc[-1]['date']}",
        "total_days": len(df_results),
        "metrics": metrics,
        "signal_distribution": action_counts,
        "accuracy_by_signal": accuracy_by_signal,
        "strategies": {name: s["equity"] for name, s in strategies.items()},
    }

    if progress_cb:
        progress_cb("Backtest complete!", 1.0)

    return output


# ════════════════════════════════════════════════
#  TERMINAL OUTPUT
# ════════════════════════════════════════════════
def print_signal(signal):
    """Pretty-print a signal to terminal."""
    action = signal["action"]
    score = signal["composite_score"]
    color_map = {
        "STRONG BUY": "\033[92m", "BUY": "\033[92m", "LEAN BULLISH": "\033[32m",
        "HOLD / WAIT": "\033[93m",
        "LEAN BEARISH": "\033[33m", "SELL": "\033[91m", "STRONG SELL": "\033[91m",
    }
    c = color_map.get(action, "")
    r = "\033[0m"

    print(f"\n{'═' * 60}")
    print(f"  S&P 500 ML ADVISOR — {signal.get('date', 'Today')}")
    print(f"{'═' * 60}")
    print(f"\n  {c}{'▲' if score > 0 else '▼'} {action}  (Score: {score:+.0f}){r}")
    print(f"  Close: {signal.get('close', 0):,.2f}")
    print(f"  Timeframe: {signal.get('timeframe', 'N/A')}")
    print(f"  Position size: {signal.get('position_size', 'N/A')}")
    print(f"\n  {signal.get('explanation', '')}")

    # Sub-actions
    subs = signal.get("sub_actions", [])
    if subs:
        print(f"\n  Suggested Actions:")
        for i, s in enumerate(subs):
            print(f"  {i+1}. {s['label']}")
            print(f"     {s['detail']}")

    # Key metrics
    preds = signal.get("predictions", {})
    regime = signal.get("regime", {})
    print(f"\n  {'─' * 40}")
    print(f"  Model:  RF={preds.get('rf_prob', 0):.1%}  GB={preds.get('gb_prob', 0):.1%}  "
          f"Ensemble={preds.get('ensemble_prob', 0):.1%}")
    print(f"  Pred:   1d={preds.get('ret1_pct', 0):+.2f}%  5d={preds.get('ret5_pct', 0):+.2f}%  "
          f"Vol={preds.get('vol_pct', 0):.1f}%")
    print(f"  Regime: RSI={regime.get('rsi', 0):.0f}  MACD={regime.get('macd_hist', 0):.1f}  "
          f"SMA50/200={regime.get('sma_50_200', 0):.3f}")

    # Score breakdown
    scores = signal.get("scores", {})
    print(f"\n  Score Breakdown:")
    for k, v in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True):
        bar_len = int(abs(v) / 3)
        bar = ("█" * bar_len)
        sign = "+" if v >= 0 else ""
        c2 = "\033[32m" if v > 0 else "\033[31m" if v < 0 else ""
        print(f"  {k:15s} {c2}{sign}{v:5.0f} {bar}{r}")
    print()


def print_backtest(results):
    """Pretty-print backtest results."""
    print(f"\n{'═' * 70}")
    print(f"  SIGNAL STRATEGY BACKTEST — {results['test_period']}")
    print(f"{'═' * 70}")
    print(f"  Total trading days: {results['total_days']}")

    print(f"\n  {'Strategy':<25} {'Return':>8} {'CAGR':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'WinRate':>8} {'Exposure':>8}")
    print(f"  {'─' * 68}")
    for name, m in sorted(results["metrics"].items(),
                           key=lambda x: x[1]["sharpe"], reverse=True):
        flag = " ★" if name != "Buy & Hold" and m["sharpe"] > results["metrics"].get("Buy & Hold", {}).get("sharpe", 0) else ""
        print(f"  {name:<25} {m['total_return']:>+7.1f}% {m['cagr']:>7.2f}% "
              f"{m['sharpe']:>8.3f} {m['max_drawdown']:>7.1f}% "
              f"{m['win_rate']:>7.1%} {m['exposure']:>7.0%}{flag}")

    print(f"\n  Signal Distribution:")
    for action, count in sorted(results["signal_distribution"].items()):
        pct = count / results["total_days"] * 100
        print(f"    {action:<20} {count:>5} ({pct:.1f}%)")

    print(f"\n  Accuracy by Signal Type:")
    for action, stats in sorted(results["accuracy_by_signal"].items()):
        print(f"    {action:<20} {stats['correct']}/{stats['count']} "
              f"({stats['accuracy']:.1%})")
    print()


# ════════════════════════════════════════════════
#  EMAIL SIGNAL (plug into alert system)
# ════════════════════════════════════════════════
def format_signal_email(signal):
    """Format signal as HTML email body."""
    action = signal["action"]
    score = signal["composite_score"]
    preds = signal.get("predictions", {})
    regime = signal.get("regime", {})

    subs_html = ""
    for i, s in enumerate(signal.get("sub_actions", [])):
        subs_html += f"<p><strong>{i+1}. {s['label']}</strong><br>{s['detail']}</p>"

    color = signal.get("color", "#f59e0b")

    return f"""
    <div style="font-family: monospace; max-width: 600px; margin: 0 auto; background: #0f172a; color: #e0e0e0; padding: 24px; border-radius: 12px;">
        <h1 style="color: #60a5fa; font-size: 18px;">S&P 500 ML Advisor</h1>
        <div style="background: {color}15; border: 1px solid {color}40; border-radius: 10px; padding: 16px; margin: 12px 0;">
            <div style="font-size: 24px; font-weight: 800; color: {color};">{action}</div>
            <div style="color: #94a3b8; font-size: 13px;">Score: {score:+.0f} | {signal.get('date', 'Today')} | S&P at {signal.get('close', 0):,.2f}</div>
        </div>
        <p style="color: #cbd5e1; line-height: 1.6;">{signal.get('explanation', '')}</p>
        <p><strong>Timeframe:</strong> {signal.get('timeframe', 'N/A')}</p>
        <p><strong>Position Size:</strong> {signal.get('position_size', 'N/A')}</p>
        <h3 style="color: #60a5fa;">Suggested Actions</h3>
        {subs_html}
        <hr style="border-color: #1e293b;">
        <div style="font-size: 12px; color: #64748b;">
            <p>RF: {preds.get('rf_prob', 0):.1%} | GB: {preds.get('gb_prob', 0):.1%} | 
               Ensemble: {preds.get('ensemble_prob', 0):.1%}</p>
            <p>1d pred: {preds.get('ret1_pct', 0):+.2f}% | 5d pred: {preds.get('ret5_pct', 0):+.2f}% | 
               Vol: {preds.get('vol_pct', 0):.1f}%</p>
            <p>RSI: {regime.get('rsi', 0):.0f} | MACD: {regime.get('macd_hist', 0):.1f} | 
               SMA50/200: {regime.get('sma_50_200', 0):.3f}</p>
        </div>
        <p style="color: #ef4444; font-size: 11px; font-style: italic;">
            For educational purposes only — not financial advice.
        </p>
    </div>
    """


# ════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════
def find_data():
    for p in ["SP500_Analysis.xlsx", os.path.expanduser("~/Downloads/SP500_Analysis.xlsx")]:
        if os.path.exists(p):
            return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="S&P 500 Market Action Advisor")
    parser.add_argument("--backtest", action="store_true", help="Backtest signal strategy")
    parser.add_argument("--report", action="store_true", help="Show recent signals")
    parser.add_argument("--email-test", action="store_true", help="Test email formatting")
    parser.add_argument("--data", type=str, default=None, help="Path to Excel data")
    args = parser.parse_args()

    data_path = args.data or find_data()

    if args.backtest:
        if not data_path:
            print("Data file not found!")
            return
        results = backtest_signals(
            data_path,
            progress_cb=lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}")
        )
        if results:
            print_backtest(results)
            # Save
            with open(os.path.join(LOG_DIR, "signal_backtest.json"), "w") as f:
                json.dump({k: v for k, v in results.items() if k != "strategies"}, f, indent=2)

    elif args.report:
        advisor = DailyAdvisor()
        signals = advisor.get_recent_signals(10)
        if signals:
            for s in signals:
                print(f"  {s.get('date', '?'):12s} {s['action']:18s} Score: {s['composite_score']:+5.0f}  "
                      f"Close: {s.get('close', 0):>8,.2f}")
        else:
            print("  No signals recorded yet. Run without flags to generate one.")

    elif args.email_test:
        advisor = DailyAdvisor()
        signals = advisor.get_recent_signals(1)
        if signals:
            html = format_signal_email(signals[-1])
            path = os.path.join(LOG_DIR, "signal_email_preview.html")
            with open(path, "w") as f:
                f.write(f"<html><body style='background:#020617;padding:20px;'>{html}</body></html>")
            print(f"  Email preview: {path}")
        else:
            print("  No signals yet. Generate one first.")

    else:
        # Generate today's signal
        if not data_path:
            print("  SP500_Analysis.xlsx not found!")
            print("  Provide with: python sp500_advisor.py --data /path/to/file.xlsx")
            return
        advisor = DailyAdvisor()
        signal = advisor.generate_signal(
            data_path,
            progress_cb=lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}")
        )
        print_signal(signal)


if __name__ == "__main__":
    main()
