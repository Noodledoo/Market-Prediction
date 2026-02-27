#!/usr/bin/env python3
"""
sp500_predictor.py — S&P 500 Market Predictor
Traditional ML (Random Forest, Gradient Boosting) + LSTM Deep Learning
Full Pygame visualization app with training dashboard & backtesting

Usage:  python sp500_predictor.py
Requires: pygame, numpy, pandas, scikit-learn, matplotlib, openpyxl
Optional: torch (for LSTM model)

⚠️ For educational/research purposes only — not financial advice.
"""

import math
import random
import pickle
import time
import os
import sys
import warnings
import numpy as np
from collections import deque

warnings.filterwarnings("ignore")

# ========================== DEPENDENCY CHECKS ========================== #
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                   GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, mean_squared_error, mean_absolute_error, r2_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ========================== CONFIG ========================== #
DATA_FILE = "SP500_Analysis.xlsx"
SAVE_TRAD = "sp500_traditional_models.pkl"
SAVE_LSTM = "sp500_lstm_model.pkl"
SAVE_DATA = "sp500_prepared_data.pkl"

WIN_W, WIN_H = 1060, 720
PANEL_LEFT = 300
CHART_W = WIN_W - PANEL_LEFT - 40
CHART_H = 260

COLORS = {
    "bg": (20, 20, 30), "panel": (30, 30, 42), "card": (40, 40, 55),
    "text": (220, 220, 220), "dim": (120, 120, 130), "accent": (0, 220, 120),
    "warn": (220, 180, 50), "err": (220, 70, 70), "blue": (80, 160, 255),
    "purple": (160, 120, 255), "grid": (45, 45, 60),
    "rf": (16, 185, 129), "gb": (245, 158, 11), "bh": (107, 114, 128),
}

# ========================== HELPERS ========================== #
def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default

def find_data_file():
    """Search for the data file in common locations."""
    candidates = [
        DATA_FILE,
        os.path.join(".", DATA_FILE),
        os.path.join(os.path.dirname(__file__), DATA_FILE),
        os.path.expanduser(f"~/Downloads/{DATA_FILE}"),
        os.path.expanduser(f"~/Desktop/{DATA_FILE}"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def plot_to_surface(plot_fn, width=520, height=250):
    """Render a matplotlib plot to a pygame surface."""
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


# ========================== DATA PIPELINE ========================== #
class DataPipeline:
    """Load, engineer features, and prepare train/test splits."""

    FEATURE_COLS = [
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
        'close_sma_5_ratio', 'close_sma_10_ratio', 'close_sma_20_ratio',
        'close_sma_50_ratio', 'close_sma_200_ratio',
        'sma_5_20_ratio', 'sma_20_50_ratio', 'sma_50_200_ratio',
        'macd', 'macd_signal', 'macd_hist',
        'rsi_14', 'bb_pct', 'atr_pct',
        'vol_5d', 'vol_10d', 'vol_20d',
        'vol_ratio', 'hl_range', 'gap', 'dow'
    ]

    TARGET_COLS = ['target_ret_1d', 'target_dir_1d', 'target_ret_5d', 'target_vol_5d']

    def __init__(self):
        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.scaler = StandardScaler()
        self.train_dates = self.test_dates = None
        self.info = {}

    def load_and_prepare(self, filepath, progress_cb=None):
        """Full pipeline: load → engineer → split → scale."""
        if progress_cb:
            progress_cb("Loading Excel data...", 0.05)

        df = pd.read_excel(filepath, sheet_name="Daily Data")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        if progress_cb:
            progress_cb(f"Loaded {len(df)} rows. Engineering features...", 0.15)

        # --- Returns ---
        for n in [1, 2, 3, 5, 10, 20]:
            df[f'ret_{n}d'] = df['Close'].pct_change(n)

        # --- Moving averages & ratios ---
        for n in [5, 10, 20, 50, 200]:
            df[f'sma_{n}'] = df['Close'].rolling(n).mean()
            df[f'close_sma_{n}_ratio'] = df['Close'] / df[f'sma_{n}']
        df['sma_5_20_ratio'] = df['sma_5'] / df['sma_20']
        df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
        df['sma_50_200_ratio'] = df['sma_50'] / df['sma_200']

        if progress_cb:
            progress_cb("Computing technical indicators...", 0.35)

        # --- EMAs & MACD ---
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # --- RSI 14 ---
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # --- Bollinger Bands ---
        bb_sma = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_pct'] = (df['Close'] - (bb_sma - 2 * bb_std)) / (4 * bb_std)

        # --- ATR ---
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / df['Close']

        if progress_cb:
            progress_cb("Computing volatility & volume features...", 0.55)

        # --- Volatility ---
        for n in [5, 10, 20]:
            df[f'vol_{n}d'] = df['ret_1d'].rolling(n).std() * np.sqrt(252)

        # --- Volume ---
        df['vol_sma_20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma_20']

        # --- Other ---
        df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        df['gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
        df['dow'] = df['Date'].dt.dayofweek

        # --- Targets ---
        df['target_ret_1d'] = df['Close'].shift(-1) / df['Close'] - 1
        df['target_dir_1d'] = (df['target_ret_1d'] > 0).astype(int)
        df['target_ret_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target_vol_5d'] = df['ret_1d'].shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)

        if progress_cb:
            progress_cb("Preparing train/test split...", 0.75)

        # --- Clean & split ---
        valid = df.dropna(subset=self.FEATURE_COLS + self.TARGET_COLS).copy()
        split = int(len(valid) * 0.8)
        train = valid.iloc[:split]
        test = valid.iloc[split:]

        self.X_train = self.scaler.fit_transform(train[self.FEATURE_COLS].values)
        self.X_test = self.scaler.transform(test[self.FEATURE_COLS].values)
        self.y_train = {col: train[col].values for col in self.TARGET_COLS}
        self.y_test = {col: test[col].values for col in self.TARGET_COLS}
        self.train_dates = train['Date'].values
        self.test_dates = test['Date'].values
        self.test_close = test['Close'].values
        self.df = df

        self.info = {
            "total_rows": len(df),
            "valid_rows": len(valid),
            "train_size": len(train),
            "test_size": len(test),
            "date_range": f"{df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}",
            "train_range": f"{train['Date'].iloc[0].strftime('%Y')}–{train['Date'].iloc[-1].strftime('%Y')}",
            "test_range": f"{test['Date'].iloc[0].strftime('%Y')}–{test['Date'].iloc[-1].strftime('%Y')}",
            "num_features": len(self.FEATURE_COLS),
        }

        if progress_cb:
            progress_cb("Data ready!", 1.0)

        return self.info


# ========================== TRADITIONAL ML ========================== #
class TraditionalModels:
    """Random Forest + Gradient Boosting for all targets."""

    MODELS = {
        "RF_direction": lambda: RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "GB_direction": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=20, random_state=42),
        "RF_return_1d": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "GB_return_1d": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=20, random_state=42),
        "RF_return_5d": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "GB_return_5d": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=20, random_state=42),
        "RF_volatility": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "GB_volatility": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=20, random_state=42),
    }

    TARGET_MAP = {
        "direction": "target_dir_1d",
        "return_1d": "target_ret_1d",
        "return_5d": "target_ret_5d",
        "volatility": "target_vol_5d",
    }

    def __init__(self):
        self.trained = {}
        self.results = {}
        self.feature_importance = {}

    def train_all(self, data: DataPipeline, progress_cb=None):
        total = len(self.MODELS)
        for i, (name, model_fn) in enumerate(self.MODELS.items()):
            target_key = name.split("_", 1)[1]
            target_col = self.TARGET_MAP[target_key]

            if progress_cb:
                progress_cb(f"Training {name}...", (i + 0.5) / total)

            model = model_fn()
            model.fit(data.X_train, data.y_train[target_col])
            self.trained[name] = model

            y_pred = model.predict(data.X_test)
            y_true = data.y_test[target_col]

            if "direction" in name:
                y_prob = model.predict_proba(data.X_test)[:, 1]
                self.results[name] = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred),
                    "recall": recall_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred),
                    "predictions": y_pred,
                    "probabilities": y_prob,
                }
            else:
                self.results[name] = {
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                    "predictions": y_pred,
                }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(
                    zip(DataPipeline.FEATURE_COLS, model.feature_importances_))

        # Backtest
        self._run_backtest(data)

        if progress_cb:
            progress_cb("Traditional models trained!", 1.0)

    def _run_backtest(self, data):
        """Simple long/cash backtest based on direction predictions."""
        ret = data.y_test['target_ret_1d']

        for prefix in ["RF", "GB"]:
            preds = self.results[f"{prefix}_direction"]["predictions"]
            equity = [1.0]
            for i in range(len(preds) - 1):
                if preds[i] == 1:
                    equity.append(equity[-1] * (1 + ret[i]))
                else:
                    equity.append(equity[-1])

            bh = [1.0]
            for i in range(len(ret) - 1):
                bh.append(bh[-1] * (1 + ret[i]))

            strat_ret = np.diff(np.log(np.clip(equity, 1e-10, None)))
            bh_ret = np.diff(np.log(np.clip(bh, 1e-10, None)))

            self.results[f"{prefix}_backtest"] = {
                "equity": equity,
                "buy_hold": bh,
                "total_return": (equity[-1] - 1) * 100,
                "bh_return": (bh[-1] - 1) * 100,
                "sharpe": np.mean(strat_ret) / max(np.std(strat_ret), 1e-10) * np.sqrt(252),
                "bh_sharpe": np.mean(bh_ret) / max(np.std(bh_ret), 1e-10) * np.sqrt(252),
            }

    def save(self):
        save_pickle(SAVE_TRAD, {"results": {k: {kk: vv for kk, vv in v.items()
                     if not isinstance(vv, np.ndarray)} for k, v in self.results.items()},
                     "feature_importance": self.feature_importance})

    def load(self):
        data = load_pickle(SAVE_TRAD)
        if data:
            self.results = data.get("results", {})
            self.feature_importance = data.get("feature_importance", {})
            return True
        return False


# ========================== LSTM MODEL ========================== #
if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
            self.fc_dir = nn.Linear(hidden, 1)
            self.fc_ret1 = nn.Linear(hidden, 1)
            self.fc_ret5 = nn.Linear(hidden, 1)
            self.fc_vol = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            h = out[:, -1, :]
            return (torch.sigmoid(self.fc_dir(h)),
                    self.fc_ret1(h), self.fc_ret5(h),
                    torch.relu(self.fc_vol(h)))

    class LSTMTrainer:
        SEQ_LEN = 30

        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.results = {}
            self.losses = []

        def _make_sequences(self, X, y_dict, seq_len):
            seqs, targets = [], {k: [] for k in y_dict}
            for i in range(seq_len, len(X)):
                seqs.append(X[i - seq_len:i])
                for k in y_dict:
                    targets[k].append(y_dict[k][i])
            return (np.array(seqs),
                    {k: np.array(v) for k, v in targets.items()})

        def train(self, data: DataPipeline, epochs=80, lr=1e-3, batch_size=128, progress_cb=None):
            n_feat = data.X_train.shape[1]
            self.model = LSTMNet(n_feat).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

            X_seq, y_seq = self._make_sequences(data.X_train, data.y_train, self.SEQ_LEN)
            X_test_seq, y_test_seq = self._make_sequences(data.X_test, data.y_test, self.SEQ_LEN)

            X_t = torch.FloatTensor(X_seq).to(self.device)
            dir_t = torch.FloatTensor(y_seq['target_dir_1d']).unsqueeze(1).to(self.device)
            r1_t = torch.FloatTensor(y_seq['target_ret_1d']).unsqueeze(1).to(self.device)
            r5_t = torch.FloatTensor(y_seq['target_ret_5d']).unsqueeze(1).to(self.device)
            vol_t = torch.FloatTensor(y_seq['target_vol_5d']).unsqueeze(1).to(self.device)

            self.losses = []
            n = len(X_seq)

            for epoch in range(epochs):
                self.model.train()
                idx = torch.randperm(n)
                epoch_loss = 0.0
                batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    bi = idx[start:end]
                    d_pred, r1_pred, r5_pred, v_pred = self.model(X_t[bi])

                    loss = (nn.BCELoss()(d_pred, dir_t[bi])
                            + nn.MSELoss()(r1_pred, r1_t[bi]) * 100
                            + nn.MSELoss()(r5_pred, r5_t[bi]) * 50
                            + nn.MSELoss()(v_pred, vol_t[bi]) * 10)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1

                avg_loss = epoch_loss / max(batches, 1)
                self.losses.append(avg_loss)

                if progress_cb:
                    progress_cb(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}",
                               (epoch + 1) / epochs)

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                X_te = torch.FloatTensor(X_test_seq).to(self.device)
                d_pred, r1_pred, r5_pred, v_pred = self.model(X_te)

                d_np = (d_pred.cpu().numpy().flatten() > 0.5).astype(int)
                d_true = y_test_seq['target_dir_1d'].astype(int)

                self.results = {
                    "direction": {
                        "accuracy": accuracy_score(d_true, d_np),
                        "f1": f1_score(d_true, d_np),
                        "predictions": d_np,
                        "probabilities": d_pred.cpu().numpy().flatten(),
                    },
                    "return_1d": {
                        "rmse": np.sqrt(mean_squared_error(y_test_seq['target_ret_1d'],
                                                            r1_pred.cpu().numpy().flatten())),
                        "r2": r2_score(y_test_seq['target_ret_1d'],
                                       r1_pred.cpu().numpy().flatten()),
                    },
                    "volatility": {
                        "rmse": np.sqrt(mean_squared_error(y_test_seq['target_vol_5d'],
                                                            v_pred.cpu().numpy().flatten())),
                        "r2": r2_score(y_test_seq['target_vol_5d'],
                                       v_pred.cpu().numpy().flatten()),
                    },
                    "losses": self.losses,
                }

            if progress_cb:
                progress_cb("LSTM training complete!", 1.0)

        def save(self):
            if self.model:
                save_pickle(SAVE_LSTM, {
                    "state_dict": self.model.state_dict(),
                    "results": self.results,
                    "losses": self.losses
                })

        def load(self):
            data = load_pickle(SAVE_LSTM)
            if data:
                self.results = data.get("results", {})
                self.losses = data.get("losses", [])
                return True
            return False


# ========================== UI ========================== #
def draw_text(screen, text, pos, font, color=COLORS["text"]):
    screen.blit(font.render(str(text), True, color), pos)

def draw_card(screen, rect, title, lines, font_t, font_b, highlight_color=COLORS["accent"]):
    pygame.draw.rect(screen, COLORS["card"], rect, border_radius=8)
    pygame.draw.rect(screen, (60, 60, 75), rect, 1, border_radius=8)
    draw_text(screen, title, (rect.x + 12, rect.y + 8), font_t, COLORS["dim"])
    for i, (txt, col) in enumerate(lines):
        draw_text(screen, txt, (rect.x + 12, rect.y + 28 + i * 22), font_b, col)

def draw_button(screen, font, rect, text, hover=False, color=COLORS["accent"]):
    c = tuple(min(x + 30, 255) for x in color) if hover else color
    pygame.draw.rect(screen, c, rect, border_radius=6)
    pygame.draw.rect(screen, (255, 255, 255), rect, 1, border_radius=6)
    ts = font.render(text, True, (255, 255, 255))
    screen.blit(ts, (rect.centerx - ts.get_width() // 2, rect.centery - ts.get_height() // 2))

def draw_progress(screen, rect, pct, color=COLORS["accent"]):
    pygame.draw.rect(screen, COLORS["card"], rect, border_radius=4)
    filled = pygame.Rect(rect.x, rect.y, int(rect.width * pct), rect.height)
    pygame.draw.rect(screen, color, filled, border_radius=4)
    pygame.draw.rect(screen, (80, 80, 95), rect, 1, border_radius=4)


# ========================== SCREENS ========================== #
def screen_load_data():
    """Load and prepare data with progress display."""
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Loading S&P 500 Data...")
    font = pygame.font.SysFont(None, 26)
    font_big = pygame.font.SysFont(None, 34)
    clock = pygame.time.Clock()

    filepath = find_data_file()
    if not filepath:
        # Show error
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
            screen.fill(COLORS["bg"])
            draw_text(screen, "File Not Found", (50, 50), font_big, COLORS["err"])
            draw_text(screen, f"Place '{DATA_FILE}' in the same folder as this script.", (50, 100), font, COLORS["text"])
            draw_text(screen, "Press ESC to return.", (50, 140), font, COLORS["dim"])
            pygame.display.flip()
            clock.tick(30)
        return None

    data = DataPipeline()
    status = ["Initializing..."]
    progress = [0.0]
    done = [False]

    def progress_cb(msg, pct):
        status[0] = msg
        progress[0] = pct
        # Pump events during long operation
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(COLORS["bg"])
        draw_text(screen, "S&P 500 Market Predictor", (50, 30), font_big, COLORS["accent"])
        draw_text(screen, status[0], (50, 100), font, COLORS["text"])
        bar = pygame.Rect(50, 150, 500, 20)
        draw_progress(screen, bar, progress[0])
        draw_text(screen, f"{int(progress[0]*100)}%", (560, 148), font, COLORS["dim"])
        pygame.display.flip()
        clock.tick(60)

    info = data.load_and_prepare(filepath, progress_cb)

    # Show summary briefly
    screen.fill(COLORS["bg"])
    draw_text(screen, "Data Ready!", (50, 30), font_big, COLORS["accent"])
    y = 80
    for k, v in info.items():
        draw_text(screen, f"{k}: {v}", (50, y), font, COLORS["text"])
        y += 28
    pygame.display.flip()
    pygame.time.delay(1500)

    return data


def screen_train_traditional(data):
    """Train RF + GB models with progress visualization."""
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Training Traditional ML Models")
    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 30)
    font_sm = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    models = TraditionalModels()
    status = ["Starting..."]
    progress = [0.0]
    log_lines = []

    def progress_cb(msg, pct):
        status[0] = msg
        progress[0] = pct
        log_lines.append(msg)
        if len(log_lines) > 18:
            log_lines.pop(0)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        _draw_training_ui(screen, font, font_big, font_sm, status[0], progress[0], log_lines)
        clock.tick(60)

    t0 = time.time()
    models.train_all(data, progress_cb)
    elapsed = time.time() - t0
    models.save()

    log_lines.append(f"Completed in {elapsed:.1f}s — saved to {SAVE_TRAD}")

    # Show results
    running = True
    result_surf = None
    if MPL_AVAILABLE:
        def plot_results(ax):
            labels = ["RF Dir Acc", "GB Dir Acc", "RF Vol R²", "GB Vol R²", "RF Sharpe", "GB Sharpe"]
            rf_bt = models.results.get("RF_backtest", {})
            gb_bt = models.results.get("GB_backtest", {})
            vals = [
                models.results.get("RF_direction", {}).get("accuracy", 0),
                models.results.get("GB_direction", {}).get("accuracy", 0),
                models.results.get("RF_volatility", {}).get("r2", 0),
                models.results.get("GB_volatility", {}).get("r2", 0),
                rf_bt.get("sharpe", 0),
                gb_bt.get("sharpe", 0),
            ]
            colors = ['#10b981', '#f59e0b'] * 3
            ax.barh(labels, vals, color=colors)
            ax.set_xlim(0, max(vals) * 1.2 if vals else 1)
            ax.set_title("Model Performance", color="#cccccc", fontsize=11)
            ax.tick_params(colors="#aaaaaa")
        result_surf = plot_to_surface(plot_results, CHART_W, CHART_H)

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                running = False

        screen.fill(COLORS["bg"])
        draw_text(screen, "Training Complete!", (30, 20), font_big, COLORS["accent"])
        draw_text(screen, f"Time: {elapsed:.1f}s | Press ENTER to continue", (30, 50), font, COLORS["dim"])

        y = 90
        for line in log_lines[-15:]:
            draw_text(screen, line, (30, y), font_sm, COLORS["text"])
            y += 20

        if result_surf:
            screen.blit(result_surf, (30, WIN_H - CHART_H - 30))

        # Show key metrics on right
        x_right = WIN_W - 350
        draw_text(screen, "Key Results", (x_right, 90), font_big, COLORS["accent"])
        y = 130
        for name, res in models.results.items():
            if "backtest" in name:
                draw_text(screen, f"{name}: Return={res.get('total_return',0):.1f}% | Sharpe={res.get('sharpe',0):.3f}",
                         (x_right, y), font_sm, COLORS["rf"] if "RF" in name else COLORS["gb"])
                y += 22
            elif "direction" in name:
                draw_text(screen, f"{name}: Acc={res.get('accuracy',0):.3f} | F1={res.get('f1',0):.3f}",
                         (x_right, y), font_sm, COLORS["rf"] if "RF" in name else COLORS["gb"])
                y += 22
            elif "volatility" in name:
                draw_text(screen, f"{name}: R²={res.get('r2',0):.3f} | RMSE={res.get('rmse',0):.4f}",
                         (x_right, y), font_sm, COLORS["rf"] if "RF" in name else COLORS["gb"])
                y += 22

        pygame.display.flip()
        clock.tick(30)

    return models


def screen_train_lstm(data):
    """Train LSTM with epoch-by-epoch progress."""
    if not TORCH_AVAILABLE:
        return None

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Training LSTM Network")
    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 30)
    font_sm = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    trainer = LSTMTrainer()
    status = ["Initializing LSTM..."]
    progress = [0.0]
    log_lines = []

    def progress_cb(msg, pct):
        status[0] = msg
        progress[0] = pct
        log_lines.append(msg)
        if len(log_lines) > 20:
            log_lines.pop(0)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        _draw_training_ui(screen, font, font_big, font_sm, status[0], progress[0], log_lines,
                         title="LSTM Training")
        clock.tick(60)

    t0 = time.time()
    trainer.train(data, epochs=80, progress_cb=progress_cb)
    elapsed = time.time() - t0
    trainer.save()

    # Show results
    running = True
    loss_surf = None
    if MPL_AVAILABLE and trainer.losses:
        def plot_loss(ax):
            ax.plot(trainer.losses, color='#10b981', linewidth=1.5)
            ax.set_title("Training Loss", color="#cccccc", fontsize=11)
            ax.set_xlabel("Epoch", color="#aaaaaa", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)
        loss_surf = plot_to_surface(plot_loss, CHART_W, CHART_H)

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                running = False

        screen.fill(COLORS["bg"])
        draw_text(screen, "LSTM Training Complete!", (30, 20), font_big, COLORS["accent"])
        draw_text(screen, f"Time: {elapsed:.1f}s | Press ENTER to continue", (30, 50), font, COLORS["dim"])

        y = 90
        r = trainer.results
        if r.get("direction"):
            draw_text(screen, f"Direction Accuracy: {r['direction']['accuracy']:.3f} | F1: {r['direction']['f1']:.3f}",
                     (30, y), font, COLORS["accent"])
            y += 28
        if r.get("return_1d"):
            draw_text(screen, f"1d Return R²: {r['return_1d']['r2']:.4f} | RMSE: {r['return_1d']['rmse']:.6f}",
                     (30, y), font, COLORS["blue"])
            y += 28
        if r.get("volatility"):
            draw_text(screen, f"Volatility R²: {r['volatility']['r2']:.4f} | RMSE: {r['volatility']['rmse']:.6f}",
                     (30, y), font, COLORS["purple"])
            y += 28

        if loss_surf:
            screen.blit(loss_surf, (30, WIN_H - CHART_H - 30))

        pygame.display.flip()
        clock.tick(30)

    return trainer


def screen_dashboard(data, trad_models, lstm_trainer=None):
    """Full comparison dashboard with charts."""
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("S&P 500 Model Comparison Dashboard")
    font = pygame.font.SysFont(None, 24)
    font_big = pygame.font.SysFont(None, 30)
    font_sm = pygame.font.SysFont(None, 20)
    font_xs = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    # Pre-render charts
    charts = {}

    if MPL_AVAILABLE and trad_models and trad_models.results:
        # Equity curve
        def plot_equity(ax):
            rf_bt = trad_models.results.get("RF_backtest", {})
            gb_bt = trad_models.results.get("GB_backtest", {})
            if rf_bt.get("equity"):
                ax.plot(rf_bt["equity"], color='#10b981', linewidth=1.5, label="Random Forest")
            if gb_bt.get("equity"):
                ax.plot(gb_bt["equity"], color='#f59e0b', linewidth=1.5, label="Gradient Boosting")
            if rf_bt.get("buy_hold"):
                ax.plot(rf_bt["buy_hold"], color='#6b7280', linewidth=1, linestyle='--', label="Buy & Hold")
            ax.set_title("Equity Curves", color="#cccccc", fontsize=11)
            ax.legend(fontsize=8, facecolor="#2a2a3a", edgecolor="#444455", labelcolor="#cccccc")
            ax.grid(True, linestyle="--", alpha=0.3)
        charts["equity"] = plot_to_surface(plot_equity, 480, 240)

        # Feature importance
        if trad_models.feature_importance:
            def plot_features(ax):
                avg = {}
                count = 0
                for imp in trad_models.feature_importance.values():
                    count += 1
                    for feat, score in imp.items():
                        avg[feat] = avg.get(feat, 0) + score
                for f in avg:
                    avg[f] /= count
                top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:15]
                names = [t[0] for t in reversed(top)]
                vals = [t[1] for t in reversed(top)]
                colors = ['#10b981' if v > 0.05 else '#3b82f6' if v > 0.03 else '#6366f1' for v in vals]
                ax.barh(names, vals, color=colors)
                ax.set_title("Top 15 Features (Avg Importance)", color="#cccccc", fontsize=11)
                ax.tick_params(colors="#aaaaaa", labelsize=8)
            charts["features"] = plot_to_surface(plot_features, 480, 240)

    # Tab state
    tabs = ["Overview", "Equity", "Features"]
    tab = 0

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_LEFT:
                    tab = (tab - 1) % len(tabs)
                elif e.key == pygame.K_RIGHT:
                    tab = (tab + 1) % len(tabs)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                for i, t in enumerate(tabs):
                    tx = 30 + i * 130
                    if tx <= e.pos[0] <= tx + 120 and 55 <= e.pos[1] <= 80:
                        tab = i

        screen.fill(COLORS["bg"])
        draw_text(screen, "S&P 500 ML Dashboard", (30, 15), font_big, COLORS["accent"])

        # Tab buttons
        for i, t in enumerate(tabs):
            tx = 30 + i * 130
            rect = pygame.Rect(tx, 55, 120, 26)
            if i == tab:
                pygame.draw.rect(screen, (30, 80, 60), rect, border_radius=4)
                pygame.draw.rect(screen, COLORS["accent"], rect, 1, border_radius=4)
            draw_text(screen, t, (tx + 8, 59), font_sm, COLORS["accent"] if i == tab else COLORS["dim"])

        y_start = 95

        if tab == 0:  # Overview
            if trad_models and trad_models.results:
                y = y_start
                draw_text(screen, "Direction Prediction", (30, y), font, COLORS["text"])
                y += 28
                for name in ["RF_direction", "GB_direction"]:
                    r = trad_models.results.get(name, {})
                    color = COLORS["rf"] if "RF" in name else COLORS["gb"]
                    label = "Random Forest" if "RF" in name else "Gradient Boosting"
                    draw_text(screen, f"  {label}: Acc={r.get('accuracy',0):.3f}  F1={r.get('f1',0):.3f}  "
                             f"Prec={r.get('precision',0):.3f}  Rec={r.get('recall',0):.3f}",
                             (30, y), font_sm, color)
                    y += 22

                y += 15
                draw_text(screen, "Return Prediction (1-Day)", (30, y), font, COLORS["text"])
                y += 28
                for name in ["RF_return_1d", "GB_return_1d"]:
                    r = trad_models.results.get(name, {})
                    color = COLORS["rf"] if "RF" in name else COLORS["gb"]
                    label = "RF" if "RF" in name else "GB"
                    draw_text(screen, f"  {label}: RMSE={r.get('rmse',0):.6f}  R²={r.get('r2',0):.4f}",
                             (30, y), font_sm, color)
                    y += 22

                y += 15
                draw_text(screen, "Volatility Prediction", (30, y), font, COLORS["text"])
                y += 28
                for name in ["RF_volatility", "GB_volatility"]:
                    r = trad_models.results.get(name, {})
                    color = COLORS["rf"] if "RF" in name else COLORS["gb"]
                    label = "RF" if "RF" in name else "GB"
                    draw_text(screen, f"  {label}: RMSE={r.get('rmse',0):.4f}  R²={r.get('r2',0):.4f}",
                             (30, y), font_sm, color)
                    y += 22

                y += 15
                draw_text(screen, "Backtest (Long/Cash Strategy)", (30, y), font, COLORS["text"])
                y += 28
                for prefix in ["RF", "GB"]:
                    bt = trad_models.results.get(f"{prefix}_backtest", {})
                    color = COLORS["rf"] if prefix == "RF" else COLORS["gb"]
                    label = "Random Forest" if prefix == "RF" else "Gradient Boosting"
                    draw_text(screen, f"  {label}: Return={bt.get('total_return',0):.1f}%  "
                             f"Sharpe={bt.get('sharpe',0):.3f}  (B&H: {bt.get('bh_return',0):.1f}%, {bt.get('bh_sharpe',0):.3f})",
                             (30, y), font_sm, color)
                    y += 22

                if lstm_trainer and lstm_trainer.results:
                    y += 15
                    draw_text(screen, "LSTM Results", (30, y), font, COLORS["purple"])
                    y += 28
                    r = lstm_trainer.results
                    if r.get("direction"):
                        draw_text(screen, f"  Direction: Acc={r['direction']['accuracy']:.3f}  F1={r['direction']['f1']:.3f}",
                                 (30, y), font_sm, COLORS["purple"])
                        y += 22
                    if r.get("volatility"):
                        draw_text(screen, f"  Volatility: R²={r['volatility']['r2']:.4f}",
                                 (30, y), font_sm, COLORS["purple"])
                        y += 22

        elif tab == 1:  # Equity
            if charts.get("equity"):
                screen.blit(charts["equity"], (30, y_start))
            draw_text(screen, "Strategy: Long when model predicts UP, Cash when DOWN", (30, y_start + 250), font_sm, COLORS["dim"])
            draw_text(screen, "Test period: 2011–2026", (30, y_start + 272), font_sm, COLORS["dim"])

        elif tab == 2:  # Features
            if charts.get("features"):
                screen.blit(charts["features"], (30, y_start))
            draw_text(screen, "Volatility features dominate — market regime is the strongest signal",
                     (30, y_start + 250), font_sm, COLORS["dim"])

        draw_text(screen, "← → to switch tabs | ESC to return", (30, WIN_H - 30), font_xs, COLORS["dim"])
        pygame.display.flip()
        clock.tick(30)


def _draw_training_ui(screen, font, font_big, font_sm, status, progress_pct, log_lines, title="Training"):
    screen.fill(COLORS["bg"])
    draw_text(screen, title, (30, 20), font_big, COLORS["accent"])
    draw_text(screen, status, (30, 55), font, COLORS["text"])
    bar = pygame.Rect(30, 85, WIN_W - 60, 16)
    draw_progress(screen, bar, progress_pct)
    draw_text(screen, f"{int(progress_pct*100)}%", (WIN_W - 50, 83), font_sm, COLORS["dim"])

    y = 120
    for line in log_lines:
        draw_text(screen, line, (30, y), font_sm, COLORS["dim"])
        y += 18
    pygame.display.flip()


# ========================== MAIN MENU ========================== #
def main_menu():
    if not PYGAME_AVAILABLE:
        print("ERROR: pygame not installed. Run: pip install pygame")
        return
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return
    if not PANDAS_AVAILABLE:
        print("ERROR: pandas not installed. Run: pip install pandas")
        return

    pygame.init()
    W, H = 560, 520
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("S&P 500 ML Predictor")
    ft = pygame.font.SysFont(None, 44)
    fn = pygame.font.SysFont(None, 28)
    fs = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    options = [
        ("Load & Prepare Data", "load"),
        ("Train: Random Forest + Gradient Boosting", "train_trad"),
        ("Train: LSTM Network", "train_lstm"),
        ("View Dashboard", "dashboard"),
        ("Quit", "quit"),
    ]

    selected = 0
    running = True
    data = None
    trad_models = None
    lstm_trainer = None

    # Try loading existing models
    trad_models = TraditionalModels()
    if not trad_models.load():
        trad_models = None
    if TORCH_AVAILABLE:
        lstm_trainer = LSTMTrainer()
        if not lstm_trainer.load():
            lstm_trainer = None

    while running:
        screen.fill(COLORS["bg"])
        screen.blit(ft.render("S&P 500 ML Predictor", True, COLORS["accent"]), (100, 22))

        # Status line
        dep_parts = []
        dep_parts.append(f"sklearn: ✓" if SKLEARN_AVAILABLE else "sklearn: ✗")
        if TORCH_AVAILABLE:
            gpu = "GPU" if torch.cuda.is_available() else "CPU"
            dep_parts.append(f"PyTorch: {gpu}")
        else:
            dep_parts.append("PyTorch: ✗ (LSTM disabled)")
        screen.blit(fs.render(" | ".join(dep_parts), True, COLORS["dim"]), (100, 65))

        # Data status
        status_parts = []
        if data:
            status_parts.append(f"Data: ✓ ({data.info.get('valid_rows', 0)} rows)")
        else:
            status_parts.append("Data: not loaded")
        if trad_models and trad_models.results:
            status_parts.append("Trad ML: ✓")
        if lstm_trainer and lstm_trainer.results:
            status_parts.append("LSTM: ✓")
        screen.blit(fs.render(" | ".join(status_parts), True,
                     COLORS["accent"] if data else COLORS["warn"]), (100, 85))

        for i, (label, act) in enumerate(options):
            is_lstm = act == "train_lstm"
            disabled = (is_lstm and not TORCH_AVAILABLE) or (act in ("train_trad", "train_lstm") and data is None)
            sel = i == selected

            if disabled:
                color = (80, 80, 80)
            elif sel:
                color = (255, 255, 0)
            else:
                color = (200, 200, 200)

            prefix = "▸ " if sel else "  "
            suffix = ""
            if disabled:
                if is_lstm and not TORCH_AVAILABLE:
                    suffix = "  (need PyTorch)"
                elif data is None:
                    suffix = "  (load data first)"

            screen.blit(fn.render(prefix + label + suffix, True, color), (40, 130 + i * 55))

        # Instructions
        screen.blit(fs.render("↑/↓ Navigate  |  ENTER Select", True, COLORS["dim"]), (140, H - 40))

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif e.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif e.key == pygame.K_RETURN:
                    _, act = options[selected]

                    if act == "load":
                        data = screen_load_data()
                        screen = pygame.display.set_mode((W, H))
                        pygame.display.set_caption("S&P 500 ML Predictor")

                    elif act == "train_trad" and data:
                        trad_models = screen_train_traditional(data)
                        screen = pygame.display.set_mode((W, H))
                        pygame.display.set_caption("S&P 500 ML Predictor")

                    elif act == "train_lstm" and data and TORCH_AVAILABLE:
                        lstm_trainer = screen_train_lstm(data)
                        screen = pygame.display.set_mode((W, H))
                        pygame.display.set_caption("S&P 500 ML Predictor")

                    elif act == "dashboard":
                        if trad_models or lstm_trainer:
                            screen_dashboard(data, trad_models, lstm_trainer)
                            screen = pygame.display.set_mode((W, H))
                            pygame.display.set_caption("S&P 500 ML Predictor")

                    elif act == "quit":
                        running = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ========================== CLI FALLBACK ========================== #
def cli_mode():
    """Run in terminal if pygame isn't available."""
    print("=" * 60)
    print("S&P 500 ML Predictor — CLI Mode")
    print("(Install pygame for the full GUI experience)")
    print("=" * 60)

    filepath = find_data_file()
    if not filepath:
        print(f"\nERROR: '{DATA_FILE}' not found. Place it in the current directory.")
        return

    data = DataPipeline()
    info = data.load_and_prepare(filepath, lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}"))
    print(f"\n{info}")

    print("\n--- Training Traditional Models ---")
    models = TraditionalModels()
    models.train_all(data, lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}"))
    models.save()

    print("\n--- Results ---")
    for name, res in models.results.items():
        print(f"\n{name}:")
        for k, v in res.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, list) and len(v) > 10:
                print(f"  {k}: [{v[0]:.4f}, ..., {v[-1]:.4f}] (len={len(v)})")

    if TORCH_AVAILABLE:
        print("\n--- Training LSTM ---")
        lstm = LSTMTrainer()
        lstm.train(data, epochs=80, progress_cb=lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}"))
        lstm.save()
        print("\nLSTM Results:")
        for k, v in lstm.results.items():
            if isinstance(v, dict):
                print(f"  {k}: {v}")

    print("\nDone! Models saved.")


# ========================== ENTRY ========================== #
if __name__ == "__main__":
    if PYGAME_AVAILABLE and SKLEARN_AVAILABLE and PANDAS_AVAILABLE:
        main_menu()
    elif SKLEARN_AVAILABLE and PANDAS_AVAILABLE:
        cli_mode()
    else:
        missing = []
        if not PANDAS_AVAILABLE:
            missing.append("pandas")
        if not SKLEARN_AVAILABLE:
            missing.append("scikit-learn")
        if not PYGAME_AVAILABLE:
            missing.append("pygame")
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
