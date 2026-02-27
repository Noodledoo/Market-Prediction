"""Feature engineering for S&P 500 prediction model."""

import os
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "sp500_raw.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "sp500_features.csv")


def build_features(df=None):
    """Generate technical indicator features from raw OHLCV data.

    Args:
        df: Optional DataFrame. If None, reads from RAW_FILE.

    Returns:
        DataFrame with features and target column.
    """
    if df is None:
        if not os.path.exists(RAW_FILE):
            raise FileNotFoundError(f"{RAW_FILE} not found. Run fetch_data.py first.")
        df = pd.read_csv(RAW_FILE, index_col="Date", parse_dates=True)

    print(f"Building features from {len(df)} rows...")

    # Price-based features
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_21d"] = df["Close"].pct_change(21)

    # Moving averages
    df["SMA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["EMA_12"] = EMAIndicator(close=df["Close"], window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close=df["Close"], window=26).ema_indicator()

    # Price relative to moving averages
    df["Close_to_SMA10"] = df["Close"] / df["SMA_10"] - 1
    df["Close_to_SMA50"] = df["Close"] / df["SMA_50"] - 1
    df["SMA10_to_SMA50"] = df["SMA_10"] / df["SMA_50"] - 1

    # RSI
    df["RSI_14"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # MACD
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["Close"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"])
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # ATR (volatility)
    df["ATR_14"] = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()
    df["ATR_Pct"] = df["ATR_14"] / df["Close"]

    # Volume features
    df["Volume_SMA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]

    # Volatility (rolling std of returns)
    df["Volatility_10d"] = df["Return_1d"].rolling(window=10).std()
    df["Volatility_21d"] = df["Return_1d"].rolling(window=21).std()

    # Target: will price go up tomorrow? (1 = up, 0 = down)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop rows with NaN from indicator warm-up periods
    df.dropna(inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FEATURES_FILE)
    print(f"Saved {len(df)} rows with {len(df.columns)} columns to {FEATURES_FILE}")
    return df


if __name__ == "__main__":
    build_features()
