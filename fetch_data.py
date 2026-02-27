"""Fetch S&P 500 historical data from Yahoo Finance."""

import os
import yfinance as yf
import pandas as pd

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "sp500_raw.csv")


def fetch_sp500_data(period="5y"):
    """Download S&P 500 (^GSPC) historical data.

    Args:
        period: How far back to fetch. Default 5 years.

    Returns:
        DataFrame with OHLCV data.
    """
    print(f"Fetching S&P 500 data for the last {period}...")
    ticker = yf.Ticker("^GSPC")
    df = ticker.history(period=period)

    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check your connection.")

    # Clean up columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE)
    print(f"Saved {len(df)} rows to {DATA_FILE}")
    return df


if __name__ == "__main__":
    fetch_sp500_data()
