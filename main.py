"""Main pipeline: fetch data, build features, train model, and predict."""

import argparse
from fetch_data import fetch_sp500_data
from features import build_features
from train_model import train_model
from predict import predict


def run_pipeline(period="5y", skip_fetch=False):
    """Run the full S&P 500 prediction pipeline.

    Args:
        period: Historical data period (e.g. "1y", "2y", "5y", "10y").
        skip_fetch: If True, skip data download and use existing data.
    """
    print("=" * 60)
    print("S&P 500 Market Prediction Pipeline")
    print("=" * 60)

    # Step 1: Fetch data
    if not skip_fetch:
        print("\n[1/4] Fetching data...")
        df = fetch_sp500_data(period=period)
    else:
        print("\n[1/4] Skipping data fetch (using existing data)")
        df = None

    # Step 2: Build features
    print("\n[2/4] Building features...")
    df = build_features(df)

    # Step 3: Train model
    print("\n[3/4] Training model...")
    train_model(df)

    # Step 4: Predict
    print("\n[4/4] Generating predictions...")
    predict(df)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 Market Prediction")
    parser.add_argument("--period", default="5y", help="Data period (default: 5y)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data download")
    args = parser.parse_args()

    run_pipeline(period=args.period, skip_fetch=args.skip_fetch)
