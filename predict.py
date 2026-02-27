"""Generate predictions using the trained S&P 500 model."""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = "models"
DATA_DIR = "data"
MODEL_FILE = os.path.join(MODEL_DIR, "sp500_rf_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_FILE = os.path.join(DATA_DIR, "sp500_features.csv")


def load_model():
    """Load the trained model, scaler, and feature columns."""
    for path in [MODEL_FILE, SCALER_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run train_model.py first.")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, scaler, feature_cols


def predict(df=None):
    """Generate predictions on the feature dataset.

    Args:
        df: Optional DataFrame with features. If None, reads from FEATURES_FILE.

    Returns:
        DataFrame with predictions added.
    """
    model, scaler, feature_cols = load_model()

    if df is None:
        if not os.path.exists(FEATURES_FILE):
            raise FileNotFoundError(f"{FEATURES_FILE} not found. Run features.py first.")
        df = pd.read_csv(FEATURES_FILE, index_col="Date", parse_dates=True)

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    df["Prediction"] = model.predict(X_scaled)
    df["Pred_Probability"] = model.predict_proba(X_scaled)[:, 1]

    # Evaluate on the last 252 trading days (~1 year)
    recent = df.tail(252)
    accuracy = (recent["Prediction"] == recent["Target"]).mean()
    print(f"\nLast 252 trading days accuracy: {accuracy:.2%}")

    up_preds = recent[recent["Prediction"] == 1]
    down_preds = recent[recent["Prediction"] == 0]
    print(f"  Up predictions: {len(up_preds)} (correct: {(up_preds['Target'] == 1).mean():.2%})")
    print(f"  Down predictions: {len(down_preds)} (correct: {(down_preds['Target'] == 0).mean():.2%})")

    # Latest prediction
    last = df.iloc[-1]
    direction = "UP" if last["Prediction"] == 1 else "DOWN"
    confidence = last["Pred_Probability"] if last["Prediction"] == 1 else 1 - last["Pred_Probability"]
    print(f"\nLatest prediction: {direction} (confidence: {confidence:.1%})")

    # Plot recent predictions vs actual
    plot_predictions(recent)

    return df


def plot_predictions(df):
    """Plot recent predictions vs actual price movement."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price chart with prediction overlay
    axes[0].plot(df.index, df["Close"], color="black", linewidth=1, label="S&P 500 Close")
    correct = df[df["Prediction"] == df["Target"]]
    wrong = df[df["Prediction"] != df["Target"]]
    axes[0].scatter(correct.index, correct["Close"], c="green", s=10, alpha=0.5, label="Correct")
    axes[0].scatter(wrong.index, wrong["Close"], c="red", s=10, alpha=0.5, label="Wrong")
    axes[0].set_ylabel("Price")
    axes[0].set_title("S&P 500 Predictions vs Actual")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Prediction probability
    axes[1].bar(df.index, df["Pred_Probability"] - 0.5, color=np.where(df["Prediction"] == 1, "green", "red"), alpha=0.6)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Pred Probability - 0.5")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/predictions.png", dpi=150)
    print("Chart saved to output/predictions.png")
    plt.close()


if __name__ == "__main__":
    predict()
