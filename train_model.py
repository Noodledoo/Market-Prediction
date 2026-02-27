"""Train a Random Forest model to predict S&P 500 direction."""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
FEATURES_FILE = os.path.join(DATA_DIR, "sp500_features.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "sp500_rf_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# Features to use for training (exclude raw price/volume and target)
EXCLUDE_COLS = ["Open", "High", "Low", "Close", "Volume", "Target",
                "SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
                "BB_Upper", "BB_Lower", "Volume_SMA20", "ATR_14"]


def get_feature_columns(df):
    """Return list of feature column names."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def train_model(df=None):
    """Train Random Forest classifier with time-series cross-validation.

    Args:
        df: Optional DataFrame with features. If None, reads from FEATURES_FILE.

    Returns:
        Tuple of (trained model, scaler, feature columns).
    """
    if df is None:
        if not os.path.exists(FEATURES_FILE):
            raise FileNotFoundError(f"{FEATURES_FILE} not found. Run features.py first.")
        df = pd.read_csv(FEATURES_FILE, index_col="Date", parse_dates=True)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["Target"].values

    print(f"Training on {len(X)} samples with {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print(f"Class balance: {np.mean(y):.2%} positive (up days)")

    # Scale features
    scaler = StandardScaler()

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        score = accuracy_score(y_val, val_pred)
        cv_scores.append(score)
        print(f"  Fold {fold + 1}: accuracy = {score:.4f}")

    print(f"  Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final model: train on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    # Feature importance
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 10 feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name:25s} {imp:.4f}")

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")

    return model, scaler, feature_cols


if __name__ == "__main__":
    train_model()
