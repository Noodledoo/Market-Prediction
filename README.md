# S&P 500 Market Prediction

ML model for predicting S&P 500 price movements using historical data and technical indicators.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline: fetch data, train model, predict
python main.py

# Individual steps
python fetch_data.py          # Download S&P 500 historical data
python features.py            # Generate features (requires data)
python train_model.py         # Train the model (requires features)
python predict.py             # Run predictions (requires trained model)
```

## Project Structure

- `fetch_data.py` - Downloads S&P 500 data from Yahoo Finance
- `features.py` - Feature engineering with technical indicators
- `train_model.py` - Trains a Random Forest classifier
- `predict.py` - Generates predictions using the trained model
- `main.py` - Orchestrates the full pipeline
