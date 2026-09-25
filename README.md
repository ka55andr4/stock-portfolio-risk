# Stock Portfolio Risk Analyzer

A Python dashboard that estimates stock and portfolio volatility. Users enter stock symbols and share counts; the app downloads recent prices, uses a trained machine-learning model to estimate each stock's volatility over the next 20 trading days, and combines those estimates with recent stock correlations to estimate portfolio volatility.

## Features

- Enter multiple stock symbols and share counts.
- View estimated position values and model-predicted volatility for each stock.
- View an estimated portfolio value and volatility that accounts for position sizes and recent correlations.
- Train the model locally and compare it against a simple baseline.

## How the model works

`data.py` downloads historical daily stock prices with yfinance. `features.py` creates three inputs for each stock and date: its previous 20-day return, previous 5-day volatility, and previous 20-day volatility. The training target is its volatility over the following 20 trading days.

`train.py` trains a scikit-learn random forest on historical examples from 15 stocks. It evaluates the model on later dates, with a 20-trading-day gap between training and testing to keep future target periods out of the test period.

On the test period used during development, the model's mean absolute error was **0.069**, compared with **0.077** for a baseline that uses the previous 20-day volatility as its prediction. These errors are measured in annualized volatility units. Results can change when the data or evaluation period changes.

The dashboard loads the saved model from `models/volatility_model.joblib`. To estimate portfolio volatility, it combines the stock predictions using each holding's share of portfolio value and correlations calculated from the most recent 60 trading days.

## Run locally

Tested with Python 3.12. In Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal. Enter one holding per line, such as `AAPL, 5`, and select **Analyze portfolio**.

To retrain the model, stop the dashboard with Ctrl+C and run:

```cmd
python train.py
```

## Scope

Volatility estimates describe the size of potential price fluctuations, not the direction of returns or a guaranteed outcome. The model was evaluated using examples from its 15 training stocks; predictions for other symbols have not been separately validated. Downloading stock data requires an internet connection.