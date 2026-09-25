from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from data import get_stock_prices
from features import make_prediction_features


FEATURES = ["return_20d", "volatility_5d", "volatility_20d"]
MODEL_PATH = Path("models/volatility_model.joblib")


def parse_holdings(text):
    """Read one 'SYMBOL, SHARES' entry per line."""
    holdings = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Line {line_number}: enter a stock symbol and share count, separated by a comma."
            )

        symbol = parts[0].strip().upper()

        try:
            shares = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Line {line_number}: shares must be a number.")

        if not symbol or shares <= 0:
            raise ValueError(
                f"Line {line_number}: enter a symbol and a share count greater than zero."
            )

        # If a symbol appears twice, combine its shares.
        holdings[symbol] = holdings.get(symbol, 0) + shares

    if not holdings:
        raise ValueError("Enter at least one stock.")

    return holdings


st.title("Stock Portfolio Risk Analyzer")
st.write(
    "Enter your holdings to estimate each stock's volatility over the next "
    "20 trading days using a model trained on historical prices."
)

with st.form("portfolio_form"):
    entries = st.text_area(
        "Stocks and shares (one per line)",
        value="AAPL, 5\nMSFT, 3\nNVDA, 2",
        height=130,
    )
    submitted = st.form_submit_button("Analyze portfolio")

if submitted:
    try:
        holdings = parse_holdings(entries)

        if not MODEL_PATH.exists():
            raise ValueError("Trained model not found. Run python train.py first.")

        # Get recent prices and calculate the same inputs used in training.
        prices = get_stock_prices(list(holdings), period="1y")
        inputs = make_prediction_features(prices)

        model = joblib.load(MODEL_PATH)
        inputs["predicted_volatility"] = model.predict(inputs[FEATURES])

        # Estimate each position's value using its latest available price.
        inputs["shares"] = inputs["symbol"].map(holdings)
        inputs["latest_price"] = inputs["symbol"].map(prices.iloc[-1].to_dict())
        inputs["position_value"] = inputs["shares"] * inputs["latest_price"]

        display = inputs[
            ["symbol", "shares", "latest_price", "position_value", "predicted_volatility"]
        ].copy()
        display.columns = [
            "Stock", "Shares", "Latest price ($)",
            "Position value ($)", "Predicted volatility",
        ]
        display["Predicted volatility"] = (
            display["Predicted volatility"] * 100
        ).round(1).astype(str) + "%"

        st.subheader("Your holdings")
        st.dataframe(display, hide_index=True)
        st.metric("Estimated portfolio value", f"${inputs['position_value'].sum():,.2f}")
        st.caption(
            "Volatility measures expected price fluctuation, not whether a stock "
            "will go up or down. Predictions are annualized for comparison."
        )

    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not analyze the portfolio: {error}")