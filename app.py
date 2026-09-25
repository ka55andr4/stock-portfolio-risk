from pathlib import Path

import joblib
import numpy as np
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
                f"Line {line_number}: enter a stock symbol and share count, "
                "separated by a comma."
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

        # Combine shares if the same stock is entered on multiple lines.
        holdings[symbol] = holdings.get(symbol, 0) + shares

    if not holdings:
        raise ValueError("Enter at least one stock.")

    return holdings


st.title("Stock Portfolio Risk Analyzer")
st.write(
    "Enter your holdings to estimate volatility over the next 20 trading days "
    "using a model trained on historical stock prices."
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

        # Download recent prices for the stocks in this portfolio.
        prices = get_stock_prices(list(holdings), period="1y")

        # Calculate the three inputs the model learned from during training.
        inputs = make_prediction_features(prices)

        # Use the saved model to predict each stock's future volatility.
        model = joblib.load(MODEL_PATH)
        inputs["predicted_volatility"] = model.predict(inputs[FEATURES])

        # A position's value is its share count times its latest price.
        inputs["shares"] = inputs["symbol"].map(holdings)
        inputs["latest_price"] = inputs["symbol"].map(prices.iloc[-1].to_dict())
        inputs["position_value"] = inputs["shares"] * inputs["latest_price"]

        display = inputs[
            [
                "symbol",
                "shares",
                "latest_price",
                "position_value",
                "predicted_volatility",
            ]
        ].copy()
        display.columns = [
            "Stock",
            "Shares",
            "Latest price ($)",
            "Position value ($)",
            "Predicted volatility",
        ]
        # Format dollar amounts for display; calculations still use inputs.
        display["Latest price ($)"] = display["Latest price ($)"].map(
            lambda amount: f"${amount:,.2f}"
        )
        display["Position value ($)"] = display["Position value ($)"].map(
            lambda amount: f"${amount:,.2f}"
        )
        display["Predicted volatility"] = (
            display["Predicted volatility"] * 100
        ).round(1).astype(str) + "%"

        st.subheader("Your holdings")
        st.dataframe(display, hide_index=True)

        portfolio_value = inputs["position_value"].sum()
        st.metric("Estimated portfolio value", f"${portfolio_value:,.2f}")

        # A stock's weight is the fraction of the portfolio's value it represents.
        weights = inputs["position_value"].to_numpy() / portfolio_value

        # Correlation measures how closely the stocks moved together recently.
        # Use the same stock order as the weights and model predictions.
        recent_returns = prices.pct_change().tail(60)
        correlations = recent_returns.corr().loc[
            inputs["symbol"], inputs["symbol"]
        ].to_numpy()

        # Combine predicted stock volatility, portfolio weights, and correlations.
        stock_volatilities = inputs["predicted_volatility"].to_numpy()
        covariance = np.outer(stock_volatilities, stock_volatilities) * correlations
        portfolio_volatility = np.sqrt(weights @ covariance @ weights)

        st.metric(
            "Estimated portfolio volatility",
            f"{portfolio_volatility:.1%} annualized",
        )
        st.write(
            "This combines the model's estimates for each stock with how closely "
            "your stocks moved together over the last 60 trading days. A higher "
            "number means larger potential fluctuations; it does not tell us "
            "whether the portfolio will gain or lose value."
        )
        st.caption(
            "Volatility estimates are annualized for comparison, even though "
            "the model predicts the next 20 trading days."
        )

    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not analyze the portfolio: {error}")