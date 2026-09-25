import numpy as np
import pandas as pd

from data import get_stock_prices


def make_training_data(prices):
    """Build past-looking inputs and future volatility targets."""
    all_stocks = []

    for symbol in prices.columns:
        # A daily return is the percentage change from the previous close.
        returns = prices[symbol].pct_change()

        stock_data = pd.DataFrame({
            "symbol": symbol,

            # Inputs: measurements available on the date of this row.
            "return_20d": prices[symbol].pct_change(20),
            "volatility_5d": returns.rolling(5).std() * np.sqrt(252),
            "volatility_20d": returns.rolling(20).std() * np.sqrt(252),

            # Target: volatility over the NEXT 20 trading days.
            # At today's row, shift(-20) brings the rolling calculation
            # from 20 days in the future back onto today's date.
            "future_volatility_20d": (
                returns.rolling(20).std().shift(-20) * np.sqrt(252)
            ),
        })

        # Early rows lack enough past data; final rows lack future data.
        all_stocks.append(stock_data.dropna())

    return pd.concat(all_stocks).reset_index(names="date")

def make_prediction_features(prices):
    """Build the three model inputs using each stock's latest available prices."""
    latest_rows = []

    for symbol in prices.columns:
        stock_prices = prices[symbol]
        returns = stock_prices.pct_change()

        latest_rows.append({
            "symbol": symbol,
            "return_20d": stock_prices.pct_change(20).iloc[-1],
            "volatility_5d": returns.rolling(5).std().iloc[-1] * np.sqrt(252),
            "volatility_20d": returns.rolling(20).std().iloc[-1] * np.sqrt(252),
        })

    # A stock needs at least 21 prices to calculate these inputs.
    result = pd.DataFrame(latest_rows)
    if result.isna().any().any():
        raise ValueError("Not enough recent price data for every stock.")

    return result

if __name__ == "__main__":
    prices = get_stock_prices(["AAPL", "MSFT", "NVDA"])
    training_data = make_training_data(prices)

    print(training_data.head().to_string(index=False))
    print(f"\nTraining rows: {len(training_data)}")
    print(f"Stocks: {training_data['symbol'].unique().tolist()}")
    print("\nInputs we can use for a prediction today:")
    print(make_prediction_features(prices).to_string(index=False))