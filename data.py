import pandas as pd
import yfinance as yf


def get_stock_prices(symbols, period="5y"):
    """Return daily closing prices for the requested stock symbols."""
    prices = {}

    for symbol in symbols:
        # Ask Yahoo Finance for this stock's daily price history.
        history = yf.Ticker(symbol).history(
            period=period,
            interval="1d",
            auto_adjust=True,
        )

        # An empty result usually means the symbol is invalid or data is unavailable.
        if history.empty:
            raise ValueError(f"No historical prices found for {symbol}.")

        # Keep the closing price and label its column with the stock symbol.
        prices[symbol] = history["Close"]

    # Put every stock's prices into one table, aligned by trading date.
    price_table = pd.DataFrame(prices)

    # Keep dates where we have a price for every requested stock.
    return price_table.dropna()


if __name__ == "__main__":
    # These are temporary example stocks so we can inspect the data.
    symbols = ["AAPL", "MSFT", "NVDA"]
    prices = get_stock_prices(symbols)

    print("First five trading days:")
    print(prices.head())
    print("\nMost recent five trading days:")
    print(prices.tail())
    print(f"\nRows of data: {len(prices)}")