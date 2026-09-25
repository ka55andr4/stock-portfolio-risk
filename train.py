from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from data import get_stock_prices
from features import make_training_data


FEATURES = ["return_20d", "volatility_5d", "volatility_20d"]


def main():
    prices = get_stock_prices([
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "JPM", "BAC", "XOM", "CVX", "JNJ",
    "PFE", "WMT", "COST", "CAT", "DIS",
    ])
    dataset = make_training_data(prices)

    # Split by DATE, so all test examples come after the training examples.
    dates = sorted(dataset["date"].unique())
    split_at = int(len(dates) * 0.8)

    # Leave 20 trading days between training and testing.
    # This prevents a training row's future target from reaching into
    # the dates used to evaluate the model.
    training_dates = dates[: split_at - 20]
    test_dates = dates[split_at:]

    train = dataset[dataset["date"].isin(training_dates)]
    test = dataset[dataset["date"].isin(test_dates)]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    # X contains information known on each date; y contains the answer.
    model.fit(train[FEATURES], train["future_volatility_20d"])

    predictions = model.predict(test[FEATURES])

    # A baseline tells us whether training a model helped at all.
    baseline_predictions = test["volatility_20d"]

    model_error = mean_absolute_error(
        test["future_volatility_20d"], predictions
    )
    baseline_error = mean_absolute_error(
        test["future_volatility_20d"], baseline_predictions
    )

    print(f"Training examples: {len(train)}")
    print(f"Test examples: {len(test)}")
    print(f"Model average error: {model_error:.3f}")
    print(f"Baseline average error: {baseline_error:.3f}")

    # Save the trained model so the dashboard can load it later.
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/volatility_model.joblib")
    print("Saved model to models/volatility_model.joblib")


if __name__ == "__main__":
    main()