import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

from market_data import (
    CAPITAL,
    SEQ_LENGTH,
    STOCKS,
    fetch_data,
    get_latest_price,
    split_train_test,
)


def create_sequences(data, seq_length):
    x_values, y_values = [], []
    for index in range(seq_length, len(data)):
        x_values.append(data[index - seq_length: index])
        y_values.append(data[index])
    return np.array(x_values), np.array(y_values)


def build_lstm_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(stock_series, epochs=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_series.values.reshape(-1, 1))

    x_values, y_values = create_sequences(scaled_data, SEQ_LENGTH)
    if len(x_values) == 0:
        raise ValueError("Not enough data to train LSTM")

    x_values = np.reshape(x_values, (x_values.shape[0], x_values.shape[1], 1))
    model = build_lstm_model((x_values.shape[1], 1))
    model.fit(x_values, y_values, epochs=epochs, batch_size=32, verbose=0)
    return model, scaler


def forecast_lstm(model, scaler, history_series, steps):
    history = history_series.dropna().astype(float).values.reshape(-1, 1)
    if len(history) < SEQ_LENGTH:
        raise ValueError("Not enough data to forecast with LSTM")

    scaled_history = scaler.transform(history)
    sequence = scaled_history[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    forecasts = []

    for _ in range(steps):
        predicted_scaled = model.predict(sequence, verbose=0)
        predicted_value = scaler.inverse_transform(predicted_scaled)[0][0]
        forecasts.append(float(predicted_value))

        next_scaled = scaler.transform(np.array([[predicted_value]]))
        sequence = np.concatenate(
            (sequence[:, 1:, :], next_scaled.reshape(1, 1, 1)), axis=1)

    return np.array(forecasts)


def forecast_arima(train_series, steps, order=(5, 1, 0)):
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return np.asarray(model_fit.forecast(steps=steps))


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted series must have the same length")

    valid_mask = np.abs(actual) > 1e-8
    if np.any(valid_mask):
        mape = float(np.mean(np.abs(
            (actual[valid_mask] - predicted[valid_mask]) / actual[valid_mask])) * 100)
    else:
        mape = np.nan

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    if len(actual) > 1:
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        directional_accuracy = float(
            np.mean(actual_direction == predicted_direction) * 100)
    else:
        directional_accuracy = np.nan

    return mape, rmse, directional_accuracy


def get_volatility_weight(stock_series):
    returns = 100 * stock_series.pct_change().dropna()
    if len(returns) < 30:
        raise ValueError("Not enough returns to fit GARCH")

    try:
        model = arch_model(returns, vol="Garch", p=1, q=1)
        fitted = model.fit(disp="off")
        forecast = fitted.forecast(horizon=1)
        next_volatility = float(np.sqrt(forecast.variance.values[-1, :][0]))
    except Exception:
        next_volatility = float(returns.std())

    if not np.isfinite(next_volatility) or next_volatility <= 0:
        next_volatility = float(returns.std())
        if not np.isfinite(next_volatility) or next_volatility <= 0:
            next_volatility = 1.0

    return 1 / next_volatility


def validate_models(data=None):
    if data is None:
        data = fetch_data()

    train_data, test_data = split_train_test(data)

    if train_data.empty or test_data.empty:
        raise ValueError("Train/test split produced empty data")

    print("\n" + "=" * 70)
    print("VALIDATION REPORT: LSTM vs ARIMA (July–Dec 2025)")
    print("=" * 70)
    print(
        f"Train period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    print(
        f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"Test samples: {len(test_data)}")

    results = []
    total_inv_vol = 0.0

    for stock in STOCKS:
        if stock not in train_data.columns or stock not in test_data.columns:
            continue

        train_series = train_data[stock].dropna()
        test_series = test_data[stock].dropna()

        print(f"\n--- {stock} ---")
        if len(train_series) < SEQ_LENGTH + 10 or len(test_series) < 5:
            print("Insufficient data for validation")
            continue

        try:
            lstm_model, lstm_scaler = train_lstm(train_series)
            lstm_pred = forecast_lstm(
                lstm_model, lstm_scaler, train_series, len(test_series))
            arima_pred = forecast_arima(
                train_series, len(test_series), order=(5, 1, 0))

            actual = test_series.values
            horizon = min(len(actual), len(lstm_pred), len(arima_pred))
            actual = actual[:horizon]
            lstm_pred = lstm_pred[:horizon]
            arima_pred = arima_pred[:horizon]

            lstm_mape, lstm_rmse, lstm_dir = calculate_metrics(
                actual, lstm_pred)
            arima_mape, arima_rmse, arima_dir = calculate_metrics(
                actual, arima_pred)

            print(
                f"  LSTM  -> MAPE: {lstm_mape:.2f}%, RMSE: ₹{lstm_rmse:.2f}, Direction: {lstm_dir:.1f}%")
            print(
                f"  ARIMA -> MAPE: {arima_mape:.2f}%, RMSE: ₹{arima_rmse:.2f}, Direction: {arima_dir:.1f}%")

            winner = "LSTM" if lstm_mape < arima_mape else "ARIMA"
            print(f"  Winner: {winner}")

            results.append(
                {
                    "Stock": stock,
                    "LSTM_MAPE": lstm_mape,
                    "LSTM_RMSE": lstm_rmse,
                    "LSTM_Dir%": lstm_dir,
                    "ARIMA_MAPE": arima_mape,
                    "ARIMA_RMSE": arima_rmse,
                    "ARIMA_Dir%": arima_dir,
                    "Winner": winner,
                }
            )
        except Exception as exc:
            print(f"  Error: {exc}")

    if results:
        summary = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print("COMPARATIVE EVALUATION TABLE")
        print("=" * 70)
        print(
            summary.to_string(
                index=False,
                formatters={
                    "LSTM_MAPE": "{:.2f}".format,
                    "LSTM_RMSE": "{:.2f}".format,
                    "LSTM_Dir%": "{:.1f}".format,
                    "ARIMA_MAPE": "{:.2f}".format,
                    "ARIMA_RMSE": "{:.2f}".format,
                    "ARIMA_Dir%": "{:.1f}".format,
                },
            )
        )

        lstm_wins = int((summary["Winner"] == "LSTM").sum())
        arima_wins = int((summary["Winner"] == "ARIMA").sum())
        print(f"\nLSTM wins: {lstm_wins}/{len(summary)}")
        print(f"ARIMA wins: {arima_wins}/{len(summary)}")

    return results


def run_system(data=None):
    if data is None:
        data = fetch_data()

    validate_models(data)

    results = []

    for stock in STOCKS:
        print(f"\n--- Analyzing {stock} ---")
        try:
            stock_series = data[stock].dropna()
            current_price = get_latest_price(stock, stock_series)

            lstm_model, scaler = train_lstm(stock_series)
            forecast_prices = forecast_lstm(
                lstm_model, scaler, stock_series, steps=2)
            forecast_price = float(forecast_prices[-1])
            inv_vol = get_volatility_weight(stock_series)

            results.append(
                {
                    "Stock": stock,
                    "Current": round(current_price, 2),
                    "Forecast": round(forecast_price, 2),
                    "Inv_Vol": inv_vol,
                }
            )
            total_inv_vol += inv_vol
        except Exception as exc:
            print(f"Error analyzing {stock}: {exc}")

    final_portfolio = []
    for result in results:
        weight = result["Inv_Vol"] / total_inv_vol if total_inv_vol else 0.0
        shares = int((CAPITAL * weight) /
                     result["Current"]) if result["Current"] else 0

        final_portfolio.append(
            [
                result["Stock"],
                round(weight * 100, 2),
                shares,
                result["Current"],
                result["Forecast"],
            ]
        )

    df_final = pd.DataFrame(
        final_portfolio,
        columns=[
            "Stock", "Final Weight (%)", "Number of Shares", "Current Price", "Forecasted Price"],
    )

    print("\n" + "=" * 70)
    print("FINAL PORTFOLIO ALLOCATION")
    print("=" * 70)
    print(df_final.to_string(index=False))
    print("=" * 70)

    return df_final


if __name__ == "__main__":
    run_system()
