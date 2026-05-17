import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA

from market_data import STOCKS, fetch_data

# Configuration
TRAIN_END = '2025-06-30'
TEST_START = '2025-07-01'
TEST_END = '2025-12-31'
SEQ_LENGTH = 60
CAPITAL = 1000000


def train_lstm(train_series):
    """Train LSTM on training data and return model, scaler, and predictions on train."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LENGTH:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler, X, y


def forecast_lstm(model, scaler, test_series, seq_length=SEQ_LENGTH):
    """Forecast using trained LSTM on test series."""
    scaled_data = scaler.transform(test_series.values.reshape(-1, 1))
    predictions = []

    # Use the last seq_length values from training to start
    current_seq = scaled_data[:seq_length].copy().reshape(1, seq_length, 1)

    for i in range(len(test_series)):
        pred_scaled = model.predict(current_seq, verbose=0)
        pred_unscaled = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred_unscaled)

        # Update sequence with new prediction
        current_seq = np.append(current_seq[:, 1:, :],
                                scaler.transform([[pred_unscaled]]).reshape(1, 1, 1), axis=1)

    return np.array(predictions)


def forecast_arima(train_series, test_length, order=(5, 1, 2)):
    """Train ARIMA and forecast on test period."""
    try:
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=test_length)
        return forecast.values
    except Exception as e:
        print(f"ARIMA failed: {e}, returning zeros")
        return np.zeros(test_length)


def validate_models():
    """Main validation comparing LSTM vs ARIMA on July-Dec 2025 data."""
    print("=" * 70)
    print("VALIDATION REPORT: LSTM vs ARIMA (July–Dec 2025)")
    print("=" * 70)

    data = fetch_data()

    # Split data
    data_train = data[data.index <= TRAIN_END]
    data_test = data[(data.index >= TEST_START) & (data.index <= TEST_END)]

    print(
        f"\nTrain period: {data_train.index[0].date()} to {data_train.index[-1].date()}")
    print(
        f"Test period: {data_test.index[0].date()} to {data_test.index[-1].date()}")
    print(f"Test samples: {len(data_test)}")

    results_summary = []

    for stock in STOCKS:
        if stock not in data.columns:
            continue

        print(f"\n--- {stock} ---")

        train_series = data_train[stock].dropna()
        test_series = data_test[stock].dropna()

        if len(train_series) < SEQ_LENGTH + 10 or len(test_series) < 5:
            print(f"Insufficient data for {stock}")
            continue

        try:
            # Train LSTM
            lstm_model, scaler, _, _ = train_lstm(train_series)
            lstm_pred = forecast_lstm(lstm_model, scaler, test_series)

            # Train ARIMA
            arima_pred = forecast_arima(train_series, len(test_series))

            # Actual values
            actual = test_series.values

            # Align lengths
            min_len = min(len(lstm_pred), len(arima_pred), len(actual))
            lstm_pred = lstm_pred[:min_len]
            arima_pred = arima_pred[:min_len]
            actual = actual[:min_len]

            # Calculate metrics
            lstm_mae = mean_absolute_error(actual, lstm_pred)
            lstm_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))

            arima_mae = mean_absolute_error(actual, arima_pred)
            arima_rmse = np.sqrt(mean_squared_error(actual, arima_pred))

            # Direction accuracy (up/down)
            lstm_dir = np.mean(np.sign(np.diff(lstm_pred)) ==
                               np.sign(np.diff(actual))) * 100
            arima_dir = np.mean(np.sign(np.diff(arima_pred))
                                == np.sign(np.diff(actual))) * 100

            print(
                f"  LSTM  → MAE: ₹{lstm_mae:.2f}, RMSE: ₹{lstm_rmse:.2f}, Direction Acc: {lstm_dir:.1f}%")
            print(
                f"  ARIMA → MAE: ₹{arima_mae:.2f}, RMSE: ₹{arima_rmse:.2f}, Direction Acc: {arima_dir:.1f}%")

            winner = "LSTM" if lstm_mae < arima_mae else "ARIMA"
            print(f"  Winner: {winner} (lower MAE)")

            results_summary.append({
                'Stock': stock,
                'LSTM_MAE': lstm_mae,
                'LSTM_RMSE': lstm_rmse,
                'LSTM_Dir%': lstm_dir,
                'ARIMA_MAE': arima_mae,
                'ARIMA_RMSE': arima_rmse,
                'ARIMA_Dir%': arima_dir,
                'Winner': winner
            })

        except Exception as e:
            print(f"  Error: {e}")

    # Summary table
    if results_summary:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        df_summary = pd.DataFrame(results_summary)
        print(df_summary.to_string(index=False))

        lstm_wins = len([r for r in results_summary if r['Winner'] == 'LSTM'])
        arima_wins = len(results_summary) - lstm_wins
        print(f"\nLSTM wins: {lstm_wins}/{len(results_summary)}")
        print(f"ARIMA wins: {arima_wins}/{len(results_summary)}")


if __name__ == "__main__":
    validate_models()
