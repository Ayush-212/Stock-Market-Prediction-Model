import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from arch import arch_model
import datetime
import os

# --- Configuration ---
STOCKS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'SUNPHARMA.NS',
          'ITC.NS', 'BHARTIARTL.NS', 'LTTS.NS', 'ASIANPAINT.NS', 'ICICIBANK.NS']
CAPITAL = 1000000  # 10 Lakhs
TRAIN_START = "2021-01-01"
# Lookback period for sequences
SEQ_LENGTH = 60

# --- 1. Data Retrieval ---


def fetch_data():
    print("Fetching historical data from Yahoo Finance...")
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    # Use auto_adjust=True to handle dividends/splits like a pro
    df = yf.download(STOCKS, start=TRAIN_START,
                     end=today, auto_adjust=True)['Close']
    return df.ffill()

# --- 2. Sequence Creation ---


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# --- 3. LSTM Prediction Engine ---


def get_lstm_prediction(stock_series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_series.values.reshape(-1, 1))

    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Architecture optimized for NIT Capstone requirements
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Epochs set to 5 for speed; increase to 10-20 for higher accuracy
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict the next step using the last SEQ_LENGTH days
    last_seq = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    predicted_scaled = model.predict(last_seq)
    return scaler.inverse_transform(predicted_scaled)[0][0]

# --- 4. GARCH Volatility Weighting ---


def get_volatility_weight(stock_series):
    # Convert prices to percentage returns
    returns = 100 * stock_series.pct_change().dropna()
    # GARCH(1,1) is the standard for financial risk modeling
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=1)
    tomorrow_vol = np.sqrt(forecast.variance.values[-1, :][0])
    # Inverse volatility weighting: lower risk = higher weight
    return 1 / tomorrow_vol

# --- 5. Main Execution Loop ---


def run_system():
    data = fetch_data()
    results = []
    total_inv_vol = 0

    for stock in STOCKS:
        print(f"\n--- Analyzing {stock} ---")
        try:
            current_price = data[stock].iloc[-1]
            pred_price = get_lstm_prediction(data[stock])
            inv_vol = get_volatility_weight(data[stock])

            results.append({
                'Stock': stock,
                'Current': round(current_price, 2),
                'Forecast': round(pred_price, 2),
                'Inv_Vol': inv_vol
            })
            total_inv_vol += inv_vol
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")

    # --- 6. Portfolio Construction ---
    final_portfolio = []
    for res in results:
        # Weight based on inverse volatility
        weight = res['Inv_Vol'] / total_inv_vol
        # Calculate shares based on capital allocation
        shares = int((CAPITAL * weight) / res['Current'])

        final_portfolio.append([
            res['Stock'],
            round(weight * 100, 2),
            shares,
            res['Current'],
            res['Forecast']
        ])

    # Display Final Output Table
    df_final = pd.DataFrame(final_portfolio,
                            columns=['Stock', 'Weight%', 'Shares', 'Current', 'Forecast'])

    print("\n" + "="*50)
    print("FINAL PORTFOLIO EXECUTION PLAN")
    print("="*50)
    print(df_final.to_string(index=False))
    print("="*50)

    return df_final


if __name__ == "__main__":
    run_system()
