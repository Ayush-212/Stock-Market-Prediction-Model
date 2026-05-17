import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from arch import arch_model

from market_data import STOCKS, fetch_data, get_latest_price
CAPITAL = 1000000
SEQ_LENGTH = 60


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def get_lstm_prediction(stock_series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_series.values.reshape(-1, 1))

    X, y = create_sequences(scaled_data, SEQ_LENGTH)
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

    last_seq = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    predicted_scaled = model.predict(last_seq)
    return scaler.inverse_transform(predicted_scaled)[0][0]


def get_volatility_weight(stock_series):
    returns = 100 * stock_series.pct_change().dropna()
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=1)
    tomorrow_vol = np.sqrt(forecast.variance.values[-1, :][0])
    return 1 / tomorrow_vol


def run_system():
    data = fetch_data()
    results = []
    total_inv_vol = 0

    for stock in STOCKS:
        print(f"\n--- Analyzing {stock} ---")
        try:
            current_price = get_latest_price(stock, data[stock])
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

    final_portfolio = []
    for res in results:
        weight = res['Inv_Vol'] / total_inv_vol
        shares = int((CAPITAL * weight) / res['Current'])

        final_portfolio.append([
            res['Stock'],
            round(weight * 100, 2),
            shares,
            res['Current'],
            res['Forecast']
        ])

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
