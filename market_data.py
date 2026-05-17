import datetime

import pandas as pd
import yfinance as yf


STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'SUNPHARMA.NS',
    'ITC.NS', 'BHARTIARTL.NS']

TARGET_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ITC.NS',
    'BHARTIARTL.NS', 'ICICIBANK.NS'
]
TRAIN_START = '2021-01-01'
TRAIN_END = '2025-06-30'
TEST_START = '2025-07-01'
TEST_END = '2025-12-31'
SEQ_LENGTH = 60
CAPITAL = 1000000


def fetch_data(start=TRAIN_START, end=None):
    print("Fetching historical data from Yahoo Finance...")
    end_date = end or datetime.datetime.now().strftime('%Y-%m-%d')
    raw = yf.download(
        STOCKS,
        start=start,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw, pd.DataFrame) and 'Close' in raw.columns:
        data = raw['Close']
    else:
        data = raw

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.sort_index().ffill().bfill()


def split_train_test(data, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END):
    train_data = data.loc[:train_end].copy().ffill().bfill()
    test_data = data.loc[test_start:test_end].copy().ffill().bfill()
    return train_data, test_data


def get_latest_price(stock_name, fallback_series=None):
    ticker = yf.Ticker(stock_name)

    try:
        intraday = ticker.history(period='1d', interval='1m', auto_adjust=True)
        if not intraday.empty:
            close_series = intraday['Close'].dropna()
            if not close_series.empty:
                return float(close_series.iloc[-1])
    except Exception:
        pass

    try:
        fast_info = ticker.fast_info
        if fast_info and fast_info.get('lastPrice') is not None:
            return float(fast_info['lastPrice'])
    except Exception:
        pass

    if fallback_series is not None:
        cleaned = fallback_series.dropna()
        if not cleaned.empty:
            return float(cleaned.iloc[-1])

    raise ValueError(f"Unable to determine latest price for {stock_name}")
