import datetime

import yfinance as yf


STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'SUNPHARMA.NS',
    'ITC.NS', 'BHARTIARTL.NS', 'LTTS.NS', 'ASIANPAINT.NS', 'ICICIBANK.NS'
]
TRAIN_START = '2021-01-01'


def fetch_data():
    print("Fetching historical data from Yahoo Finance...")
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    df = yf.download(STOCKS, start=TRAIN_START,
                     end=today, auto_adjust=True)['Close']
    return df.ffill()


def get_latest_price(stock_name, fallback_series=None):
    ticker = yf.Ticker(stock_name)

    try:
        intraday = ticker.history(period='1d', interval='1m', auto_adjust=True)
        if not intraday.empty:
            return float(intraday['Close'].dropna().iloc[-1])
    except Exception:
        pass

    try:
        fast_info = ticker.fast_info
        if fast_info and fast_info.get('lastPrice') is not None:
            return float(fast_info['lastPrice'])
    except Exception:
        pass

    if fallback_series is not None and not fallback_series.dropna().empty:
        return float(fallback_series.dropna().iloc[-1])

    raise ValueError(f"Unable to determine latest price for {stock_name}")
