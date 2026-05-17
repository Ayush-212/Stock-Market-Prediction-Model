
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from Model import STOCKS, fetch_data


def check_stationarity(stock_name, series):
    print(f"--- ADF Test for {stock_name} ---")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("Result: Stationary (No differencing needed)")
    else:
        print("Result: Non-Stationary (Differencing required for models like ARIMA)")


def plot_decomposition(stock_name, series):
    stl = STL(series.dropna(), period=252)
    res = stl.fit()
    res.plot()
    plt.suptitle(f"STL Decomposition - {stock_name}")
    plt.tight_layout()
    plt.show()


def run_other_models():
    data = fetch_data()

    print("\nRunning ADF stationarity checks...\n")
    for stock in STOCKS:
        if stock in data.columns:
            check_stationarity(stock, data[stock])
            print()

    target_stock = "BHARTIARTL.NS"
    if target_stock in data.columns:
        print(f"Plotting STL decomposition for {target_stock}...")
        plot_decomposition(target_stock, data[target_stock])
    else:
        print(f"{target_stock} not found in fetched data.")


if __name__ == "__main__":
    run_other_models()
