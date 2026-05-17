from market_data import STOCKS, fetch_data
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


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
    fig = res.plot()
    fig.suptitle(f"STL Decomposition - {stock_name}")
    fig.tight_layout()
    output_path = f"{stock_name.replace('.', '_').lower()}_stl.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved STL decomposition plot to {output_path}")


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
