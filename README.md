# Time Series Analysis Capstone Project: Portfolio Allocation

A Python-based portfolio optimization system using LSTM neural networks, ARIMA baseline models, and GARCH volatility weighting for Indian stock market analysis.

## 📋 Project Overview

This project implements a **strict train/test validation framework** for forecasting 9 Indian NSE stocks (RELIANCE, TCS, HDFCBANK, SUNPHARMA, ITC, BHARTIARTL, LTTS, ASIANPAINT, ICICIBANK) using:

1. **LSTM (Stacked Long Short-Term Memory)** - Deep learning time series forecaster
2. **ARIMA(5,1,0)** - Baseline statistical model for comparison
3. **GARCH(1,1)** - Volatility forecasting for portfolio weighting
4. **ADF Test** - Stationarity diagnostics
5. **STL Decomposition** - Seasonal-Trend analysis

**Capital Limit**: ₹10,00,000 (10 lakh rupees)

---

## 📁 File Structure

### `market_data.py` (Data Pipeline)
Central module for all data operations. Implements the strict train/test split as required by the academic rubric.

**Key Components:**
- `STOCKS` - List of 9 NSE tickers
- `TARGET_STOCKS` - Subset of 6 stocks for ADF analysis
- `TRAIN_START = '2021-01-01'`
- `TRAIN_END = '2025-06-30'`
- `TEST_START = '2025-07-01'` 
- `TEST_END = '2025-12-31'`

**Functions:**
- `fetch_data(start, end)` - Downloads Yahoo Finance close prices with auto-adjustment and forward-fill handling
- `split_train_test(data)` - Splits data into training (Jan 2021 – Jun 2025) and test (Jul–Dec 2025) periods
- `get_latest_price(stock_name, fallback_series)` - Real-time price with 3-tier fallback strategy

---

### `Model.py` (Main Orchestrator)
Executable portfolio engine. Run this to validate, forecast, and allocate capital.

**Workflow:**
1. **Fetch Data** - Download 5+ years of historical prices
2. **Validate Models** - Compare LSTM vs ARIMA on the explicit test period
3. **Calculate Metrics** - MAPE, RMSE, Directional Accuracy for each model
4. **Forecast** - Next 2 trading days using LSTM
5. **Weight Portfolio** - Inverse-volatility allocation using GARCH
6. **Print Allocation Table** - Stock, Current Price, Forecasted Price

**Key Functions:**
- `train_lstm(series)` - Train stacked LSTM (50→50 neurons, 2 Dropout layers)
- `forecast_lstm(model, scaler, series, steps)` - Rolling forecast for n steps ahead
- `forecast_arima(series, steps, order=(5,1,0))` - ARIMA baseline
- `calculate_metrics(actual, predicted)` - MAPE%, RMSE, Direction Accuracy%
- `validate_models(data)` - Print comparison table
- `get_volatility_weight(series)` - GARCH(1,1) inverse-volatility weighting
- `run_system(data)` - Execute full portfolio pipeline

**Architecture (LSTM):**
```
Input(60×1) 
  → LSTM(50, return_seq=True) 
  → Dropout(0.2) 
  → LSTM(50, return_seq=False) 
  → Dropout(0.2) 
  → Dense(1 output)

Training: 5 epochs, batch_size=32, Adam optimizer, MSE loss
```

---

### `otherModels.py` (Preprocessing & Diagnostics)
Time series analysis and stationarity testing.

**Functions:**
- `check_stationarity(stock_name, series)` - ADF test (p-value ≤ 0.05 = stationary)
- `plot_decomposition(stock_name, series)` - STL decomposition with 252-day seasonal period
- `run_other_models()` - Execute ADF for all target stocks, STL plot for BHARTIARTL.NS

**Output:**
- Terminal: ADF statistics and p-values
- File: `bhartiartl_ns_stl.png` - Seasonal-Trend-Residual plot

---

## 🚀 How to Run

### Prerequisites
Ensure your Python environment is configured:
```bash
pip install tensorflow pandas numpy scikit-learn yfinance statsmodels arch
```

### Execute Main Pipeline
```bash
python Model.py
```

**Output:**
1. Data fetch confirmation
2. **VALIDATION REPORT** table:
   | Stock | LSTM_MAPE | LSTM_RMSE | LSTM_Dir% | ARIMA_MAPE | ARIMA_RMSE | ARIMA_Dir% | Winner |
3. Win summary (LSTM vs ARIMA count)
4. Per-stock analysis (current price, forecast, volatility weight)
5. **FINAL PORTFOLIO ALLOCATION** table:
   | Stock | Final Weight (%) | Number of Shares | Current Price | Forecasted Price |

### Run Time Series Analysis
```bash
python otherModels.py
```

**Output:**
- ADF test results for all 6 target stocks
- `bhartiartl_ns_stl.png` plot (if 252+ observations available)

---

## 📊 Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `SEQ_LENGTH` | 60 days | LSTM lookback window |
| `CAPITAL` | ₹10,00,000 | Portfolio investment amount |
| `LSTM epochs` | 5 | Training iterations |
| `LSTM batch_size` | 32 | Samples per update |
| `GARCH(p,q)` | (1,1) | Volatility model order |
| `ARIMA order` | (5,1,0) | Baseline AR-I-MA configuration |
| `STL period` | 252 days | Trading days per year |

---

## 📈 Output Explained

### Validation Table Columns
- **LSTM_MAPE** - Mean Absolute Percentage Error (%)
- **LSTM_RMSE** - Root Mean Squared Error (₹)
- **LSTM_Dir%** - Directional accuracy (% correct up/down predictions)
- **ARIMA_MAPE, ARIMA_RMSE, ARIMA_Dir%** - Same metrics for baseline
- **Winner** - Model with lower MAPE on the test period

### Portfolio Allocation Columns
- **Stock** - NSE ticker name
- **Final Weight (%)** - Inverse-volatility normalized portfolio weight
- **Number of Shares** - Integer shares sized using ₹10,00,000 capital limit
- **Current Price** - Real-time price (3-tier fallback: intraday → fast_info → historical)
- **Forecasted Price** - LSTM prediction for next trading day

### Allocation Strategy
1. Train LSTM + GARCH on full history
2. Forecast next 2 trading days
3. Calculate volatility (GARCH)
4. Weight inversely: `weight = 1/volatility / Σ(1/volatility_all)`
5. Allocate capital proportionally

---

## 🔧 Data Handling

- **Missing Data**: Forward-fill then backward-fill (`ffill().bfill()`)
- **Price Adjustment**: Yahoo Finance `auto_adjust=True` (splits/dividends included)
- **Series Validation**: Drops NaN, checks minimum length thresholds
- **Scaler**: MinMaxScaler (0,1) for LSTM training only

---

## ✅ Rubric Compliance

- **Task 2 & 3 (Train/Test Split)**: ✓ Strict Jan 2021 – Jun 2025 train; Jul–Dec 2025 test
- **Task 4 (Preprocessing)**: ✓ ADF stationarity tests, STL decomposition
- **Task 5 (Portfolio Sizing)**: ✓ GARCH weighting, ₹10L capital allocation
- **Task 6 & 8 (Baseline & Metrics)**: ✓ ARIMA(5,1,0), MAPE/RMSE/Direction Accuracy
- **Task 2 (Data Pipeline)**: ✓ Centralized market_data.py with ffill/bfill

---

## 📝 Notes

- The test period (Jul–Dec 2025) is **explicitly hard-coded** in `market_data.py` for reproducibility
- LSTM is retrained on each run; results vary slightly due to random initialization
- GARCH may warn if volatility is very low; falls back to std deviation
- STL decomposition requires ≥252 observations (skipped otherwise)
- Real-time price fetches may fail in offline environments; uses historical fallback

---

## 📦 Dependencies

```
tensorflow>=2.21.0
pandas>=3.0.3
numpy>=2.4.5
scikit-learn>=1.8.0
yfinance>=1.3.0
statsmodels>=0.14.6
arch>=8.0.0
```

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'tensorflow'` | `pip install tensorflow` |
| `yfinance: No data found` | Check ticker format (use `.NS` for NSE) |
| `GARCH fitting error` | Reduce returns smoothing or skip stock |
| `STL insufficient data` | Need ≥252 trading days (about 1 year) |
| `Real-time price fetch timeout` | Script falls back to historical close |

---

## 🎓 Academic Context

This project demonstrates:
- Time series forecasting with deep learning (LSTM)
- Statistical baseline comparison (ARIMA)
- Volatility modeling (GARCH)
- Portfolio optimization principles
- Proper train/test validation methodology
- Handling real market data with YFinance API

**Capstone Focus**: Rigorous train/test separation with explicit dates, side-by-side model evaluation, and production-grade error handling.
