import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Load stock ticker data with an extended list
def load_stock_tickers():
    return [
        "AAPL", "TSLA", "GOOG", "AMZN", "MSFT", "FB", "NFLX", "NVDA", "BRK-B", "JPM",
        "V", "JNJ", "WMT", "PG", "MA", "DIS", "PYPL", "ADBE", "CMCSA", "NFLX",
        "PEP", "INTC", "T", "XOM", "CSCO", "VZ", "NKE", "CRM", "ABT", "CVX",
        "TMO", "MDT", "IBM", "HON", "QCOM", "LLY", "NEE", "TXN", "COST", "NOW",
        "AMGN", "SBUX", "INTU", "ISRG", "PM", "BKNG", "SYK", "GILD", "BSX", "LMT",
        "CAT", "BA", "MMM", "DHR", "TGT", "SPGI", "ATVI", "MDLZ", "VRTX", "EL",
        "ZTS", "CSX", "USB", "MS", "NEM", "TJX", "LRCX", "ADP", "HIG", "PGR",
        "FIS", "SYF", "MCO", "APD", "KMB", "DG", "WBA", "FISV", "ETR", "HST",
        "D", "ALGN", "PRGO", "CARR", "NTRS", "DLR", "SRE", "IDXX", "JCI", "K",
        "AES", "NDAQ", "WDC", "TTWO", "MTD", "SGEN", "KMX", "VFC", "VTRS", "PXD",
        "PSA", "CHRW", "DOV", "ODFL", "CNC", "WAB", "OMC", "NWL", "IPG", "DRE"
    ]

# Function to get stock data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker):
    stock_data = yf.download(ticker, start="2021-01-01", end="2025-01-01")
    return stock_data

# Function to prepare the data for training
def prepare_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Data is empty or None.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_train = np.array([scaled_data[i - 60:i, 0] for i in range(60, len(scaled_data))])
    y_train = scaled_data[60:, 0]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler

# Function to build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=False, input_shape=(60, 1)))  # Reduced units
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future prices
def predict_future_prices(model, last_60_days, scaler, num_days=30):
    predictions = []
    current_input = last_60_days.reshape(-1, 1)

    for _ in range(num_days):
        current_input_scaled = scaler.transform(current_input)
        X_test = np.array([current_input_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_price = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price)
        predictions.append(predicted_price[0][0])

        current_input = np.append(current_input[1:], predicted_price).reshape(-1, 1)

    return predictions

# Streamlit App
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")  # Set wider layout
st.title("Stock Price Prediction Dashboard")
st.sidebar.header("Settings")

# Autocomplete for stock ticker
stock_tickers = load_stock_tickers()
stock_ticker = st.sidebar.selectbox("Select Stock Ticker", stock_tickers)

if st.sidebar.button("Predict"):
    with st.spinner("Fetching data and training model..."):
        try:
            stock_data = get_stock_data(stock_ticker)

            # Check if stock_data is valid
            if stock_data.empty:
                st.error("No data found for the selected ticker.")
                st.stop()

            closing_prices = stock_data["Close"].values.reshape(-1, 1)

            # Check if closing_prices are valid
            if len(closing_prices) < 60:
                st.error("Not enough data to make predictions. Please select a different ticker.")
                st.stop()

            x_train, y_train, scaler = prepare_data(closing_prices)

            model = build_model()
            model.fit(x_train, y_train, batch_size=8, epochs=10)  # Reduced epochs and batch size

            last_60_days = closing_prices[-60:]
            future_predictions = predict_future_prices(model, last_60_days, scaler)

            future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=30)

            # Use Streamlit to plot
            plt.figure(figsize=(14, 7))
            plt.plot(stock_data.index, closing_prices, color="blue", label="Actual Price")
            plt.plot(future_dates, future_predictions, color="orange", label="Predicted Price")
            plt.title(f"Stock Price Prediction for {stock_ticker}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.axvline(x=stock_data.index[-1], color="red", linestyle="--", label="Prediction Start")
            plt.legend()
            st.pyplot(plt)

            st.write(f"The predicted price for **{stock_ticker}** for tomorrow is: **${future_predictions[0]:.2f}**")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.sidebar.markdown("### About")
st.sidebar.write("This dashboard uses an enhanced LSTM model to predict stock prices based on historical data.")
st.sidebar.write("Select a stock ticker and click 'Predict' to see the predictions.")
st.sidebar.write("Data is fetched from Yahoo Finance.")
