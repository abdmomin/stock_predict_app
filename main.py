import yfinance as yf
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date
import plotly.graph_objs as go

START = "2015-01-01"
END = date.today().strftime("%Y-%m-%d")

st.write(
    """
# Stock Price Forecast

Apple, Microsoft, Google, Netflix, and S&P 500 stock price forecast.
"""
)


stocks = ("AAPL", "MSFT", "GOOG", "NFLX", "^GSPC")

stock = st.selectbox(label="Select Stock", options=stocks)


@st.cache_data
def load_history(stock):
    ticker = yf.Ticker(stock)
    data = ticker.history(start=START, end=END)
    data.reset_index(inplace=True)
    return data


data_loader = st.text(body="Loading Stock History...")
data = load_history(stock=stock)
data_loader.text(body="Data Loaded.")
period = st.slider(label="How many days to forecast?", min_value=7, max_value=30)

st.subheader(body="Preview Data")
st.write(data.tail())
# print(data.dtypes)

st.write(f""" ### Visualize Close Price of {stock} """)


# def plot_raw_data():
#     fig = go.Figure()
#     fig.add_trace(go.Line(x=data["Date"], y=data["Close"], name="close_price"))
#     fig.layout.update(xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)


# plot_raw_data()

st.line_chart(data=data.Close)

#### Forecast time series data ####
data["Date"] = data["Date"].dt.date
train_data = data[["Date", "Close"]]
train_data.columns = ["ds", "y"]

print(train_data.head())
m = Prophet()
m.fit(train_data)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write(f""" ### Forecast Data for {period} days """)
st.write(forecast.tail(period))

st.write(f""" ### Visualize Future Stock Price of {stock} """)
forecast_fig = plot_plotly(
    m=m, fcst=forecast, figsize=(720, 500), xlabel="Date", ylabel="Close Price in USD"
)
st.plotly_chart(forecast_fig)

st.subheader(body="Component Plots")
comp_fig = m.plot_components(forecast)
st.pyplot(comp_fig)
