import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

st.sidebar.header("Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

if st.sidebar.button("🔍 Analyze Stock"):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data found for “{ticker}”.")
        st.stop()

    # --- Calculations with min_periods=1 ---
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain/avg_loss)))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- Show charts ---
    st.header("🔍 Technical Indicators")

    st.subheader("RSI (14 days)")
    st.line_chart(df['RSI'])

    st.subheader("SMA 20 over Close Price")
    # Preview to confirm column exists
    st.write(df[['Close','SMA20']].head(5))
    st.line_chart(df[['Close', 'SMA20']])

    st.subheader("MACD & Signal Line")
    st.line_chart(df[['MACD', 'Signal Line']])
else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
