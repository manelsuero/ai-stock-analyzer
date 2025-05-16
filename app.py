import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Config y título ──────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

if st.sidebar.button("🔍 Analyze Stock"):

    # 1️⃣ Descargar datos
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data for “{ticker}”.")
        st.stop()

    # 2️⃣ Indicadores técnicos
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

    st.header("🔍 Technical Indicators")

    # ── RSI Chart ──────────────────────────────────────────────────────────────
    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['RSI'], label='RSI')
    ax.set_ylabel('RSI')
    ax.legend()
    st.pyplot(fig)

    # ── SMA Chart ──────────────────────────────────────────────────────────────
    st.subheader("SMA 20 over Close Price")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Close Price')
    ax.plot(df.index, df['SMA20'], label='SMA20')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # ── MACD Chart ─────────────────────────────────────────────────────────────
    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['MACD'], label='MACD')
    ax.plot(df.index, df['Signal Line'], label='Signal Line')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.info("✅ Technical indicators loaded. Next: Social Media Sentiment & News Analysis.")

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
