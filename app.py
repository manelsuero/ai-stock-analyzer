import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📈 AI Stock Analyzer")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Options")
ticker = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

# ── Run analysis when button clicked ─────────────────────────────────────────
if st.sidebar.button("🔍 Analyze Stock"):

    # 1️⃣ Download price data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data found for “{ticker}”. Check the ticker symbol.")
    else:
        # ── Manual calculation of technical indicators ──────────────────────

        # 2️⃣ SMA (20)
        df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()

        # 3️⃣ RSI (14)
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0)
        loss  = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 4️⃣ MACD (12,26,9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']        = ema12 - ema26
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ── Display charts ────────────────────────────────────────────────────

        st.header("🔍 Technical Indicators")

        st.subheader("RSI (14 days)")
        st.line_chart(df['RSI'])

        st.subheader("SMA 20 over Close Price")
        st.line_chart(df[['Close', 'SMA20']])

        st.subheader("MACD & Signal Line")
        st.line_chart(df[['MACD', 'Signal Line']])

        # ── Placeholder for next phases ──────────────────────────────────────
        st.markdown("---")
        st.info("✅ Technical indicators loaded. Next: Social Media Sentiment & News Analysis.")

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to run the analysis.")
