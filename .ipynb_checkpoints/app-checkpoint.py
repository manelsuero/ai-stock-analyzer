import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer with AI")

# Sidebar inputs
st.sidebar.header("Options")
ticker = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())

# Run analysis when button is clicked
if st.sidebar.button("🔍 Analyze Stock"):

    # ─── Technical Indicators (from your Jupyter prototype) ──────────
    st.header("📊 Technical Indicators")

    # 1️⃣ Descarga de datos (último mes, diario)
    df = yf.download(ticker, period="1mo", interval="1d")

    if df.empty:
        st.error(f"No data found for {ticker}. Check the ticker symbol.")
    else:
        # 2️⃣ Cálculo de tus indicadores
        df["SMA20"] = ta.sma(df["Close"], length=20)
        df["RSI"]   = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        df["MACD"]   = macd["MACD_12_26_9"]
        df["Signal"] = macd["MACDs_12_26_9"]

        # 3️⃣ Mostrar en Streamlit
        st.subheader("RSI (14 days)")
        st.line_chart(df["RSI"])

        st.subheader("SMA 20 over Close Price")
        st.line_chart(df[["Close", "SMA20"]])

        st.subheader("MACD & Signal Line")
        st.line_chart(df[["MACD", "Signal"]])

else:
    st.info("👈 Enter a stock ticker and click 'Analyze Stock' to begin.")
