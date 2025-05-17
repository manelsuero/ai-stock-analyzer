import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ── 0. Configuración y título ────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Análisis Fundamental ───────────────────────────────────────────────────
st.sidebar.header("Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date", pd.Timestamp.today())

if st.sidebar.button("🔍 Analyze Stock"):

    # 1️⃣ Descargar datos
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data for “{ticker}”.")
        st.stop()

    # 2️⃣ Indicadores técnicos
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()

    delta     = df["Close"].diff()
    gain      = delta.where(delta > 0, 0)
    loss      = -delta.where(delta < 0, 0)
    avg_gain  = gain.ewm(span=14, adjust=False).mean()
    avg_loss  = loss.ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    ema12                = df["Close"].ewm(span=12, adjust=False).mean()
    ema26                = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]           = ema12 - ema26
    df["Signal Line"]    = df["MACD"].ewm(span=9, adjust=False).mean()

    st.header("🔍 Technical Indicators")

    # RSI Chart
    st.subheader("RSI (14 days)")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["RSI"], label="RSI")
    ax.set_ylabel("RSI")
    ax.legend()
    st.pyplot(fig)

    # SMA Chart
    st.subheader("SMA 20 over Close Price")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close Price")
    ax.plot(df.index, df["SMA20"], label="SMA20")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # MACD Chart
    st.subheader("MACD & Signal Line")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["Signal Line"], label="Signal Line")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.success("✅ Technical indicators loaded. Next: News Analysis.")

    # ── 2. News Analysis ────────────────────────────────────────────────────────
    st.header("📰 News Analysis")

    # Parámetros de noticias
    days_news   = st.sidebar.slider("Days of news history", 1, 7, 3)
    max_articles = st.sidebar.slider("Max articles to fetch", 10, 100, 30)

    # Función para obtener noticias (ejemplo sencillo con Finnhub)
    # NOTA: sustituye por tu API de noticias favorita
    def fetch_news(ticker, days, limit):
        api_key = st.secrets.get("FINNHUB_KEY", "")
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker.upper(),
            "from":   (pd.Timestamp.today() - pd.Timedelta(days=days)).date(),
            "to":     pd.Timestamp.today().date(),
            "token":  api_key
        }
        r = requests.get(url, params=params)
        data = r.json() if r.status_code == 200 else []
        # Tomamos solo título y fecha
        news = [{
            "datetime": pd.to_datetime(item["datetime"], unit="s"),
            "headline": item["headline"]
        } for item in data[:limit]]
        return pd.DataFrame(news)

    df_news = fetch_news(ticker, days_news, max_articles)
    if df_news.empty:
        st.warning("No news found for that ticker.")
    else:
        # Mostrar tabla de noticias
        st.subheader("Latest News")
        st.dataframe(df_news.sort_values("datetime", ascending=False).reset_index(drop=True))

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
