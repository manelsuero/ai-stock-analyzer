import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ── 0. Configuración de página ────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── Sidebar (inputs comunes) ─────────────────────────────────────────────────
st.sidebar.header("Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",   value=pd.Timestamp.today())
st.sidebar.markdown("---")

# ── Section 1: Fundamental Analysis ────────────────────────────────────────────
st.sidebar.subheader("1️⃣ Technical Indicators")
if st.sidebar.button("🔍 Analyze Stock"):
    # 1️⃣ Descargar datos
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data for “{ticker}”.")
        st.stop()

    # 2️⃣ Calcular indicadores
    df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
    delta       = df["Close"].diff()
    gain        = delta.clip(lower=0)
    loss        = -delta.clip(upper=0)
    avg_gain    = gain.ewm(span=14, adjust=False).mean()
    avg_loss    = loss.ewm(span=14, adjust=False).mean()
    df["RSI"]   = 100 - (100 / (1 + avg_gain/avg_loss))
    ema12       = df["Close"].ewm(span=12, adjust=False).mean()
    ema26       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26
    df["Signal Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    st.subheader("🔍 Technical Indicators")

    # RSI
    st.markdown("**RSI (14 days)**")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["RSI"], label="RSI")
    ax.legend(); ax.set_ylabel("RSI")
    st.pyplot(fig)

    # SMA20
    st.markdown("**SMA20 over Close Price**")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["SMA20"], label="SMA20")
    ax.legend(); ax.set_ylabel("Price")
    st.pyplot(fig)

    # MACD
    st.markdown("**MACD & Signal Line**")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["Signal Line"], label="Signal Line")
    ax.legend(); ax.set_ylabel("Value")
    st.pyplot(fig)

    st.success("✅ Technical indicators loaded.")


    # ── Section 2: News Analysis via Finnhub ────────────────────────────────────
    st.markdown("---")
    st.header("📰 News Analysis")
    days_news = st.sidebar.slider("Days of news history", 1, 7, 3)
    max_news  = st.sidebar.slider("Max articles to fetch", 10, 100, 30)

    # leer clave secreta
    api_key = st.secrets.get("FINNHUB_KEY", "")
    if not api_key:
        st.error("🔑 Please set your FINNHUB_KEY in Streamlit Secrets.")
        st.stop()

    # endpoints de Finnhub: company-news
    from_str = (pd.Timestamp.today() - pd.Timedelta(days=days_news)).strftime("%Y-%m-%d")
    to_str   = pd.Timestamp.today().strftime("%Y-%m-%d")
    url = (
        "https://finnhub.io/api/v1/company-news?"
        f"symbol={ticker.upper()}&from={from_str}&to={to_str}&token={api_key}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        st.warning("⚠️ Error fetching news from Finnhub.")
    else:
        articles = resp.json()[:max_news]
        if not articles:
            st.info("No news found for that ticker.")
        else:
            # montar tabla
            news_df = pd.DataFrame([{
                "Date": pd.to_datetime(a["datetime"], unit="s").date(),
                "Headline": a["headline"],
                "Source": a.get("source", ""),
                "URL": a["url"]
            } for a in articles])
            st.table(news_df)

else:
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
