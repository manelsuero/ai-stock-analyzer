import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
import os

# ── 0. Configuración inicial ─────────────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# ── 1. Sidebar / Inputs ───────────────────────────────────────────────────────
st.sidebar.header("Market Data Options")
ticker     = st.sidebar.text_input("Enter a stock ticker (e.g. AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date",  pd.to_datetime("2024-04-15"))
end_date   = st.sidebar.date_input("End Date",    pd.Timestamp.today())

st.sidebar.markdown("---")
st.sidebar.header("🔔 News Options")
news_days = st.sidebar.slider("Days of news history", 1, 7, 3)
news_qty  = st.sidebar.slider("Max articles to fetch", 10, 100, 30)

if not st.sidebar.button("🔍 Analyze Stock"):
    st.info("👈 Enter a ticker and click **Analyze Stock** to begin.")
    st.stop()

# ── 2. Technical Indicators ──────────────────────────────────────────────────
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error(f"No market data for “{ticker}”")
    st.stop()

# SMA20
df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
# RSI 14
delta     = df["Close"].diff()
gain      = delta.where(delta > 0, 0)
loss      = -delta.where(delta < 0, 0)
avg_gain  = gain.ewm(span=14, adjust=False).mean()
avg_loss  = loss.ewm(span=14, adjust=False).mean()
df["RSI"] = 100 - (100 / (1 + avg_gain/avg_loss))
# MACD & Signal
ema12               = df["Close"].ewm(span=12, adjust=False).mean()
ema26               = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"]          = ema12 - ema26
df["Signal Line"]   = df["MACD"].ewm(span=9, adjust=False).mean()

st.header("🔍 Technical Indicators")
st.subheader("RSI (14 days)")
st.line_chart(df["RSI"])
st.subheader("SMA 20 over Close Price")
st.line_chart(df[["Close", "SMA20"]])
st.subheader("MACD & Signal Line")
st.line_chart(df[["MACD", "Signal Line"]])

st.markdown("---")

# ── 3. News Analysis with NewsAPI ─────────────────────────────────────────────
# 3.1 Inicializa cliente
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # o ponla en st.secrets["NEWSAPI_KEY"]
if not NEWSAPI_KEY:
    st.error("No NewsAPI key found. Add it to your secrets as NEWSAPI_KEY.")
    st.stop()

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# 3.2 Fetch articles
from datetime import datetime, timedelta
to_date   = datetime.utcnow()
from_date = to_date - timedelta(days=news_days)

resp = newsapi.get_everything(
    q=ticker,
    from_param=from_date.strftime("%Y-%m-%d"),
    to=to_date.strftime("%Y-%m-%d"),
    language="en",
    sort_by="relevancy",
    page_size=news_qty,
)

articles = resp.get("articles", [])
if not articles:
    st.warning("No news articles found for that ticker.")
else:
    st.header("📰 News Analysis")
    # 3.3 Mostrar lista de titulares + URLs
    for art in articles:
        published = art["publishedAt"][:10]
        title     = art["title"]
        url       = art["url"]
        st.markdown(f"- **{published}**: [{title}]({url})")

    # (Opcional) puedes contar menciones positivas/negativas si quisieras
